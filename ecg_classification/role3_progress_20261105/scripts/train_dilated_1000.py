"""Train a separately initialized dilated baseline; --validate-only writes nothing.

Only train/validation waveforms enter loaders. Test IDs are integrity metadata
only. Existing baseline files and datasets are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ECGModel
from model_dilated import ECGDilatedModel
from evaluate_1000 import validate_experiment
from train import make_loader, mean_loss, save_history_csv, save_loss_plot, set_seed
from train_1000 import (DATASET, EXCLUSION_POLICY, PREPROCESSING, PROJECT, SETTINGS,
                        environment, load_and_select, new_id, require, sha256_file, write_json)


def structure(model):
    """Describe the actual class, source, layers, and theoretical receptive field."""
    require(type(model) in (ECGModel, ECGDilatedModel), "Unsupported model class")
    source = "model.py" if type(model) is ECGModel else "model_dilated.py"
    layers, receptive, jump = [], 1, 1
    for name, layer in model.network.named_children():
        if isinstance(layer, nn.Conv1d):
            receptive += (layer.kernel_size[0] - 1) * layer.dilation[0] * jump
            jump *= layer.stride[0]
            layers.append({"name": name, "type": "Conv1d", "in_channels": layer.in_channels,
                           "out_channels": layer.out_channels, "kernel_size": list(layer.kernel_size),
                           "stride": list(layer.stride), "dilation": list(layer.dilation),
                           "padding": list(layer.padding), "bias": layer.bias is not None,
                           "groups": layer.groups, "padding_mode": layer.padding_mode})
        else:
            require(type(layer) is nn.ReLU, "Unexpected non-convolution layer")
            layers.append({"name": name, "type": "ReLU", "inplace": layer.inplace})
    return {"class": f"{type(model).__module__}.{type(model).__name__}",
            "representation": repr(model), "model_source_sha256": sha256_file(PROJECT / source),
            "state_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
            "parameter_count": sum(p.numel() for p in model.parameters()), "layers": layers,
            "receptive_field_samples": receptive, "receptive_field_ms_at_100hz": receptive * 10,
            "first_to_last_sample_ms_at_100hz": (receptive - 1) * 10}


def parameter_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialization_alignment():
    """Use constructors only, never a trained state or ECG forward pass."""
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SETTINGS["seed"])
        baseline = ECGModel()
        torch.manual_seed(SETTINGS["seed"])
        candidate = ECGDilatedModel()
        equal = all(torch.equal(value, candidate.state_dict()[key])
                    for key, value in baseline.state_dict().items())
        require(equal, "Fresh baseline/candidate initial weights differ")
        # DataLoader consumes a base seed before sampling. Iterate a dummy ID
        # loader to include that step; this is not waveform/model inference.
        generator = torch.Generator().manual_seed(SETTINGS["seed"])
        loader = DataLoader(TensorDataset(torch.arange(798)), batch_size=8,
                            shuffle=True, num_workers=0, generator=generator)
        order = torch.cat([batch[0] for batch in loader]).tolist()
    return {"fresh_constructor_weights_equal": equal,
            "initial_state_sha256": parameter_digest(candidate),
            "first_epoch_record_positions": order,
            "order": "load/validate baseline, set_seed(42), construct model, Adam/MSE, metadata, generator(42), train loader, validation loader",
            "limitation": "Historical baseline initial tensors and batch orders were not saved; equality is verified for fresh constructors in the current environment only. Cross-version/device bitwise equivalence is not guaranteed."}


def validate_candidate(checkpoint, model, description):
    require(type(model) is ECGDilatedModel, "Candidate requires ECGDilatedModel")
    metadata = checkpoint.get("experiment")
    require(isinstance(metadata, dict), "Missing candidate experiment metadata")
    require(metadata.get("schema_version") == 1 and metadata.get("experiment_type") == "dilated_1000",
            "Not a dilated_1000 experiment")
    require(re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{32}", str(metadata.get("run_id", ""))) is not None,
            "Invalid candidate run ID")
    require(metadata.get("model_structure") == structure(model), "Candidate structure/source mismatch (including dilation/padding)")
    for key in ("dataset_sha256", "original_split_counts", "used_split_counts", "split_ecg_ids",
                "excluded_records", "lead_names", "sampling_rate_hz", "plot_selection"):
        require(metadata.get(key) == description[key], f"Candidate/data mismatch: {key}")
    for key, expected in (("exclusion_policy", EXCLUSION_POLICY), ("preprocessing", PREPROCESSING),
                          ("settings", SETTINGS)):
        require(metadata.get(key) == expected, f"Candidate mismatch: {key}")
    require(isinstance(metadata.get("baseline_reference"), dict), "Missing baseline reference")
    for key in ("seed", "batch_size", "learning_rate"):
        require(checkpoint.get(key) == SETTINGS[key], f"Candidate setting mismatch: {key}")
    require(isinstance(checkpoint.get("epoch"), int) and 1 <= checkpoint["epoch"] <= 30, "Invalid epoch")
    require(isinstance(checkpoint.get("validation_loss"), (float, int)) and
            np.isfinite(checkpoint["validation_loss"]) and checkpoint["validation_loss"] >= 0,
            "Invalid validation loss")
    return metadata


def load_checked_checkpoint(path, role, description):
    """CPU load, exact architecture metadata, exact state keys/shapes; no inference."""
    require(role in ("baseline", "candidate"), "Invalid checkpoint role")
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(sha256_file(path) == digest, "Checkpoint changed while loading")
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary")
    model = ECGModel() if role == "baseline" else ECGDilatedModel()
    if role == "baseline":
        metadata = validate_experiment(checkpoint, model, description)
        # Baseline schema predates the explicit layers list; repr includes dilation
        # and padding. Its exact repr + source hash are checked by the existing validator.
        source_matches = {name: metadata.get("source_sha256", {}).get(name) == sha256_file(PROJECT / name)
                          for name in ("train_1000.py", "train.py", "evaluate.py")}
        require(all(source_matches.values()), f"Baseline source provenance mismatch: {source_matches}")
    else:
        metadata = validate_candidate(checkpoint, model, description)
    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict), "Missing model_state_dict")
    expected = model.state_dict()
    require(set(state) == set(expected), "State dictionary keys differ")
    for key in expected:
        require(isinstance(state[key], torch.Tensor) and state[key].shape == expected[key].shape,
                f"State shape mismatch: {key}")
        require(state[key].dtype == expected[key].dtype and torch.isfinite(state[key]).all().item(),
                f"Invalid state dtype/value: {key}")
    model.load_state_dict(state, strict=True)
    print(f"{role} checkpoint metadata/state PASS: {path}")
    return model, checkpoint, {"path": str(path.resolve()), "sha256": digest, "run_id": metadata["run_id"]}


def prepare(args):
    arrays, masks, description = load_and_select(args.dataset)
    baseline_model, baseline, reference = load_checked_checkpoint(args.baseline_checkpoint, "baseline", description)
    del baseline_model
    alignment = initialization_alignment()
    current = environment(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    alignment["baseline_environment"] = baseline["experiment"]["environment"]
    alignment["environment_matches"] = baseline["experiment"]["environment"] == current
    print("Fresh initial weights equal; train seed/generator/creation order preserved.")
    print("Initialization limitation:", alignment["limitation"])
    return arrays, masks, description, reference, alignment


def train_candidate(args):
    arrays, masks, description, reference, alignment = prepare(args)
    # Everything consuming random numbers above precedes this final seed reset.
    set_seed(SETTINGS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGDilatedModel().to(device)  # Never load baseline trained weights.
    require(parameter_digest(model) == alignment["initial_state_sha256"], "Actual candidate initialization differs")
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    criterion = nn.MSELoss()
    run_id = new_id()
    checkpoint_dir = PROJECT / "checkpoints/dilated_1000" / run_id
    result_dir = PROJECT / "results/dilated_1000" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "experiment_type": "dilated_1000", "run_id": run_id,
                **description, "baseline_reference": reference, "exclusion_policy": EXCLUSION_POLICY,
                "settings": SETTINGS, "preprocessing": PREPROCESSING, "model_structure": structure(model),
                "environment": environment(device), "initialization_alignment": alignment,
                "determinism": {"cudnn_deterministic": True, "cudnn_benchmark": False},
                "test_usage": "Integrity metadata only; no test loader, inference, metrics or selection",
                "source_sha256": {name: sha256_file(PROJECT / name) for name in
                                  ("model_dilated.py", "train_dilated_1000.py", "train_1000.py", "train.py")}}
    require(metadata["model_structure"]["parameter_count"] == 31819, "Unexpected parameter count")
    write_json(result_dir / "experiment_config.json", metadata)
    with (result_dir / "excluded_records.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "index", "patient_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(description["excluded_records"])
    generator = torch.Generator().manual_seed(SETTINGS["seed"])
    train_loader = make_loader(arrays["X"][masks["train"]], arrays["Y"][masks["train"]],
                               SETTINGS["batch_size"], shuffle=True, generator=generator)
    validation_loader = make_loader(arrays["X"][masks["validation"]], arrays["Y"][masks["validation"]],
                                    SETTINGS["batch_size"], shuffle=False)
    del arrays
    history, best_loss = [], float("inf")
    for epoch in range(1, SETTINGS["epochs"] + 1):
        train_loss = mean_loss(model, train_loader, criterion, device, optimizer=optimizer)
        validation_loss = mean_loss(model, validation_loader, criterion, device)
        require(np.isfinite(train_loss) and np.isfinite(validation_loss), "Nonfinite epoch loss")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch:02d}/30 | train={train_loss:.6f} | validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "validation_loss": best_loss,
                        "seed": SETTINGS["seed"], "batch_size": SETTINGS["batch_size"],
                        "learning_rate": SETTINGS["learning_rate"], "experiment": metadata},
                       checkpoint_dir / "best_model.pth")
    save_history_csv(history, result_dir / "training_history.csv")
    save_loss_plot(history, result_dir / "training_loss.png")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pth'}\nResults: {result_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-only", action="store_true", help="Check data, baseline and initialization only; no training/inference/writes.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.validate_only:
        prepare(args)
        print("Validation only complete; no training or waveform inference.")
    else:
        train_candidate(args)


if __name__ == "__main__":
    main()
