"""Shared multiseed protocol; original scripts and legacy validators are unchanged.

No import-time model construction, dataset loading, training or inference.
New checkpoints retain prediction schema 1 and add training_protocol metadata.
"""

from __future__ import annotations

import argparse
import copy
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
from train import make_loader, mean_loss, save_history_csv, save_loss_plot, set_seed
from train_1000 import (DATASET, EXCLUSION_POLICY, PREPROCESSING, PROJECT, SETTINGS,
                        environment, load_and_select, new_id, require, sha256_file, write_json)
from train_dilated_1000 import parameter_digest, structure, load_checked_checkpoint as load_legacy

PROTOCOL = "multiseed_v1"
SEED_MAX = 2**32 - 1  # Intersection: np.random.seed legacy API and torch.manual_seed.
SOURCE_FILES = (
    "multiseed_common.py", "train_baseline_multiseed.py", "train_dilated_multiseed.py",
    "compare_validation_multiseed.py", "model.py", "model_dilated.py", "train.py",
    "train_1000.py", "train_dilated_1000.py", "evaluate.py", "evaluate_1000.py",
    "compare_validation_1000.py",
)


def seed_argument(text):
    try:
        seed = int(text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if not 0 <= seed <= SEED_MAX:
        raise argparse.ArgumentTypeError(f"seed must be in 0..{SEED_MAX} (NumPy/PyTorch intersection)")
    return seed


def run_settings(seed):
    require(type(seed) is int and 0 <= seed <= SEED_MAX, "Invalid seed")
    settings = copy.deepcopy(SETTINGS)
    settings["seed"] = seed
    return settings


def source_manifest():
    return {name: sha256_file(PROJECT / name) for name in SOURCE_FILES}


def initial_alignment(seed):
    """Reconstruct fresh constructors in isolated torch RNG state; no forward."""
    run_settings(seed)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        baseline = ECGModel()
        torch.manual_seed(seed)
        candidate = ECGDilatedModel()
        require(all(torch.equal(value, candidate.state_dict()[key])
                    for key, value in baseline.state_dict().items()), "Fresh initialization differs")
        digest = parameter_digest(baseline)
    return {"seed": seed, "fresh_constructor_weights_equal": True,
            "expected_initial_state_sha256": digest,
            "hash_format": "state_dict iteration: UTF-8 key followed by contiguous CPU tensor bytes",
            "limitation": "Bitwise equivalence across devices/library versions is not guaranteed; actual initial state hashes are checked."}


def order_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


class RecordedTrainBatches:
    """Observe IDs in consumed batches; yield the original two loss arguments.

    No sampler preview, extra iterator, random call, or tensor math on X/Y.
    num_workers=0 means these are the batches consumed by mean_loss itself.
    """

    def __init__(self, loader, expected_ids):
        self.loader = loader
        self.expected_ids = [int(value) for value in expected_ids]
        self.observed = None
        self.complete = False

    def __iter__(self):
        self.observed, self.complete = [], False
        for inputs, targets, ids in self.loader:
            self.observed.extend(ids.tolist())
            yield inputs, targets
        self.complete = True

    def finish_epoch(self, epoch):
        require(self.complete and self.observed is not None, "Training iterator was not fully consumed")
        require(sorted(self.observed) == sorted(self.expected_ids), "Training epoch IDs are not the expected permutation")
        return {"epoch": epoch, "records": len(self.observed),
                "ecg_id_order_sha256": order_digest(self.observed), "ecg_ids": self.observed.copy()}


def make_recorded_loader(x, y, ecg_ids, settings):
    tensors = (torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))),
               torch.from_numpy(np.ascontiguousarray(y.transpose(0, 2, 1))),
               torch.tensor(ecg_ids, dtype=torch.int64))
    require(len(x) == len(y) == len(ecg_ids), "Train array/ID lengths differ")
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(TensorDataset(*tensors), batch_size=settings["batch_size"], shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
                        drop_last=False)
    return RecordedTrainBatches(loader, ecg_ids)


def validate_orders(orders, train_ids, expected_epochs):
    require(isinstance(orders, list) and len(orders) == expected_epochs, "Incomplete epoch order log")
    for epoch, row in enumerate(orders, 1):
        require(row.get("epoch") == epoch and row.get("records") == len(train_ids), "Invalid order log epoch/count")
        ids = row.get("ecg_ids")
        require(isinstance(ids, list) and all(type(value) is int for value in ids)
                and sorted(ids) == sorted(train_ids), "Invalid epoch ECG ID permutation")
        require(row.get("ecg_id_order_sha256") == order_digest(ids), "Epoch order hash mismatch")


def validate_checkpoint(checkpoint, role, description, model, expected_seed=None):
    """Strict new protocol only. Legacy files use their original validator."""
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary")
    metadata = checkpoint.get("experiment")
    require(isinstance(metadata, dict) and metadata.get("training_protocol") == PROTOCOL,
            "A new multiseed checkpoint is required")
    require(role in ("baseline", "candidate"), "Invalid role")
    expected_class = ECGModel if role == "baseline" else ECGDilatedModel
    require(type(model) is expected_class, "Wrong runtime model class")
    require(metadata.get("schema_version") == 1 and metadata.get("model_role") == role,
            "Wrong checkpoint schema/role")
    require(metadata.get("experiment_type") == ("baseline_1000" if role == "baseline" else "dilated_1000"),
            "Wrong architecture experiment_type")
    require(re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{32}", str(metadata.get("run_id", ""))) is not None,
            "Invalid run ID")
    settings = metadata.get("settings")
    require(isinstance(settings, dict), "Missing settings")
    seed = settings.get("seed")
    require(settings == run_settings(seed), "Non-seed training settings changed")
    require(expected_seed is None or seed == expected_seed, "Requested/baseline seed mismatch")
    for key in ("seed", "batch_size", "learning_rate"):
        require(checkpoint.get(key) == settings[key], f"Top-level/metadata setting mismatch: {key}")
    for key in ("dataset_sha256", "original_split_counts", "used_split_counts", "split_ecg_ids",
                "excluded_records", "lead_names", "sampling_rate_hz", "plot_selection"):
        require(metadata.get(key) == description[key], f"Checkpoint/data mismatch: {key}")
    require(metadata.get("exclusion_policy") == EXCLUSION_POLICY, "Exclusion policy mismatch")
    require(metadata.get("preprocessing") == PREPROCESSING, "Preprocessing mismatch")
    require(metadata.get("model_structure") == structure(model), "Model structure/source mismatch")
    # Exact manifests retain provenance checks for both new scripts and reused
    # legacy code. Nothing substitutes or edits the historical source hashes.
    require(metadata.get("source_sha256") == source_manifest(), "Source provenance mismatch")
    alignment = initial_alignment(seed)
    require(metadata.get("initialization_alignment") == alignment, "Initialization metadata mismatch")
    require(metadata.get("actual_initial_state_sha256") == alignment["expected_initial_state_sha256"],
            "Actual initial weight hash mismatch")
    reference = metadata.get("baseline_reference")
    if role == "candidate":
        require(isinstance(reference, dict) and reference.get("seed") == seed
                and reference.get("actual_initial_state_sha256") == metadata["actual_initial_state_sha256"]
                and isinstance(reference.get("run_id"), str)
                and re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{32}", reference["run_id"]) is not None
                and isinstance(reference.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None,
                "Invalid candidate baseline reference")
    else:
        require(reference is None, "Baseline cannot reference another baseline")
    require(metadata.get("training_complete") is True, "Checkpoint training is incomplete")
    validate_orders(metadata.get("train_epoch_orders"), description["split_ecg_ids"]["train"], settings["epochs"])
    epoch, loss = checkpoint.get("epoch"), checkpoint.get("validation_loss")
    require(type(epoch) is int and 1 <= epoch <= settings["epochs"], "Invalid best epoch")
    require(isinstance(loss, (float, int)) and np.isfinite(loss) and loss >= 0, "Invalid validation loss")
    state, expected = checkpoint.get("model_state_dict"), model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected), "State keys mismatch")
    for key, target in expected.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.shape == target.shape
                and value.dtype == target.dtype and torch.isfinite(value).all().item(),
                f"Invalid state: {key}")
    return metadata


def load_checkpoint(path, role, description, expected_seed=None, allow_legacy=False):
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(sha256_file(path) == digest, "Checkpoint changed while loading")
    metadata = checkpoint.get("experiment", {}) if isinstance(checkpoint, dict) else {}
    if allow_legacy and isinstance(metadata, dict) and "training_protocol" not in metadata:
        model, checkpoint, reference = load_legacy(path, role, description)
        require(reference["sha256"] == digest, "Legacy checkpoint changed while loading")
        require(expected_seed is None or checkpoint["seed"] == expected_seed, "Legacy seed mismatch")
        return model, checkpoint, reference
    model = ECGModel() if role == "baseline" else ECGDilatedModel()
    metadata = validate_checkpoint(checkpoint, role, description, model, expected_seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint, {"path": str(path.resolve()), "sha256": digest, "run_id": metadata["run_id"]}


def validate_pair(baseline, candidate, baseline_reference):
    left, right = baseline["experiment"], candidate["experiment"]
    require(left.get("training_protocol") == right.get("training_protocol"), "Cannot mix legacy/new protocols")
    require(baseline["seed"] == candidate["seed"] and left["settings"] == right["settings"],
            "Pair seed/settings mismatch")
    for key in ("dataset_sha256", "split_ecg_ids", "exclusion_policy", "preprocessing"):
        require(left[key] == right[key], f"Pair mismatch: {key}")
    reference = right.get("baseline_reference", {})
    require(isinstance(reference, dict) and reference.get("sha256") == baseline_reference["sha256"] and
            reference.get("run_id") == baseline_reference["run_id"], "Wrong baseline reference")
    if left.get("training_protocol") == PROTOCOL:
        require(left["actual_initial_state_sha256"] == right["actual_initial_state_sha256"],
                "Pair initial weights differ")
        require(left["train_epoch_orders"] == right["train_epoch_orders"], "Pair observed epoch orders differ")
        require(reference.get("seed") == baseline["seed"] and
                reference.get("actual_initial_state_sha256") == left["actual_initial_state_sha256"],
                "Baseline reference seed/initialization mismatch")


def training_prepare(args, role):
    settings = run_settings(args.seed)
    arrays, masks, description = load_and_select(args.dataset)
    baseline, reference = None, None
    if role == "candidate":
        model, baseline, reference = load_checkpoint(args.baseline_checkpoint, "baseline", description,
                                                     expected_seed=args.seed, allow_legacy=False)
        del model
        reference.update(seed=args.seed, actual_initial_state_sha256=baseline["experiment"]["actual_initial_state_sha256"])
    alignment = initial_alignment(args.seed)
    if baseline is not None:
        require(baseline["experiment"]["actual_initial_state_sha256"] == alignment["expected_initial_state_sha256"],
                "Fresh candidate differs from actual baseline initialization")
    print(f"Preparation PASS: role={role}, seed={args.seed}, splits=798/100/100; fresh initial weights match")
    return arrays, masks, description, settings, alignment, baseline, reference


def train_run(args, role):
    arrays, masks, description, settings, alignment, baseline, reference = training_prepare(args, role)
    if args.validate_only:
        print("Validation only: no training, waveform inference or output writes.")
        return
    set_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (ECGModel() if role == "baseline" else ECGDilatedModel()).to(device)
    actual_initial_hash = parameter_digest(model)
    require(actual_initial_hash == alignment["expected_initial_state_sha256"], "Actual initialization changed")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.MSELoss()
    run_id = new_id()
    group = "baseline_multiseed" if role == "baseline" else "dilated_multiseed"
    checkpoint_dir, result_dir = PROJECT / "checkpoints" / group / run_id, PROJECT / "results" / group / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "training_protocol": PROTOCOL, "model_role": role,
                "experiment_type": "baseline_1000" if role == "baseline" else "dilated_1000",
                "run_id": run_id, **description, "settings": settings,
                "exclusion_policy": copy.deepcopy(EXCLUSION_POLICY), "preprocessing": copy.deepcopy(PREPROCESSING),
                "model_structure": structure(model), "source_sha256": source_manifest(),
                "environment": environment(device), "initialization_alignment": alignment,
                "actual_initial_state_sha256": actual_initial_hash, "baseline_reference": reference,
                "train_epoch_orders": [], "training_complete": False,
                "order_hash_format": "SHA-256 of consumed ECG IDs as contiguous little-endian int64 bytes",
                "determinism": {"cudnn_deterministic": True, "cudnn_benchmark": False},
                "test_usage": "Integrity metadata only; no test DataLoader, inference, metrics or selection"}
    write_json(result_dir / "experiment_config.json", metadata)
    with (result_dir / "excluded_records.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "index", "patient_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(description["excluded_records"])
    train_loader = make_recorded_loader(arrays["X"][masks["train"]], arrays["Y"][masks["train"]],
                                        description["split_ecg_ids"]["train"], settings)
    validation_loader = make_loader(arrays["X"][masks["validation"]], arrays["Y"][masks["validation"]],
                                    settings["batch_size"], shuffle=False)
    del arrays
    history, orders, best_loss = [], [], float("inf")
    checkpoint_path = checkpoint_dir / "best_model.pth"
    with (result_dir / "train_epoch_orders.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "records", "ecg_id_order_sha256"])
        writer.writeheader()
        for epoch in range(1, settings["epochs"] + 1):
            train_loss = mean_loss(model, train_loader, criterion, device, optimizer=optimizer)
            observed = train_loader.finish_epoch(epoch)
            orders.append(observed)
            writer.writerow({key: observed[key] for key in writer.fieldnames})
            stream.flush()
            if baseline is not None:
                require(observed == baseline["experiment"]["train_epoch_orders"][epoch-1],
                        f"Actual candidate/baseline training order differs at epoch {epoch}")
            validation_loss = mean_loss(model, validation_loader, criterion, device)
            require(np.isfinite(train_loss) and np.isfinite(validation_loss), "Nonfinite loss")
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
            print(f"Seed {args.seed} epoch {epoch:02d}/30 | train={train_loss:.6f} | validation={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                snapshot = copy.deepcopy(metadata)
                snapshot["train_epoch_orders"] = copy.deepcopy(orders)
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "validation_loss": best_loss,
                            "seed": args.seed, "batch_size": settings["batch_size"],
                            "learning_rate": settings["learning_rate"], "experiment": snapshot}, checkpoint_path)
    # Preserve best weights/optimizer while attaching all 30 actually consumed
    # epoch orders. Incomplete runs cannot be accepted as a new baseline.
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best["experiment"]["train_epoch_orders"] = copy.deepcopy(orders)
    best["experiment"]["training_complete"] = True
    torch.save(best, checkpoint_path)
    write_json(result_dir / "completed_experiment.json", best["experiment"])
    save_history_csv(history, result_dir / "training_history.csv")
    save_loss_plot(history, result_dir / "training_loss.png")
    print(f"Best checkpoint: {checkpoint_path}\nResults: {result_dir}")


def training_parser(role, argv=None):
    parser = argparse.ArgumentParser(description=f"Train {role} multiseed baseline; test is never inferred.")
    parser.add_argument("--seed", type=seed_argument, default=42, help=f"Integer 0..{SEED_MAX}; default 42.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-only", action="store_true", help="Validate data/metadata/constructors only; no training or writes.")
    if role == "candidate":
        parser.add_argument("--baseline-checkpoint", type=Path, required=True,
                            help="Completed same-seed checkpoint from train_baseline_multiseed.py.")
    return parser.parse_args(argv)
