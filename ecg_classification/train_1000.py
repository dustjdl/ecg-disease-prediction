"""Isolated 1,000-record baseline. --validate-only never trains or writes files.

Source NPZ stays unchanged. Only train ECGs 421/710 are excluded after manual
waveform review; validation/test retain their original rows and order.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np
import torch
from torch import nn

from model import ECGModel
from train import make_loader, mean_loss, save_history_csv, save_loss_plot, set_seed

PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "data/processed/ptbxl_100hz_1000records.npz"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RAW_COUNTS = {"train": 800, "validation": 100, "test": 100}
USED_COUNTS = {"train": 798, "validation": 100, "test": 100}
SETTINGS = {"seed": 42, "batch_size": 8, "learning_rate": 0.001, "epochs": 30,
            "optimizer": "Adam", "loss": "MSELoss", "loss_reduction": "mean",
            "checkpoint_selection": "strictly lowest validation MSE",
            "epoch_aggregation": "sum(batch_mean_loss * batch_records) / total_records",
            "train_shuffle": True, "validation_shuffle": False, "num_workers": 0,
            "drop_last": False, "scheduler": None, "early_stopping": False}
PREPROCESSING = {"normalization": "none", "clipping": "none", "filtering": "none",
                 "augmentation": "none", "axis_conversion": "NTC -> NCT", "voltage_unit": "mV"}
REVIEW_NOTE = (
    "로컬 WFDB 원본과 float32 기준으로 일치함. 수동 파형 검토 후 "
    "이번 실험의 학습에서만 제외하며 validation/test는 유지함."
)
EXCLUSION_POLICY = {
    "scope": "train_only", "selection": "explicit ECG IDs after manual waveform review",
    "records": [
        {"ecg_id": 421, "expected_index": 36, "expected_patient_id": 2490,
         "reason": "V4의 국소적인 큰 음의 돌출 및 기준선 변화. " + REVIEW_NOTE},
        {"ecg_id": 710, "expected_index": 249, "expected_patient_id": 5542,
         "reason": "V2 시작 구간의 큰 음의 편차 및 기준선 변화. " + REVIEW_NOTE},
    ],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_id(prefix="run") -> str:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex}"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def model_structure(model: ECGModel) -> dict:
    """Describe executable layers as well as parameter shapes, without inference."""
    return {"class": "model.ECGModel", "representation": repr(model),
            "state_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_source_sha256": sha256_file(PROJECT / "model.py")}


def environment(device: torch.device) -> dict:
    return {"python": platform.python_version(), "pytorch": str(torch.__version__),
            "numpy": str(np.__version__), "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version()}


def load_and_select(dataset_path: Path):
    """Read, validate, and build masks only: no tensors, model, or file writes."""
    digest = sha256_file(dataset_path)
    keys = {"X", "Y", "ecg_id", "patient_id", "strat_fold", "split", "lead_names", "sampling_rate_hz"}
    with np.load(dataset_path, allow_pickle=False) as archive:
        print("Actual NPZ keys:", ", ".join(archive.files), flush=True)
        require(keys.issubset(archive.files), f"Missing keys: {sorted(keys.difference(archive.files))}")
        arrays = {key: archive[key] for key in keys}
    require(sha256_file(dataset_path) == digest, "Dataset changed while being loaded")
    for key, shape in (("X", (1000, 1000, 1)), ("Y", (1000, 1000, 11))):
        value = arrays[key]
        require(value.shape == shape, f"{key} shape: {value.shape}, expected {shape}")
        require(value.dtype == np.float32, f"{key} must be float32")
        require(np.isfinite(value).all(), f"{key} contains NaN/inf")
    for key in ("ecg_id", "patient_id", "strat_fold", "split"):
        require(arrays[key].shape == (1000,), f"Invalid metadata shape: {key}")
    for key in ("ecg_id", "patient_id", "strat_fold"):
        require(np.issubdtype(arrays[key].dtype, np.integer), f"{key} must be integer")
    require(len(np.unique(arrays["ecg_id"])) == 1000, "Duplicate ECG IDs")
    patients, frequencies = np.unique(arrays["patient_id"], return_counts=True)
    require(np.all(frequencies == 1), f"Duplicate patients: {patients[frequencies > 1].tolist()}")
    lead_array = arrays["lead_names"]
    require(lead_array.shape == (12,), "Expected 12 lead names")
    actual_leads = lead_array.astype(str).tolist()
    require(actual_leads == LEADS, f"Invalid lead order: {actual_leads}")
    rate = arrays["sampling_rate_hz"]
    require(rate.shape == () and np.issubdtype(rate.dtype, np.number), "Invalid sampling rate metadata")
    require(float(rate.item()) == 100.0, f"Expected 100 Hz, got {rate}")
    split = arrays["split"].astype(str)
    counts = {name: int(np.count_nonzero(split == name)) for name in RAW_COUNTS}
    require(counts == RAW_COUNTS and set(split) == set(RAW_COUNTS), f"Invalid original splits: {counts}")
    folds = arrays["strat_fold"]
    require(np.isin(folds, np.arange(1, 11)).all(), "Fold outside 1..10")
    expected_split = np.where(folds <= 8, "train", np.where(folds == 9, "validation", "test"))
    require(np.array_equal(split, expected_split), "Official fold/split mapping mismatch")
    masks = {name: split == name for name in RAW_COUNTS}
    excluded = []
    for policy in EXCLUSION_POLICY["records"]:
        indices = np.flatnonzero(arrays["ecg_id"] == policy["ecg_id"])
        require(len(indices) == 1, f"Expected one ECG {policy['ecg_id']}")
        index = int(indices[0])
        require(split[index] == "train", f"ECG {policy['ecg_id']} is not train")
        require(index == policy["expected_index"] and
                int(arrays["patient_id"][index]) == policy["expected_patient_id"],
                f"Reviewed record identity mismatch: ECG {policy['ecg_id']}")
        masks["train"][index] = False
        excluded.append({"ecg_id": policy["ecg_id"], "index": index,
                         "patient_id": int(arrays["patient_id"][index]),
                         "split": str(split[index]), "reason": policy["reason"]})
    used_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    require(used_counts == USED_COUNTS, f"Invalid used splits: {used_counts}")
    for name in ("validation", "test"):
        require(np.array_equal(masks[name], split == name), f"{name} was modified")
    selected_ids = {name: arrays["ecg_id"][mask].tolist() for name, mask in masks.items()}
    description = {"dataset_path": str(dataset_path.resolve()), "dataset_sha256": digest,
                   "original_split_counts": counts, "used_split_counts": used_counts,
                   "split_ecg_ids": selected_ids, "lead_names": actual_leads,
                   "sampling_rate_hz": 100, "excluded_records": excluded,
                   "plot_selection": {"policy": "first test row in original NPZ order, fixed before training",
                                      "ecg_id": selected_ids["test"][0],
                                      "index": int(np.flatnonzero(masks["test"])[0])}}
    print(f"Validation PASS: original={counts}; used={used_counts}; excluded={[r['ecg_id'] for r in excluded]}")
    print(f"Data SHA-256: {digest}; predetermined plot: {description['plot_selection']}")
    return arrays, masks, description


def train_baseline(dataset_path: Path):
    arrays, masks, description = load_and_select(dataset_path)
    set_seed(SETTINGS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    criterion = nn.MSELoss()
    run_id = new_id()
    checkpoint_dir = PROJECT / "checkpoints/baseline_1000" / run_id
    result_dir = PROJECT / "results/baseline_1000" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "run_id": run_id, **description,
                "exclusion_policy": EXCLUSION_POLICY, "settings": SETTINGS,
                "preprocessing": PREPROCESSING, "model_structure": model_structure(model),
                "environment": environment(device),
                "determinism": {"cudnn_deterministic": True, "cudnn_benchmark": False,
                                "seed_scope": "Python, NumPy, torch CPU/CUDA, train loader generator",
                                "limitation": "Bitwise equivalence across devices/versions is not guaranteed"},
                "test_usage": "IDs/counts/integrity validation only; no training inference or model selection",
                "source_sha256": {name: sha256_file(PROJECT / name) for name in
                                  ("train_1000.py", "train.py", "evaluate.py")}}
    write_json(result_dir / "experiment_config.json", metadata)
    with (result_dir / "excluded_records.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "index", "patient_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(description["excluded_records"])
    # Only train/validation waveform subsets are converted to tensors.
    generator = torch.Generator().manual_seed(SETTINGS["seed"])
    train_loader = make_loader(arrays["X"][masks["train"]], arrays["Y"][masks["train"]],
                               SETTINGS["batch_size"], shuffle=True, generator=generator)
    validation_loader = make_loader(arrays["X"][masks["validation"]], arrays["Y"][masks["validation"]],
                                    SETTINGS["batch_size"], shuffle=False)
    del arrays
    print(f"Run: {run_id}; device={device}; settings={SETTINGS}")
    print("Test held out: 100; no test DataLoader or inference during training.")
    history = []
    best_loss = float("inf")
    checkpoint_path = checkpoint_dir / "best_model.pth"
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
                        "learning_rate": SETTINGS["learning_rate"], "experiment": metadata}, checkpoint_path)
    save_history_csv(history, result_dir / "training_history.csv")
    save_loss_plot(history, result_dir / "training_loss.png")
    print(f"Best checkpoint: {checkpoint_path}\nResults: {result_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-only", action="store_true", help="Validate data/masks only; no training or output writes.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.validate_only:
        load_and_select(args.dataset)
    else:
        train_baseline(args.dataset)


if __name__ == "__main__":
    main()
