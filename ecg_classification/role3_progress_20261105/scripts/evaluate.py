"""Evaluate the saved baseline ECGModel on the held-out PTB-XL test split only.

This script performs inference only. It never calls backward(), optimizer.step(),
or uses the test split to choose a model or hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ECGModel


# Save images to a file rather than opening an interactive plotting window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_DATASET_KEYS = {"X", "Y", "split", "lead_names", "sampling_rate_hz"}
EXPECTED_TEST_RECORDS = 10
VOLTAGE_UNIT = "mV"


def ensure_outputs_are_safe(paths: tuple[Path, Path], overwrite: bool) -> None:
    """Refuse to replace a prior evaluation result unless --overwrite is explicit."""
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        joined_paths = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Existing evaluation outputs were found. Inspect them first, then rerun with --overwrite "
            f"only if replacement is intended:\n{joined_paths}"
        )


def load_compatible_model(checkpoint_path: Path, device: torch.device) -> tuple[ECGModel, Mapping[str, Any]]:
    """Inspect checkpoint contents and require an exact ECGModel state_dict match."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions without weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a dictionary containing model_state_dict")
    print("Checkpoint keys:", sorted(checkpoint.keys()))
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise TypeError("model_state_dict is not a state dictionary")

    model = ECGModel()
    expected_state = model.state_dict()
    saved_state = checkpoint["model_state_dict"]
    missing_keys = sorted(set(expected_state).difference(saved_state))
    unexpected_keys = sorted(set(saved_state).difference(expected_state))
    shape_mismatches = {
        key: (tuple(expected_state[key].shape), tuple(saved_state[key].shape))
        for key in expected_state.keys() & saved_state.keys()
        if expected_state[key].shape != saved_state[key].shape
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "Checkpoint is not compatible with the current ECGModel. "
            f"missing={missing_keys}, unexpected={unexpected_keys}, shape_mismatches={shape_mismatches}"
        )

    model.load_state_dict(saved_state, strict=True)
    model.to(device)
    model.eval()
    print("Checkpoint compatibility: PASS (all state_dict keys and tensor shapes match ECGModel)")
    return model, checkpoint


def load_test_data(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Load exactly the held-out test rows and convert them to channels-first arrays."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing_keys = REQUIRED_DATASET_KEYS.difference(archive.files)
        if missing_keys:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing_keys)}")
        x_data = archive["X"]
        y_data = archive["Y"]
        split_labels = archive["split"].astype(str)
        lead_names = tuple(archive["lead_names"].astype(str).tolist())
        sampling_rate = int(np.asarray(archive["sampling_rate_hz"]).item())

    if x_data.dtype != np.float32 or y_data.dtype != np.float32:
        raise TypeError(f"X/Y must be float32, got X={x_data.dtype}, Y={y_data.dtype}")
    if x_data.ndim != 3 or x_data.shape[1:] != (1000, 1):
        raise ValueError(f"X must have shape [records, 1000, 1], got {x_data.shape}")
    if y_data.ndim != 3 or y_data.shape[1:] != (1000, 11):
        raise ValueError(f"Y must have shape [records, 1000, 11], got {y_data.shape}")
    if len(x_data) != len(y_data) or len(x_data) != len(split_labels):
        raise ValueError("X, Y, and split arrays must have the same record count")
    if not np.isfinite(x_data).all() or not np.isfinite(y_data).all():
        raise ValueError("X or Y contains NaN or infinity")
    if len(lead_names) != 12 or "I" not in lead_names:
        raise ValueError(f"Invalid lead_names: {lead_names}")

    split_counts = {name: int(np.sum(split_labels == name)) for name in ("train", "validation", "test")}
    if split_counts != {"train": 80, "validation": 10, "test": 10}:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    test_mask = split_labels == "test"
    if int(test_mask.sum()) != EXPECTED_TEST_RECORDS:
        raise ValueError(f"Expected {EXPECTED_TEST_RECORDS} test records, got {int(test_mask.sum())}")

    # Only these 10 rows are returned. Train/validation waveform values are never used below.
    x_test = x_data[test_mask]
    y_test = y_data[test_mask]
    x_channels_first = np.ascontiguousarray(np.transpose(x_test, (0, 2, 1)))
    y_channels_first = np.ascontiguousarray(np.transpose(y_test, (0, 2, 1)))
    print(f"Split counts: {split_counts}; evaluating test records only: {len(x_test)}")
    print("Voltage unit: mV (the prepared dataset was made from WFDB record.p_signal physical signals)")
    return x_channels_first, y_channels_first, lead_names, sampling_rate


def run_inference(model: ECGModel, x_test: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Run test-only batched inference with eval mode and inference_mode."""
    test_dataset = TensorDataset(torch.from_numpy(x_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (inputs,) in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            if outputs.shape != (inputs.shape[0], 11, 1000):
                raise RuntimeError(f"Unexpected model output shape: {outputs.shape}")
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def pearson_correlation(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, str]:
    """Return lead correlation; label a zero-variance signal instead of dividing by zero."""
    actual_std = float(np.std(actual))
    predicted_std = float(np.std(predicted))
    if actual_std == 0.0 and predicted_std == 0.0:
        return float("nan"), "zero_variance_actual_and_prediction"
    if actual_std == 0.0:
        return float("nan"), "zero_variance_actual"
    if predicted_std == 0.0:
        return float("nan"), "zero_variance_prediction"
    return float(np.corrcoef(actual, predicted)[0, 1]), "ok"


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, lead_names: tuple[str, ...]) -> list[dict[str, str | float | int]]:
    """Calculate overall and per-target-lead MAE, RMSE, and Pearson correlation."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}")

    target_lead_names = [lead for lead in lead_names if lead != "I"]
    if len(target_lead_names) != 11:
        raise ValueError(f"Expected 11 target leads, got {target_lead_names}")

    errors = predictions - targets
    overall_mae = float(np.mean(np.abs(errors)))
    overall_rmse = float(np.sqrt(np.mean(np.square(errors))))
    overall_correlation, overall_status = pearson_correlation(targets.ravel(), predictions.ravel())
    rows: list[dict[str, str | float | int]] = [
        {
            "scope": "overall",
            "lead": "ALL_11_TARGET_LEADS",
            "mae_mV": overall_mae,
            "rmse_mV": overall_rmse,
            "pearson_correlation": overall_correlation,
            "correlation_status": overall_status,
            "records": targets.shape[0],
            "samples_per_record": targets.shape[2],
            "voltage_unit": VOLTAGE_UNIT,
        }
    ]

    for lead_index, lead_name in enumerate(target_lead_names):
        actual = targets[:, lead_index, :].ravel()
        predicted = predictions[:, lead_index, :].ravel()
        correlation, status = pearson_correlation(actual, predicted)
        rows.append(
            {
                "scope": "per_lead",
                "lead": lead_name,
                "mae_mV": float(np.mean(np.abs(predicted - actual))),
                "rmse_mV": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "pearson_correlation": correlation,
                "correlation_status": status,
                "records": targets.shape[0],
                "samples_per_record": targets.shape[2],
                "voltage_unit": VOLTAGE_UNIT,
            }
        )
    return rows


def save_metrics_csv(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    """Write one overall row and 11 per-lead rows to a readable CSV file."""
    fieldnames = [
        "scope", "lead", "mae_mV", "rmse_mV", "pearson_correlation",
        "correlation_status", "records", "samples_per_record", "voltage_unit",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_first_comparison_plot(
    x_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    lead_names: tuple[str, ...],
    sampling_rate: int,
    output_path: Path,
) -> None:
    """Plot the first test ECG: actual Lead I input and actual/predicted target leads."""
    time_seconds = np.arange(x_test.shape[2]) / sampling_rate
    figure, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    figure.suptitle("First held-out test ECG: input Lead I and 11-lead prediction comparison", fontsize=15)

    target_index = 0
    for axis, lead_name in zip(axes, lead_names, strict=True):
        if lead_name == "I":
            # Lead I is the observed input, not a prediction.
            axis.plot(time_seconds, x_test[0, 0, :], color="tab:blue", linewidth=0.9, label="Actual input (Lead I)")
        else:
            axis.plot(time_seconds, y_test[0, target_index, :], color="black", linewidth=0.9, label="Actual target")
            axis.plot(time_seconds, predictions[0, target_index, :], color="tab:red", linewidth=0.8, alpha=0.8, label="Prediction")
            target_index += 1
        axis.set_ylabel(f"{lead_name}\n({VOLTAGE_UNIT})", rotation=0, labelpad=28, va="center")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    if target_index != 11:
        raise ValueError(f"Expected to plot 11 target leads, plotted {target_index}")
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved ECGModel on PTB-XL test records only.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "processed" / "ptbxl_100hz_100records.npz",
        help="Prepared PTB-XL NPZ file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "best_model.pth",
        help="Best validation checkpoint created by train.py.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (default: 8).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing test metric and comparison files.",
    )
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    metrics_path = Path("results") / "test_metrics.csv"
    comparison_plot_path = Path("results") / "test_prediction_comparison.png"
    ensure_outputs_are_safe((metrics_path, comparison_plot_path), arguments.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Compatibility is validated before any ECG data is sent through the model.
    model, checkpoint = load_compatible_model(arguments.checkpoint, device)
    print(f"Checkpoint best epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(f"Checkpoint validation loss: {checkpoint.get('validation_loss', 'not recorded')}")
    print("Normalization/signal transform: none in train.py; only [time, channel] -> [channel, time] axis conversion is applied.")

    x_test, y_test, lead_names, sampling_rate = load_test_data(arguments.dataset)
    predictions = run_inference(model, x_test, arguments.batch_size, device)
    metric_rows = calculate_metrics(predictions, y_test, lead_names)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(metric_rows, metrics_path)
    save_first_comparison_plot(x_test, y_test, predictions, lead_names, sampling_rate, comparison_plot_path)

    overall = metric_rows[0]
    print("\n===== Held-out test evaluation complete =====")
    print(f"Overall MAE: {overall['mae_mV']:.6f} {VOLTAGE_UNIT}")
    print(f"Overall RMSE: {overall['rmse_mV']:.6f} {VOLTAGE_UNIT}")
    print(f"Overall Pearson correlation: {overall['pearson_correlation']:.6f} ({overall['correlation_status']})")
    print(f"Metrics CSV: {metrics_path.resolve()}")
    print(f"Comparison plot: {comparison_plot_path.resolve()}")
    print("No test record was used for additional training or model selection.")


if __name__ == "__main__":
    main()
