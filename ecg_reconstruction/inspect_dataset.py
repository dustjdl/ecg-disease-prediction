"""Inspect the prepared PTB-XL dataset without training an AI model.

The script reads the actual keys stored in the NPZ file, validates the dataset,
and saves a 12-lead plot reconstructed from the first X/Y pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

# A file is saved instead of opening an interactive plot window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_KEYS = {
    "X",
    "Y",
    "ecg_id",
    "patient_id",
    "strat_fold",
    "split",
    "lead_names",
    "sampling_rate_hz",
}
EXPECTED_SPLIT_COUNTS = {"train": 80, "validation": 10, "test": 10}


def print_check(name: str, passed: bool, detail: str) -> bool:
    """Print one easy-to-read inspection result and return its status."""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return passed


def load_npz(dataset_path: Path) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Load every array after first reporting the actual saved key names."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as archive:
        actual_keys = tuple(archive.files)
        print("NPZ keys:", ", ".join(actual_keys))
        missing_keys = REQUIRED_KEYS.difference(actual_keys)
        if missing_keys:
            raise KeyError(f"Required NPZ keys are missing: {sorted(missing_keys)}")
        arrays = {key: archive[key] for key in actual_keys}
    return arrays, actual_keys


def inspect_arrays(arrays: dict[str, np.ndarray]) -> bool:
    """Check shapes, dtypes, split sizes, patient leakage, and finite values."""
    x_data = arrays["X"]
    y_data = arrays["Y"]
    ecg_id = arrays["ecg_id"]
    patient_id = arrays["patient_id"]
    strat_fold = arrays["strat_fold"]
    splits = arrays["split"].astype(str)

    record_count = len(ecg_id)
    results: list[bool] = []
    results.append(
        print_check(
            "X shape and dtype",
            x_data.shape == (record_count, 1000, 1) and x_data.dtype == np.float32,
            f"shape={x_data.shape}, dtype={x_data.dtype}; expected=({record_count}, 1000, 1), float32",
        )
    )
    results.append(
        print_check(
            "Y shape and dtype",
            y_data.shape == (record_count, 1000, 11) and y_data.dtype == np.float32,
            f"shape={y_data.shape}, dtype={y_data.dtype}; expected=({record_count}, 1000, 11), float32",
        )
    )

    metadata_lengths = {
        "ecg_id": len(ecg_id),
        "patient_id": len(patient_id),
        "strat_fold": len(strat_fold),
        "split": len(splits),
    }
    results.append(
        print_check(
            "Metadata lengths",
            all(length == record_count for length in metadata_lengths.values()),
            str(metadata_lengths),
        )
    )

    split_counts = {split_name: int(np.sum(splits == split_name)) for split_name in EXPECTED_SPLIT_COUNTS}
    unknown_splits = sorted(set(splits).difference(EXPECTED_SPLIT_COUNTS))
    results.append(
        print_check(
            "Train / validation / test counts",
            split_counts == EXPECTED_SPLIT_COUNTS and not unknown_splits,
            f"actual={split_counts}, expected={EXPECTED_SPLIT_COUNTS}, unknown={unknown_splits}",
        )
    )

    unique_patient_id, patient_counts = np.unique(patient_id, return_counts=True)
    duplicate_patients = unique_patient_id[patient_counts > 1]
    results.append(
        print_check(
            "Unique patient IDs",
            len(duplicate_patients) == 0,
            f"records={record_count}, unique_patients={len(unique_patient_id)}, duplicates={duplicate_patients.tolist()}",
        )
    )

    patient_splits: dict[int, set[str]] = {}
    for patient, split_name in zip(patient_id, splits, strict=True):
        patient_splits.setdefault(int(patient), set()).add(split_name)
    cross_split_patients = {
        patient: sorted(split_names)
        for patient, split_names in patient_splits.items()
        if len(split_names) > 1
    }
    results.append(
        print_check(
            "No patient overlap across splits",
            not cross_split_patients,
            f"cross_split_patients={cross_split_patients}",
        )
    )

    x_non_finite = int(np.size(x_data) - np.isfinite(x_data).sum())
    y_non_finite = int(np.size(y_data) - np.isfinite(y_data).sum())
    results.append(
        print_check("No NaN or infinity", x_non_finite == 0 and y_non_finite == 0,
                    f"X non-finite={x_non_finite}, Y non-finite={y_non_finite}")
    )

    print("Lead names:", arrays["lead_names"].astype(str).tolist())
    print("Sampling rate:", int(np.asarray(arrays["sampling_rate_hz"]).item()), "Hz")
    return all(results)


def save_first_record_plot(arrays: dict[str, np.ndarray], output_path: Path, overwrite: bool) -> None:
    """Reconstruct the first record in original lead order and save all 12 plots."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Plot already exists: {output_path}. Inspect it first; use --overwrite to replace it."
        )

    x_data = arrays["X"]
    y_data = arrays["Y"]
    lead_names = tuple(arrays["lead_names"].astype(str).tolist())
    sampling_rate = int(np.asarray(arrays["sampling_rate_hz"]).item())

    if x_data.ndim != 3 or y_data.ndim != 3 or len(x_data) == 0:
        raise ValueError("Cannot plot: X/Y must contain at least one 3-dimensional ECG record.")
    if "I" not in lead_names or len(lead_names) != 12:
        raise ValueError(f"Cannot plot: invalid lead_names={lead_names}")

    time_seconds = np.arange(x_data.shape[1]) / sampling_rate
    figure, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)
    ecg_id = arrays["ecg_id"][0]
    patient_id = arrays["patient_id"][0]
    figure.suptitle(f"Prepared PTB-XL first ECG: ecg_id={ecg_id}, patient_id={patient_id}", fontsize=15)

    target_index = 0
    for axis, lead_name in zip(axes, lead_names, strict=True):
        if lead_name == "I":
            waveform = x_data[0, :, 0]
            color = "tab:blue"
            source_label = "X (input)"
        else:
            waveform = y_data[0, :, target_index]
            target_index += 1
            color = "tab:orange"
            source_label = "Y (target)"

        axis.plot(time_seconds, waveform, color=color, linewidth=0.8)
        axis.set_ylabel(f"{lead_name}\n(mV)", rotation=0, labelpad=28, va="center")
        axis.set_title(source_label, loc="right", fontsize=9)
        axis.grid(True, alpha=0.3)

    if target_index != 11:
        raise ValueError(f"Expected 11 target leads, but plotted {target_index}.")

    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prepared PTB-XL 1-lead to 12-lead dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "processed" / "ptbxl_100hz_100records.npz",
        help="Path to the prepared NPZ dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "first_ecg_12lead_inspection.png",
        help="PNG path for the first reconstructed 12-lead ECG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing inspection PNG.",
    )
    arguments = parser.parse_args()

    arrays, _ = load_npz(arguments.dataset)
    passed = inspect_arrays(arrays)
    save_first_record_plot(arrays, arguments.output, arguments.overwrite)
    if not passed:
        raise SystemExit("Inspection completed, but one or more validation checks failed.")

    print("All dataset checks passed. No AI model was trained.")


if __name__ == "__main__":
    main()
