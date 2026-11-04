"""Read-only validation and three seeded ECG plots for the 1,000-record NPZ.

Each execution creates a NEW folder under results/inspection_1000. Source
datasets are read only. Signal flags are engineering checks for manual review,
not diagnostic thresholds; no records are removed or used to train a model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
REQUIRED_KEYS = (
    "X", "Y", "ecg_id", "patient_id", "strat_fold", "split", "lead_names", "sampling_rate_hz",
)
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
SPLIT_COUNTS = {"train": 800, "validation": 100, "test": 100}


class InspectionReport:
    """Collect PASS/FAIL results and identify every record with a detected issue."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.checks: list[dict] = []
        self.issues: list[dict] = []

    def check(self, name: str, passed: bool, detail: str) -> bool:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}", flush=True)
        self.checks.append({"check": name, "status": status, "detail": detail})
        return passed

    def issue(self, check: str, index: int, detail: str, lead: str = "") -> None:
        row = {"check": check, "index": index, "lead": lead, "detail": detail}
        for key in ("ecg_id", "patient_id", "split"):
            array = self.arrays.get(key)
            row[key] = str(array[index]) if array is not None and array.ndim == 1 and index < len(array) else "unavailable"
        self.issues.append(row)
        print(
            f"  index={index}, ecg_id={row['ecg_id']}, patient_id={row['patient_id']}, "
            f"split={row['split']}, lead={lead or '-'}: {detail}", flush=True,
        )

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(row["status"] == "PASS" for row in self.checks)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Print actual keys BEFORE loading array values; disallow pickle objects."""
    with np.load(path, allow_pickle=False) as archive:
        print("Actual NPZ keys:", ", ".join(archive.files), flush=True)
        missing = sorted(set(REQUIRED_KEYS).difference(archive.files))
        if missing:
            raise ValueError(f"Missing NPZ keys: {missing}")
        return {key: archive[key] for key in REQUIRED_KEYS}


def inspect_dataset(arrays: dict[str, np.ndarray], max_abs_mv: float, flat_tolerance_mv: float) -> tuple[InspectionReport, bool]:
    """Validate the dataset and return whether shapes/metadata allow safe plots."""
    report = InspectionReport(arrays)
    report.check("Required NPZ keys", True, ", ".join(REQUIRED_KEYS))
    shape_ok = True
    dtype_ok = True
    for key, expected_shape in (("X", (1000, 1000, 1)), ("Y", (1000, 1000, 11))):
        shape_ok = report.check(f"{key} shape", arrays[key].shape == expected_shape,
                                f"actual={arrays[key].shape}, expected={expected_shape}") and shape_ok
        dtype_ok = report.check(f"{key} dtype", arrays[key].dtype == np.float32,
                                f"actual={arrays[key].dtype}, expected=float32") and dtype_ok

    metadata_ok = True
    for key in ("ecg_id", "patient_id", "strat_fold", "split"):
        metadata_ok = report.check(f"{key} metadata shape", arrays[key].shape == (1000,),
                                   f"actual={arrays[key].shape}, expected=(1000,)") and metadata_ok
    for key in ("ecg_id", "patient_id", "strat_fold"):
        integer_ok = np.issubdtype(arrays[key].dtype, np.integer)
        metadata_ok = report.check(f"{key} integer dtype", integer_ok,
                                   f"actual={arrays[key].dtype}") and metadata_ok

    names = arrays["lead_names"]
    lead_ok = names.shape == (12,) and tuple(str(name).upper() for name in names) == LEADS
    report.check("Lead order (case insensitive)", lead_ok,
                 f"actual={names.tolist()}; X=I, Y={LEADS[1:]}")
    print("Lead mapping is checked against saved metadata; NPZ alone cannot prove that a column was never mislabeled.")

    rate = arrays["sampling_rate_hz"]
    rate_ok = rate.shape == () and np.issubdtype(rate.dtype, np.number) and rate.item() == 100
    report.check("Sampling rate", rate_ok, f"actual={rate.tolist()}, expected=100 Hz")

    if metadata_ok:
        splits = arrays["split"].astype(str)
        counts = {name: int(np.count_nonzero(splits == name)) for name in SPLIT_COUNTS}
        unknown = np.flatnonzero(~np.isin(splits, list(SPLIT_COUNTS)))
        report.check("Split counts", counts == SPLIT_COUNTS and len(unknown) == 0,
                     f"actual={counts}, expected={SPLIT_COUNTS}")
        for index in unknown:
            report.issue("Split counts", int(index), f"Unknown split label: {splits[index]}")

        for key in ("ecg_id", "patient_id"):
            values, frequencies = np.unique(arrays[key], return_counts=True)
            duplicates = values[frequencies > 1]
            report.check(f"Unique {key}", len(duplicates) == 0,
                         f"unique={len(values)}/1000, duplicated IDs={duplicates.tolist()}")
            for index in np.flatnonzero(np.isin(arrays[key], duplicates)):
                report.issue(f"Unique {key}", int(index), f"Repeated {key}={arrays[key][index]}")

        patient_sets = {name: set(arrays["patient_id"][splits == name].tolist()) for name in SPLIT_COUNTS}
        overlaps = {}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = patient_sets[first] & patient_sets[second]
            overlaps[f"{first}/{second}"] = sorted(overlap)
        report.check("No cross-split patient overlap", not any(overlaps.values()), str(overlaps))
        overlapping_ids = set().union(*(set(ids) for ids in overlaps.values()))
        for index in np.flatnonzero(np.isin(arrays["patient_id"], list(overlapping_ids))):
            report.issue("No cross-split patient overlap", int(index), "Patient appears in multiple splits")

        folds = arrays["strat_fold"]
        fold_valid = np.isin(folds, np.arange(1, 11))
        expected_split = np.where(folds <= 8, "train", np.where(folds == 9, "validation", "test"))
        mismatch = np.flatnonzero(~fold_valid | (splits != expected_split))
        report.check("Official fold/split mapping", len(mismatch) == 0,
                     "folds 1-8=train, 9=validation, 10=test")
        for index in mismatch:
            report.issue("Official fold/split mapping", int(index), f"fold={folds[index]}, split={splits[index]}")
    else:
        for check in ("Split counts", "Unique ecg_id", "Unique patient_id", "No cross-split patient overlap", "Official fold/split mapping"):
            report.check(check, False, "Cannot inspect safely: invalid metadata shape or dtype")

    print(f"Signal review settings: |voltage| > {max_abs_mv:g} mV; flat range <= {flat_tolerance_mv:g} mV")
    print("These are configurable engineering flags, not clinical limits. No records are deleted.")
    for key, expected_channels in (("X", 1), ("Y", 11)):
        signal = arrays[key]
        if signal.ndim != 3 or signal.shape[2] != expected_channels or not np.issubdtype(signal.dtype, np.floating):
            for label in ("NaN/inf", "Voltage range", "Flat channels"):
                report.check(f"{key} {label}", False, "Cannot inspect safely: invalid waveform shape/dtype")
            continue

        channel_names = LEADS[:1] if key == "X" else LEADS[1:]
        finite = np.isfinite(signal)
        nan_count = int(np.isnan(signal).sum())
        inf_count = int(np.isinf(signal).sum())
        report.check(f"{key} NaN/inf", nan_count == 0 and inf_count == 0,
                     f"NaN={nan_count}, inf={inf_count}")
        for index, channel in np.argwhere(np.any(~finite, axis=1)):
            bad_times = np.flatnonzero(~finite[index, :, channel])
            report.issue(f"{key} NaN/inf", int(index),
                         f"bad samples={len(bad_times)}, first time indices={bad_times[:10].tolist()}", channel_names[channel])

        extreme = finite & (np.abs(signal) > max_abs_mv)
        finite_values = signal[finite]
        voltage_range = f"min={finite_values.min():.6f}, max={finite_values.max():.6f} mV" if finite_values.size else "no finite samples"
        report.check(f"{key} Voltage range", not extreme.any(), voltage_range)
        for index, channel in np.argwhere(np.any(extreme, axis=1)):
            positions = np.flatnonzero(extreme[index, :, channel])
            report.issue(f"{key} Voltage range", int(index),
                         f"|voltage| exceeds {max_abs_mv:g} mV; samples={len(positions)}, first indices={positions[:10].tolist()}", channel_names[channel])

        # Only channels with entirely finite values are checked for flatness.
        safe_signal = np.where(finite, signal, 0.0)
        spans = np.ptp(safe_signal, axis=1)
        flat = np.all(finite, axis=1) & (spans <= flat_tolerance_mv)
        report.check(f"{key} Flat channels", not flat.any(), f"flat channels={int(flat.sum())}")
        for index, channel in np.argwhere(flat):
            report.issue(f"{key} Flat channels", int(index),
                         f"peak-to-peak range={spans[index, channel]:.9g} mV", channel_names[channel])

    return report, shape_ok and dtype_ok and metadata_ok and lead_ok and rate_ok


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Exclusive creation also prevents overwrites inside the new run folder."""
    with path.open("x", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_sample_plots(arrays: dict[str, np.ndarray], output_dir: Path, seed: int, report: InspectionReport) -> None:
    """Choose exactly three records without replacement and draw X plus 11 Y leads."""
    indices = np.random.default_rng(seed).choice(1000, size=3, replace=False)
    time_seconds = np.arange(1000) / 100.0
    selected = []
    for index in indices:
        index = int(index)
        ecg_id = int(arrays["ecg_id"][index])
        patient_id = int(arrays["patient_id"][index])
        split = str(arrays["split"][index])
        path = output_dir / f"ecg_index{index:04d}_id{ecg_id}_{split}.png"
        signal = np.concatenate((arrays["X"][index], arrays["Y"][index]), axis=1)
        figure, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)
        try:
            figure.suptitle(f"ECG {ecg_id} | patient {patient_id} | {split} | index {index} | seed {seed}")
            for channel, (axis, lead) in enumerate(zip(axes, LEADS, strict=True)):
                axis.plot(time_seconds, signal[:, channel], linewidth=0.8,
                          color="tab:blue" if channel == 0 else "tab:orange")
                axis.set_ylabel(f"{lead}\n(mV)", rotation=0, labelpad=28, va="center")
                axis.set_title("X: observed input" if channel == 0 else f"Y: target column {channel - 1}", loc="right", fontsize=9)
                axis.grid(True, alpha=0.3)
            axes[-1].set_xlabel("Time (seconds)")
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            with path.open("xb") as output:
                figure.savefig(output, format="png", dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        selected.append({"index": index, "ecg_id": ecg_id, "patient_id": patient_id,
                         "split": split, "seed": seed, "png": path.name})
        report.check(f"Plot ECG {ecg_id}", True, str(path))
    save_csv(output_dir / "selected_ecgs.csv", selected,
             ["index", "ecg_id", "patient_id", "split", "seed", "png"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the 1,000-record PTB-XL dataset without model training.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data/processed/ptbxl_100hz_1000records.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results/inspection_1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-abs-mv", type=float, default=10.0, help="Manual-review threshold for absolute voltage (default: 10 mV).")
    parser.add_argument("--flat-tolerance-mv", type=float, default=1e-8, help="Flat-channel peak-to-peak tolerance (default: 1e-8 mV).")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.max_abs_mv) or args.max_abs_mv <= 0:
        parser.error("--max-abs-mv must be finite and positive")
    if not np.isfinite(args.flat_tolerance_mv) or args.flat_tolerance_mv < 0:
        parser.error("--flat-tolerance-mv must be finite and nonnegative")

    try:
        arrays = load_dataset(args.dataset)
        report, can_plot = inspect_dataset(arrays, args.max_abs_mv, args.flat_tolerance_mv)
        # Exclusive creation of a unique run folder keeps every earlier result.
        korea_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"run_{korea_time}_seed{args.seed}_{uuid.uuid4().hex[:8]}"
        output_dir.mkdir(parents=True, exist_ok=False)
        if can_plot:
            save_sample_plots(arrays, output_dir, args.seed, report)
        else:
            report.check("Three ECG plots", False, "Skipped: invalid shapes, metadata, lead order, or sampling rate")
        save_csv(output_dir / "inspection_checks.csv", report.checks, ["check", "status", "detail"])
        save_csv(output_dir / "inspection_issues.csv", report.issues,
                 ["check", "index", "ecg_id", "patient_id", "split", "lead", "detail"])
        report_status = "PASS" if report.passed else "FAIL"
        print(f"[{report_status}] Overall inspection; results: {output_dir}")
        print("Test rows were used only for quality inspection/plots; no training or model selection occurred.")
        return 0 if report.passed else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"[FAIL] Inspection stopped: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
