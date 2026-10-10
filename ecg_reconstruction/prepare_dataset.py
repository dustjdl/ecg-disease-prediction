"""Prepare 100 patient-unique PTB-XL records for 1-lead to 12-lead learning.

This script downloads only the metadata CSV and the required 100 Hz WFDB files.
It does not train a model. Existing processed outputs are protected unless
--overwrite is supplied explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import wfdb


DATASET_VERSION = "1.0.3"
PHYSIONET_BASE_URL = f"https://physionet.org/files/ptb-xl/{DATASET_VERSION}"
METADATA_FILENAME = "ptbxl_database.csv"
# PTB-XL headers may use aVR/aVL/aVF or AVR/AVL/AVF.  The canonical form is
# uppercase so lead-name comparison is case-insensitive while position remains strict.
EXPECTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_SAMPLES = 1000  # 10 seconds * 100 Hz for PTB-XL records100.

# Keeping 10 records from each fold preserves the official recommended split:
# folds 1-8 -> train (80 records), fold 9 -> validation (10), fold 10 -> test (10).
FOLD_TARGETS = {fold: 10 for fold in range(1, 11)}
SPLIT_BY_FOLD = {**{fold: "train" for fold in range(1, 9)}, 9: "validation", 10: "test"}


def download_if_missing(source_url: str, destination: Path) -> None:
    """Download one file only if it is not already present.

    A temporary .part file prevents a partially downloaded file from appearing
    to be a valid ECG record. Existing files are never replaced here.
    """
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    if temporary_path.exists():
        raise FileExistsError(
            f"Partial download found: {temporary_path}. Check or remove it manually before retrying."
        )

    print(f"DOWNLOAD: {source_url}")
    try:
        urlretrieve(source_url, temporary_path)
        temporary_path.replace(destination)
    except (URLError, OSError) as error:
        if temporary_path.exists():
            print(f"Partial file kept for inspection: {temporary_path}")
        raise RuntimeError(f"Could not download {source_url}") from error


def read_metadata(raw_root: Path) -> pd.DataFrame:
    """Download and validate the official PTB-XL metadata table."""
    metadata_path = raw_root / METADATA_FILENAME
    download_if_missing(f"{PHYSIONET_BASE_URL}/{METADATA_FILENAME}", metadata_path)
    metadata = pd.read_csv(metadata_path)

    required_columns = {"ecg_id", "patient_id", "strat_fold", "filename_lr"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    metadata = metadata.copy()
    metadata["strat_fold"] = pd.to_numeric(metadata["strat_fold"], errors="coerce")
    metadata["ecg_id"] = pd.to_numeric(metadata["ecg_id"], errors="coerce")
    metadata["patient_id"] = pd.to_numeric(metadata["patient_id"], errors="coerce")
    metadata = metadata.dropna(subset=["ecg_id", "patient_id", "strat_fold", "filename_lr"])
    metadata["ecg_id"] = metadata["ecg_id"].astype(np.int64)
    metadata["patient_id"] = metadata["patient_id"].astype(np.int64)
    metadata["strat_fold"] = metadata["strat_fold"].astype(np.int64)
    metadata = metadata[metadata["strat_fold"].isin(FOLD_TARGETS)].copy()
    metadata["split"] = metadata["strat_fold"].map(SPLIT_BY_FOLD)
    return metadata


def remove_cross_split_patients(metadata: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    """Exclude any patient assigned to more than one selected split.

    PTB-XL v1.0.3 already constructs strat_fold at patient level, so this is a
    defensive check that makes the no-patient-overlap rule explicit.
    """
    split_count_per_patient = metadata.groupby("patient_id")["split"].nunique()
    cross_split_patients = set(split_count_per_patient[split_count_per_patient > 1].index)
    if cross_split_patients:
        print(
            "WARNING: excluding "
            f"{len(cross_split_patients)} patients found in more than one split."
        )
        metadata = metadata[~metadata["patient_id"].isin(cross_split_patients)].copy()
    else:
        print("Patient split check: no patient appears in more than one split.")
    return metadata, cross_split_patients


def download_and_read_record(raw_root: Path, filename_lr: str):
    """Download the WFDB header and signal specified by filename_lr, then read it."""
    relative_base = Path(filename_lr)
    if relative_base.is_absolute() or relative_base.parts[0] != "records100":
        raise ValueError(f"Unexpected filename_lr path: {filename_lr}")

    local_base = raw_root / relative_base
    remote_base = f"{PHYSIONET_BASE_URL}/{relative_base.as_posix()}"
    download_if_missing(f"{remote_base}.hea", local_base.with_suffix(".hea"))
    download_if_missing(f"{remote_base}.dat", local_base.with_suffix(".dat"))
    return wfdb.rdrecord(str(local_base))


def normalize_lead_names(lead_names: tuple[str, ...]) -> tuple[str, ...]:
    """Convert only letter case, without changing lead positions or names."""
    return tuple(name.upper() for name in lead_names)


def split_record(record: Any) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate one 100 Hz record and split Lead I from the other 11 leads."""
    signal = np.asarray(record.p_signal)
    raw_lead_names = tuple(record.sig_name)
    lead_names = normalize_lead_names(raw_lead_names)

    if record.fs != 100:
        raise ValueError(f"unexpected_sampling_rate={record.fs}")
    if signal.shape != (EXPECTED_SAMPLES, len(EXPECTED_LEADS)):
        raise ValueError(f"unexpected_signal_shape={signal.shape}")
    if lead_names != EXPECTED_LEADS:
        raise ValueError(
            f"unexpected_lead_order=raw:{raw_lead_names}, normalized:{lead_names}"
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"non_finite_values={np.size(signal) - np.isfinite(signal).sum()}")

    lead_i_index = lead_names.index("I")
    target_indices = [index for index, name in enumerate(lead_names) if name != "I"]
    x_record = signal[:, lead_i_index : lead_i_index + 1].astype(np.float32, copy=False)
    y_record = signal[:, target_indices].astype(np.float32, copy=False)

    if x_record.shape != (EXPECTED_SAMPLES, 1):
        raise ValueError(f"unexpected_X_shape={x_record.shape}")
    if y_record.shape != (EXPECTED_SAMPLES, 11):
        raise ValueError(f"unexpected_Y_shape={y_record.shape}")
    if x_record.dtype != np.float32 or y_record.dtype != np.float32:
        raise TypeError(f"unexpected_dtype=X:{x_record.dtype}, Y:{y_record.dtype}")
    return x_record, y_record, lead_names


def output_paths(processed_dir: Path) -> tuple[Path, Path, Path]:
    """Return the NPZ dataset path and two small CSV audit files."""
    return (
        processed_dir / "ptbxl_100hz_100records.npz",
        processed_dir / "ptbxl_100hz_100records_manifest.csv",
        processed_dir / "ptbxl_100hz_100records_excluded.csv",
    )


def ensure_outputs_are_safe(paths: tuple[Path, Path, Path], overwrite: bool) -> None:
    """Refuse to replace a previous prepared dataset without explicit consent."""
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        joined_paths = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Processed output already exists. Inspect it first; then rerun with --overwrite only "
            f"if replacement is intended:\n{joined_paths}"
        )


def prepare_dataset(raw_root: Path, processed_dir: Path, overwrite: bool) -> None:
    """Collect valid records fold by fold and write NumPy arrays plus audit metadata."""
    paths = output_paths(processed_dir)
    ensure_outputs_are_safe(paths, overwrite)

    metadata = read_metadata(raw_root)
    metadata, cross_split_patients = remove_cross_split_patients(metadata)

    x_records: list[np.ndarray] = []
    y_records: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    used_patient_ids: set[int] = set()
    rejection_counts: Counter[str] = Counter()
    collected_by_fold: Counter[int] = Counter()

    for fold, target_count in FOLD_TARGETS.items():
        candidates = metadata.loc[metadata["strat_fold"] == fold].sort_values("ecg_id")
        print(f"\nFold {fold} ({SPLIT_BY_FOLD[fold]}): target={target_count}, candidates={len(candidates)}")

        for row in candidates.itertuples(index=False):
            if collected_by_fold[fold] >= target_count:
                break

            ecg_id = int(row.ecg_id)
            patient_id = int(row.patient_id)
            if patient_id in used_patient_ids:
                reason = "patient_already_selected"
                rejection_counts[reason] += 1
                excluded_rows.append({"ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold, "reason": reason})
                continue

            try:
                record = download_and_read_record(raw_root, row.filename_lr)
                x_record, y_record, lead_names = split_record(record)
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                reason = str(error)
                rejection_counts[reason] += 1
                excluded_rows.append({"ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold, "reason": reason})
                print(f"SKIP ecg_id={ecg_id}: {reason}")
                continue

            used_patient_ids.add(patient_id)
            x_records.append(x_record)
            y_records.append(y_record)
            selected_rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": patient_id,
                    "strat_fold": fold,
                    "split": SPLIT_BY_FOLD[fold],
                    "filename_lr": row.filename_lr,
                }
            )
            collected_by_fold[fold] += 1
            print(f"ACCEPT ecg_id={ecg_id}, patient_id={patient_id}, fold={fold}")

        actual_count = collected_by_fold[fold]
        if actual_count < target_count:
            print(
                f"INSUFFICIENT fold={fold}: collected={actual_count}/{target_count}. "
                "All available eligible candidates were exhausted; see the exclusions CSV for details."
            )
        else:
            print(f"COMPLETE fold={fold}: collected={actual_count}/{target_count}")

    if not x_records:
        raise RuntimeError("No valid ECG records were collected; no processed dataset was written.")

    x_data = np.stack(x_records).astype(np.float32, copy=False)
    y_data = np.stack(y_records).astype(np.float32, copy=False)
    manifest = pd.DataFrame(selected_rows)
    exclusions = pd.DataFrame(
        excluded_rows,
        columns=["ecg_id", "patient_id", "strat_fold", "reason"],
    )

    # Final dataset-level checks: each row is a record, each record has 1000 time samples.
    record_count = len(manifest)
    assert x_data.shape == (record_count, EXPECTED_SAMPLES, 1), f"Invalid X shape: {x_data.shape}"
    assert y_data.shape == (record_count, EXPECTED_SAMPLES, 11), f"Invalid Y shape: {y_data.shape}"
    assert x_data.dtype == np.float32, f"Invalid X dtype: {x_data.dtype}"
    assert y_data.dtype == np.float32, f"Invalid Y dtype: {y_data.dtype}"
    assert np.isfinite(x_data).all(), "X contains NaN or infinity"
    assert np.isfinite(y_data).all(), "Y contains NaN or infinity"
    assert manifest["patient_id"].is_unique, "Patient leakage: a selected patient appears more than once"

    processed_dir.mkdir(parents=True, exist_ok=True)
    npz_path, manifest_path, exclusions_path = paths
    np.savez_compressed(
        npz_path,
        X=x_data,
        Y=y_data,
        ecg_id=manifest["ecg_id"].to_numpy(dtype=np.int64),
        patient_id=manifest["patient_id"].to_numpy(dtype=np.int64),
        strat_fold=manifest["strat_fold"].to_numpy(dtype=np.int64),
        split=manifest["split"].to_numpy(dtype="U10"),
        lead_names=np.asarray(EXPECTED_LEADS, dtype="U4"),
        sampling_rate_hz=np.asarray(100, dtype=np.int64),
    )
    manifest.to_csv(manifest_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    print("\n===== Preparation summary =====")
    print(f"Collected records: {record_count}/{sum(FOLD_TARGETS.values())}")
    print(f"Train / validation / test: "
          f"{(manifest['split'] == 'train').sum()} / "
          f"{(manifest['split'] == 'validation').sum()} / "
          f"{(manifest['split'] == 'test').sum()}")
    print(f"X shape={x_data.shape}, dtype={x_data.dtype}")
    print(f"Y shape={y_data.shape}, dtype={y_data.dtype}")
    print(f"Unique patients: {manifest['patient_id'].nunique()}")
    print(f"Cross-split patients excluded: {len(cross_split_patients)}")
    print(f"Rejected records: {len(exclusions)}")
    if rejection_counts:
        print(f"Rejection reasons: {dict(rejection_counts)}")
    print(f"Saved dataset: {npz_path.resolve()}")
    print(f"Saved manifest: {manifest_path.resolve()}")
    print(f"Saved exclusions: {exclusions_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 100 PTB-XL records for 1-lead to 12-lead estimation.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw" / "ptb-xl" / DATASET_VERSION,
        help="Directory for the official metadata and downloaded records100 files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory for the prepared NPZ dataset and audit CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files in data/processed.",
    )
    arguments = parser.parse_args()
    prepare_dataset(arguments.raw_dir, arguments.processed_dir, arguments.overwrite)


if __name__ == "__main__":
    main()
