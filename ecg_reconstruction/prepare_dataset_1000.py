"""Prepare a patient-unique 1,000-record PTB-XL dataset for 1-lead ECG estimation.

The script uses the official PTB-XL v1.0.3 metadata, records100 paths in
filename_lr, and the official folds: 1-8 train, 9 validation, and 10 test.
It does not train a model and does not alter the existing 100-record dataset.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import Counter
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import wfdb


DATASET_VERSION = "1.0.3"
PHYSIONET_BASE_URL = f"https://physionet.org/files/ptb-xl/{DATASET_VERSION}"
METADATA_FILENAME = "ptbxl_database.csv"
EXPECTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_SAMPLES = 1000  # 10 seconds * 100 Hz for records100.

# 100 from every official fold: 800 train + 100 validation + 100 test.
FOLD_TARGETS = {fold: 100 for fold in range(1, 11)}
SPLIT_BY_FOLD = {**{fold: "train" for fold in range(1, 9)}, 9: "validation", 10: "test"}
TOTAL_TARGET = sum(FOLD_TARGETS.values())
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_MAX_SECONDS = 120


def output_paths(processed_dir: Path) -> tuple[Path, Path, Path]:
    """Use names distinct from the existing 100-record dataset."""
    return (
        processed_dir / "ptbxl_100hz_1000records.npz",
        processed_dir / "ptbxl_100hz_1000records_manifest.csv",
        processed_dir / "ptbxl_100hz_1000records_excluded.csv",
    )


def ensure_outputs_are_safe(paths: tuple[Path, Path, Path], overwrite: bool) -> None:
    """Do not replace a previous 1,000-record output without explicit consent."""
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        joined_paths = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "A 1,000-record output already exists. Inspect it first, then use --overwrite only "
            f"if replacement is intended:\n{joined_paths}"
        )


def download_with_retries(
    source_url: str,
    destination: Path,
    max_attempts: int,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_download_seconds: int = DOWNLOAD_MAX_SECONDS,
) -> str:
    """Reuse an existing file, otherwise download it safely with limited retries.

    timeout_seconds limits blocking connection/read operations. A second timer
    checks elapsed time between reads, so even a slow stream can be retried.
    It is not a strict wall-clock deadline: a pending read may take up to the
    socket timeout, and operating-system DNS resolution may take longer.
    Each attempt owns a unique .part file; final cache files are never replaced.
    """
    if max_attempts < 1 or timeout_seconds <= 0 or max_download_seconds <= 0:
        raise ValueError("Download attempts and time limits must be positive")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"invalid_existing_cache_preserved: {destination}")
        return "reused"

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        temporary_path = None
        started = time.monotonic()
        print(
            f"DOWNLOAD attempt {attempt}/{max_attempts} "
            f"(socket timeout={timeout_seconds}s, elapsed limit={max_download_seconds}s): {source_url}",
            flush=True,
        )
        try:
            # Connect and read with an explicit timeout. A silent server can no
            # longer hold the first attempt indefinitely without an exception.
            with urlopen(source_url, timeout=timeout_seconds) as response:
                declared_length = response.headers.get("Content-Length")
                expected_bytes = int(declared_length) if declared_length is not None else None
                if expected_bytes is not None and expected_bytes <= 0:
                    raise OSError(f"Invalid Content-Length: {expected_bytes}")
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
                )
                temporary_path = Path(temporary_name)
                received_bytes = 0
                with os.fdopen(descriptor, "wb") as output:
                    while True:
                        if time.monotonic() - started >= max_download_seconds:
                            raise TimeoutError(f"Download elapsed limit exceeded: {max_download_seconds}s")
                        # read1 makes at most one underlying read, allowing the
                        # elapsed-time check to run even for a slow byte stream.
                        chunk = response.read1(64 * 1024)
                        if time.monotonic() - started >= max_download_seconds:
                            raise TimeoutError(f"Download elapsed limit exceeded: {max_download_seconds}s")
                        if not chunk:
                            break
                        output.write(chunk)
                        received_bytes += len(chunk)

                if received_bytes == 0:
                    raise OSError("Empty download")
                if expected_bytes is not None and received_bytes != expected_bytes:
                    raise OSError(
                        f"Incomplete download: received={received_bytes}, expected={expected_bytes} bytes"
                    )

            if destination.exists():
                # Another process completed the same file while this attempt ran.
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise OSError(f"Invalid cache appeared during download: {destination}")
                return "reused"
            temporary_path.rename(destination)
            print(f"DOWNLOAD COMPLETE: {destination.name} ({received_bytes} bytes)", flush=True)
            return "downloaded"
        except (HTTPError, URLError, OSError, HTTPException, ValueError) as error:
            print(f"DOWNLOAD ERROR attempt {attempt}/{max_attempts}: {type(error).__name__}: {error}", flush=True)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"download_failed_after_{max_attempts}_attempts: {source_url}; {error}"
                ) from error
        finally:
            # Cleanup only this attempt's temporary file, including on Ctrl+C.
            # Completed ECG files and earlier interrupted .part files are kept.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        time.sleep(attempt)  # 1 second, then 2 seconds before retrying.

    raise RuntimeError("unreachable download retry state")


def read_metadata(
    raw_root: Path, max_attempts: int,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_download_seconds: int = DOWNLOAD_MAX_SECONDS,
) -> pd.DataFrame:
    """Load the official metadata and validate the columns needed for collection."""
    metadata_path = raw_root / METADATA_FILENAME
    download_with_retries(
        f"{PHYSIONET_BASE_URL}/{METADATA_FILENAME}", metadata_path, max_attempts,
        timeout_seconds, max_download_seconds,
    )
    metadata = pd.read_csv(metadata_path)

    required_columns = {"ecg_id", "patient_id", "strat_fold", "filename_lr"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"metadata_missing_columns={sorted(missing_columns)}")

    metadata = metadata.copy()
    for column in ("ecg_id", "patient_id", "strat_fold"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata = metadata.dropna(subset=["ecg_id", "patient_id", "strat_fold", "filename_lr"])
    metadata["ecg_id"] = metadata["ecg_id"].astype(np.int64)
    metadata["patient_id"] = metadata["patient_id"].astype(np.int64)
    metadata["strat_fold"] = metadata["strat_fold"].astype(np.int64)
    metadata = metadata[metadata["strat_fold"].isin(FOLD_TARGETS)].copy()
    metadata["split"] = metadata["strat_fold"].map(SPLIT_BY_FOLD)
    return metadata


def remove_cross_split_patients(metadata: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    """Defensively remove a patient if metadata assigns it to multiple splits."""
    split_count = metadata.groupby("patient_id")["split"].nunique()
    cross_split_patients = set(split_count[split_count > 1].index)
    if cross_split_patients:
        print(f"WARNING: excluding {len(cross_split_patients)} cross-split patients.")
        metadata = metadata[~metadata["patient_id"].isin(cross_split_patients)].copy()
    else:
        print("Patient leakage check: no metadata patient appears in more than one split.")
    return metadata, cross_split_patients


def record_base_path(raw_root: Path, filename_lr: str) -> Path:
    """Resolve a safe local path for an official records100 filename_lr value."""
    relative_base = Path(filename_lr)
    if (
        relative_base.is_absolute()
        or not relative_base.parts
        or relative_base.parts[0] != "records100"
        or ".." in relative_base.parts
    ):
        raise ValueError(f"unexpected_filename_lr={filename_lr}")
    return raw_root / relative_base


def is_cached_complete(raw_root: Path, filename_lr: str) -> bool:
    """Return true only when both local WFDB files already exist."""
    try:
        local_base = record_base_path(raw_root, filename_lr)
    except ValueError:
        return False
    return local_base.with_suffix(".hea").exists() and local_base.with_suffix(".dat").exists()


def download_and_read_record(
    raw_root: Path, filename_lr: str, max_attempts: int,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_download_seconds: int = DOWNLOAD_MAX_SECONDS,
) -> tuple[Any, Counter[str]]:
    """Reuse/download header and waveform files, then read the WFDB record."""
    local_base = record_base_path(raw_root, filename_lr)
    remote_base = f"{PHYSIONET_BASE_URL}/{Path(filename_lr).as_posix()}"
    file_status = Counter()
    for extension in (".hea", ".dat"):
        status = download_with_retries(
            f"{remote_base}{extension}", local_base.with_suffix(extension), max_attempts,
            timeout_seconds, max_download_seconds,
        )
        file_status[status] += 1
    return wfdb.rdrecord(str(local_base)), file_status


def normalize_lead_names(lead_names: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize case only; do not alter lead names or their order."""
    return tuple(name.upper() for name in lead_names)


def split_record(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate one record and return Lead I X plus the other 11-lead Y."""
    signal = np.asarray(record.p_signal)
    raw_lead_names = tuple(record.sig_name)
    lead_names = normalize_lead_names(raw_lead_names)

    if record.fs != 100:
        raise ValueError(f"unexpected_sampling_rate={record.fs}")
    if signal.shape != (EXPECTED_SAMPLES, len(EXPECTED_LEADS)):
        raise ValueError(f"unexpected_signal_shape={signal.shape}")
    if lead_names != EXPECTED_LEADS:
        raise ValueError(f"unexpected_lead_order=raw:{raw_lead_names}, normalized:{lead_names}")
    non_finite_count = int(np.size(signal) - np.isfinite(signal).sum())
    if non_finite_count:
        raise ValueError(f"non_finite_values={non_finite_count}")

    lead_i_index = lead_names.index("I")
    target_indices = [index for index, name in enumerate(lead_names) if name != "I"]
    x_record = signal[:, lead_i_index : lead_i_index + 1].astype(np.float32, copy=False)
    y_record = signal[:, target_indices].astype(np.float32, copy=False)
    if x_record.shape != (EXPECTED_SAMPLES, 1):
        raise ValueError(f"unexpected_X_shape={x_record.shape}")
    if y_record.shape != (EXPECTED_SAMPLES, 11):
        raise ValueError(f"unexpected_Y_shape={y_record.shape}")
    if x_record.dtype != np.float32 or y_record.dtype != np.float32:
        raise TypeError(f"unexpected_dtype=X:{x_record.dtype},Y:{y_record.dtype}")
    return x_record, y_record


def write_outputs_atomically(
    paths: tuple[Path, Path, Path],
    x_data: np.ndarray,
    y_data: np.ndarray,
    manifest: pd.DataFrame,
    exclusions: pd.DataFrame,
    overwrite: bool,
) -> None:
    """Write complete temporary files first, then publish all three outputs.

    Existing outputs are not touched until every new NPZ/CSV write succeeds.
    """
    npz_path, manifest_path, exclusions_path = paths
    processed_dir = npz_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = (
        npz_path.with_name(f".{npz_path.stem}.tmp.npz"),
        manifest_path.with_name(f".{manifest_path.stem}.tmp.csv"),
        exclusions_path.with_name(f".{exclusions_path.stem}.tmp.csv"),
    )
    if any(path.exists() for path in temporary_paths):
        raise FileExistsError(
            "Temporary output exists from an interrupted earlier run. Inspect it manually before retrying: "
            + ", ".join(str(path) for path in temporary_paths if path.exists())
        )

    try:
        np.savez_compressed(
            temporary_paths[0],
            X=x_data,
            Y=y_data,
            ecg_id=manifest["ecg_id"].to_numpy(dtype=np.int64),
            patient_id=manifest["patient_id"].to_numpy(dtype=np.int64),
            strat_fold=manifest["strat_fold"].to_numpy(dtype=np.int64),
            split=manifest["split"].to_numpy(dtype="U10"),
            lead_names=np.asarray(EXPECTED_LEADS, dtype="U4"),
            sampling_rate_hz=np.asarray(100, dtype=np.int64),
        )
        manifest.to_csv(temporary_paths[1], index=False)
        exclusions.to_csv(temporary_paths[2], index=False)

        # Recheck immediately before publishing in case another process created an output.
        ensure_outputs_are_safe(paths, overwrite)
        for temporary_path, final_path in zip(temporary_paths, paths, strict=True):
            if final_path.exists() and overwrite:
                temporary_path.replace(final_path)
            else:
                temporary_path.rename(final_path)
    except Exception:
        # Temporary files from this run are intentionally retained for manual inspection.
        raise


def prepare_dataset(
    raw_root: Path, processed_dir: Path, max_attempts: int, overwrite: bool,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_download_seconds: int = DOWNLOAD_MAX_SECONDS,
) -> None:
    """Collect exactly 1,000 valid, patient-unique records before writing processed outputs."""
    paths = output_paths(processed_dir)
    ensure_outputs_are_safe(paths, overwrite)
    print(f"Download settings: attempts={max_attempts}, socket timeout={timeout_seconds}s, elapsed limit={max_download_seconds}s")
    metadata = read_metadata(raw_root, max_attempts, timeout_seconds, max_download_seconds)
    metadata, cross_split_patients = remove_cross_split_patients(metadata)

    x_records: list[np.ndarray] = []
    y_records: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    used_patient_ids: set[int] = set()
    collected_by_fold: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    file_status_counts: Counter[str] = Counter()

    for fold, target_count in FOLD_TARGETS.items():
        candidates = metadata.loc[metadata["strat_fold"] == fold].copy()
        # Prefer complete local cache entries. It saves bandwidth while remaining deterministic.
        candidates["cached_complete"] = candidates["filename_lr"].map(
            lambda filename: is_cached_complete(raw_root, filename)
        )
        candidates = candidates.sort_values(["cached_complete", "ecg_id"], ascending=[False, True])
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
                record, file_status = download_and_read_record(
                    raw_root, row.filename_lr, max_attempts, timeout_seconds, max_download_seconds
                )
                x_record, y_record = split_record(record)
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
            file_status_counts.update(file_status)
            collected_total = len(selected_rows)
            print(
                f"PROGRESS {collected_total}/{TOTAL_TARGET} ({collected_total / TOTAL_TARGET:.1%}) | "
                f"fold {fold}: {collected_by_fold[fold]}/{target_count} | "
                f"ACCEPT ecg_id={ecg_id}, patient_id={patient_id}, cache={dict(file_status)}"
            )

        if collected_by_fold[fold] < target_count:
            print(
                f"INSUFFICIENT fold={fold}: {collected_by_fold[fold]}/{target_count}. "
                "No processed dataset will be written unless all 1,000 valid records are collected."
            )

    if len(selected_rows) != TOTAL_TARGET:
        print("\n===== Collection failed safely =====")
        print(f"Collected valid records: {len(selected_rows)}/{TOTAL_TARGET}")
        print(f"Per-fold counts: {dict(sorted(collected_by_fold.items()))}")
        print(f"Rejection reasons: {dict(rejection_counts)}")
        raise RuntimeError("Could not collect exactly 1,000 valid records; no processed output was written.")

    x_data = np.stack(x_records).astype(np.float32, copy=False)
    y_data = np.stack(y_records).astype(np.float32, copy=False)
    manifest = pd.DataFrame(selected_rows)
    exclusions = pd.DataFrame(excluded_rows, columns=["ecg_id", "patient_id", "strat_fold", "reason"])

    # Final checks apply to the entire new dataset before anything is published.
    assert x_data.shape == (TOTAL_TARGET, EXPECTED_SAMPLES, 1), f"Invalid X shape: {x_data.shape}"
    assert y_data.shape == (TOTAL_TARGET, EXPECTED_SAMPLES, 11), f"Invalid Y shape: {y_data.shape}"
    assert x_data.dtype == np.float32 and y_data.dtype == np.float32, "X/Y must be float32"
    assert np.isfinite(x_data).all() and np.isfinite(y_data).all(), "X/Y contains NaN or infinity"
    assert manifest["patient_id"].is_unique, "Patient duplication detected"
    assert (manifest["split"] == "train").sum() == 800, "Train count must be 800"
    assert (manifest["split"] == "validation").sum() == 100, "Validation count must be 100"
    assert (manifest["split"] == "test").sum() == 100, "Test count must be 100"

    write_outputs_atomically(paths, x_data, y_data, manifest, exclusions, overwrite)
    print("\n===== 1,000-record dataset complete =====")
    print("Records — train: 800, validation: 100, test: 100")
    print(f"X shape={x_data.shape}, dtype={x_data.dtype}")
    print(f"Y shape={y_data.shape}, dtype={y_data.dtype}")
    print(f"Unique patients: {manifest['patient_id'].nunique()}")
    print(f"Cross-split metadata patients excluded: {len(cross_split_patients)}")
    print(f"Raw files — reused: {file_status_counts['reused']}, downloaded: {file_status_counts['downloaded']}")
    print(f"Excluded records: {len(exclusions)}; reasons: {dict(rejection_counts)}")
    print(f"Dataset: {paths[0].resolve()}")
    print(f"Manifest: {paths[1].resolve()}")
    print(f"Exclusions: {paths[2].resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 1,000 patient-unique PTB-XL records for ECG lead estimation.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw" / "ptb-xl" / DATASET_VERSION,
        help="Shared PTB-XL v1.0.3 cache; existing records100 files are reused.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory for new 1,000-record NPZ and audit CSV files.",
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=3,
        help="Maximum total attempts per missing metadata/WFDB file, including the first (default: 3).",
    )
    parser.add_argument(
        "--download-timeout", type=int, default=DOWNLOAD_TIMEOUT_SECONDS,
        help="Timeout in seconds for blocking connection/read operations (default: 30).",
    )
    parser.add_argument(
        "--download-max-seconds", type=int, default=DOWNLOAD_MAX_SECONDS,
        help="Elapsed time limit checked between reads per attempt (default: 120).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement only of this script's 1,000-record output filenames.",
    )
    arguments = parser.parse_args()
    if arguments.download_attempts < 1:
        parser.error("--download-attempts must be at least 1")
    if arguments.download_timeout < 1 or arguments.download_max_seconds < 1:
        parser.error("Download time limits must be at least 1 second")
    prepare_dataset(
        arguments.raw_dir, arguments.processed_dir, arguments.download_attempts, arguments.overwrite,
        arguments.download_timeout, arguments.download_max_seconds,
    )


if __name__ == "__main__":
    main()
