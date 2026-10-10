"""Read-only review of ECG 421 and 710; never download or modify source data."""

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

PROJECT = Path(__file__).resolve().parent
TARGETS = ((421, 36, 2490, "V4", (495, 496), (4.5, 5.5)),
           (710, 249, 5542, "V2", tuple(range(36, 45)), (0.0, 1.0)))
LIMIT = 10.0


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_record(signal, names, fs, ecg_id, out, channels, window, suffix):
    time = np.arange(len(signal)) / fs
    selected = (time >= window[0]) & (time <= window[1])
    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 2.4 * len(channels)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for ax, channel in zip(axes[:, 0], channels):
        values = signal[:, channel]
        bad = selected & (np.abs(values) > LIMIT)
        ax.plot(time[selected], values[selected], linewidth=0.8)
        ax.scatter(time[bad], values[bad], color="red", s=24, zorder=3,
                   label="|voltage| > 10 mV")
        for level in (-LIMIT, LIMIT):
            ax.axhline(level, color="darkorange", linestyle="--", linewidth=0.8)
        # Autoscaling includes both the full waveform and the reference lines.
        ax.margins(y=0.12)
        ax.set_ylabel(f"{names[channel]} (mV)")
        ax.grid(alpha=0.25)
        if bad.any():
            ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 0].set_xlim(*window)
    fig.suptitle(f"ECG {ecg_id} | {suffix} | {fs:g} Hz")
    fig.savefig(out / f"ecg_{ecg_id}_{suffix}.png", dpi=160)
    plt.close(fig)


def compare_original(raw_root, row, signal, names, fs, log):
    """Only local paths are passed to WFDB; absent files are never fetched."""
    relative = Path(row["filename_lr"])
    base = (raw_root / relative).resolve()
    if relative.is_absolute() or not base.is_relative_to(raw_root.resolve()):
        raise ValueError("Unsafe filename_lr outside raw root")
    log(f"Local WFDB base: {base}")
    if not base.with_suffix(".hea").is_file():
        raise FileNotFoundError(f"Local header missing: {base}.hea")
    import wfdb

    header = wfdb.rdheader(str(base))
    log(f"WFDB fs={header.fs}; sig_name={header.sig_name}; units={header.units}")
    for filename in header.file_name:
        local = (base.parent / filename).resolve()
        if not local.is_relative_to(raw_root.resolve()):
            raise ValueError("WFDB signal path outside raw root")
        if not local.is_file():
            raise FileNotFoundError(f"Local signal file missing: {local}")
    record = wfdb.rdrecord(str(base), physical=True)
    raw_names = [str(name).upper() for name in record.sig_name]
    if len(raw_names) != 12 or len(set(raw_names)) != 12 or set(raw_names) != set(names):
        raise ValueError(f"Incompatible WFDB lead names: {raw_names}")
    if record.fs != fs or record.p_signal.shape != signal.shape:
        raise ValueError(f"Incompatible fs/shape: {record.fs}, {record.p_signal.shape}")
    if any(str(unit).lower() != "mv" for unit in record.units):
        raise ValueError(f"Cannot compare as mV: units={record.units}")
    original = record.p_signal.astype(np.float32)
    if not np.isfinite(original).all():
        raise ValueError("Original contains nonfinite samples")
    results = []
    for channel, name in enumerate(names):
        source_channel = raw_names.index(name)
        difference = float(np.max(np.abs(original[:, source_channel].astype(np.float64)
                                         - signal[:, channel].astype(np.float64))))
        log(f"  {name}: max_abs_difference={difference:.12g} mV (original cast to float32)")
        results.append({"lead": name, "max_abs_difference_mv": difference})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=PROJECT / "data/processed/ptbxl_100hz_1000records.npz")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/processed/ptbxl_100hz_1000records_manifest.csv")
    parser.add_argument("--raw-root", type=Path, default=PROJECT / "data/raw/ptb-xl/1.0.3")
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    out = PROJECT / "results/flagged_ecg_review" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "review_log.txt").open("x", encoding="utf-8") as logfile:
        def log(message):
            print(message, flush=True)
            print(message, file=logfile, flush=True)

        log(f"Output: {out}")
        log(f"NPZ: {args.npz.resolve()}\nManifest: {args.manifest.resolve()}")
        with np.load(args.npz, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in
                      ("X", "Y", "ecg_id", "patient_id", "split", "lead_names", "sampling_rate_hz")}
        names = [str(name).upper() for name in arrays["lead_names"]]
        fs = float(arrays["sampling_rate_hz"].item())
        log(f"Actual lead_names={arrays['lead_names'].tolist()}; sampling_rate_hz={fs:g}")
        if len(names) != 12 or len(set(names)) != 12 or names[0] != "I":
            raise ValueError("Expected 12 unique lead names with Lead I first (X)")
        x, y = arrays["X"], arrays["Y"]
        n = len(arrays["ecg_id"])
        if x.ndim != 3 or x.shape[0] != n or x.shape[2] != 1 or y.shape != (*x.shape[:2], 11):
            raise ValueError(f"Invalid X/Y shapes: {x.shape}, {y.shape}")
        if not np.isfinite(fs) or fs <= 0 or not np.isclose(x.shape[1] / fs, 10):
            raise ValueError("Expected a finite positive sampling rate and a 10-second recording")
        for key in ("patient_id", "split", "ecg_id"):
            if arrays[key].shape != (n,):
                raise ValueError(f"Invalid metadata shape: {key}")
        with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        samples, comparisons, checks = [], [], []
        for ecg_id, expected_index, patient, problem, expected_samples, window in TARGETS:
            matches = np.flatnonzero(arrays["ecg_id"] == ecg_id)
            if len(matches) != 1:
                raise ValueError(f"ECG {ecg_id}: expected one match, found {len(matches)}")
            index = int(matches[0])
            actual_patient = int(arrays["patient_id"][index])
            split = str(arrays["split"][index])
            valid = index == expected_index and actual_patient == patient and split == "train"
            log(f"[{'PASS' if valid else 'FAIL'}] ECG {ecg_id}: index={index} (expected {expected_index}), "
                f"patient_id={actual_patient} (expected {patient}), split={split} (expected train)")
            checks.append({"ecg_id": ecg_id, "check": "NPZ identity", "passed": valid})
            signal = np.concatenate((x[index], y[index]), axis=1)
            if not np.isfinite(signal).all():
                raise ValueError(f"ECG {ecg_id}: nonfinite waveform")
            for sample, channel in np.argwhere(np.abs(signal) > LIMIT):
                entry = dict(ecg_id=ecg_id, index=index, patient_id=actual_patient, split=split,
                             lead=names[channel], sample_index=int(sample), time_s=float(sample / fs),
                             voltage_mv=float(signal[sample, channel]))
                samples.append(entry)
                log(f"  {entry['lead']} sample={sample}, time={sample / fs:.6f} s, "
                    f"voltage={signal[sample, channel]:.9g} mV")
            actual_samples = tuple(np.flatnonzero(np.abs(signal[:, names.index(problem)]) > LIMIT))
            valid = actual_samples == expected_samples
            log(f"[{'PASS' if valid else 'FAIL'}] {problem} exceedance indices={actual_samples}; expected={expected_samples}")
            checks.append({"ecg_id": ecg_id, "check": "Expected exceedance indices", "passed": valid})
            plot_record(signal, names, fs, ecg_id, out, range(12), (0, 10), "full_12lead")
            for name in ("I", problem):
                plot_record(signal, names, fs, ecg_id, out, [names.index(name)], window, f"zoom_{name}")
            try:
                rows = [row for row in manifest if int(row["ecg_id"]) == ecg_id]
                if len(rows) != 1:
                    raise ValueError(f"Manifest has {len(rows)} matches")
                row = rows[0]
                if int(row["patient_id"]) != actual_patient or row["split"] != split:
                    raise ValueError("Manifest patient_id/split disagrees with NPZ")
                log(f"Manifest identity matches; filename_lr={row['filename_lr']}")
                for result in compare_original(args.raw_root, row, signal, names, fs, log):
                    comparisons.append(dict(ecg_id=ecg_id, status="compared", reason="", **result))
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                log(f"ECG {ecg_id}: original comparison unavailable: {reason}; no download attempted")
                comparisons.append(dict(ecg_id=ecg_id, status="unavailable", reason=reason,
                                        lead="", max_abs_difference_mv=""))
        write_csv(out / "exceeding_samples.csv", samples,
                  ["ecg_id", "index", "patient_id", "split", "lead", "sample_index", "time_s", "voltage_mv"])
        write_csv(out / "original_comparison.csv", comparisons,
                  ["ecg_id", "status", "reason", "lead", "max_abs_difference_mv"])
        write_csv(out / "identity_checks.csv", checks, ["ecg_id", "check", "passed"])
        log(f"Finished: {len(samples)} exceeding samples; artifacts saved to {out}")


if __name__ == "__main__":
    main()
