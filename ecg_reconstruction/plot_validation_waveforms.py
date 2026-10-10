"""Seed-42 validation-selected waveform cases, not independent test examples.

Reuse compare_validation_1000.prepare unchanged. No training or test loader.
Each run saves three selection categories with full 10 s and 0-2 s plots.
--validate-only checks metadata without waveform inference or output writes.
CSV overall_change describes the record's overall MSE; metric_change describes
the current row's MSE, not its MAE or RMSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_validation_records import METRICS, SCOPES, per_record_metrics, write_csv
from compare_validation_1000 import MSE_ATOL, MSE_RTOL, loss_check, prepare
from train import make_loader
from train_1000 import DATASET, LEADS, PROJECT, SETTINGS, environment, new_id, require, sha256_file, write_json

OUTPUT_ROOT = PROJECT / "results/validation_waveform_comparison"
SELECTION_NOTE = "Cases selected after examining all 100 validation outcomes; not independent test evidence."


def mse_change(difference):
    """Classify candidate-minus-baseline MSE with no tie tolerance."""
    require(np.isfinite(difference), "MSE difference must be finite")
    return "improved" if difference < 0 else "worsened" if difference > 0 else "equal"


def select_cases(ecg_ids, baseline_mse, candidate_mse):
    ids = np.asarray(ecg_ids)
    left, right = np.asarray(baseline_mse, dtype=np.float64), np.asarray(candidate_mse, dtype=np.float64)
    require(ids.ndim == 1 and len(ids) > 0 and np.issubdtype(ids.dtype, np.integer)
            and len(np.unique(ids)) == len(ids), "Expected unique integer ECG IDs")
    require(left.shape == right.shape == ids.shape and np.isfinite(left).all()
            and np.isfinite(right).all() and (left >= 0).all() and (right >= 0).all(), "Invalid record MSE arrays")
    delta = right - left
    median = float(np.median(delta))
    definitions = (
        ("most_improved", delta, "minimum candidate-minus-baseline overall MSE"),
        ("most_worsened", -delta, "maximum candidate-minus-baseline overall MSE"),
        ("median_nearest", np.abs(delta - median), "minimum absolute distance to median candidate-minus-baseline overall MSE"),
    )
    selected = []
    for category, score, criterion in definitions:
        index = int(np.lexsort((ids, score))[0])
        difference = float(delta[index])
        selected.append({"category": category, "validation_position": index, "ecg_id": int(ids[index]),
                         "selection_criterion": criterion, "tie_break": "ecg_id ascending",
                         "median_delta_mse_mV2": median, "candidate_minus_baseline_mse_mV2": difference,
                         "overall_change": mse_change(difference),
                         "selected_using_validation_results": True})
    # Categories are independent: in degenerate/tied data, the same ECG may
    # satisfy multiple criteria. Never substitute a different ranked record.
    return selected


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    require(name in ("cpu", "cuda") and (name != "cuda" or torch.cuda.is_available()), "Requested device unavailable")
    return torch.device(name)


def infer_validation(model, loader, device):
    model.to(device).eval()
    predictions, historical_total, records = [], 0.0, 0
    with torch.inference_mode():
        for inputs, targets in loader:
            output = model(inputs.to(device))
            target_device = targets.to(device)
            require(output.shape == target_device.shape and output.shape[1:] == (11, 1000)
                    and torch.isfinite(output).all().item(), "Invalid model output")
            predictions.append(output.cpu().numpy())
            historical_total += torch.nn.functional.mse_loss(output, target_device).item() * len(inputs)
            records += len(inputs)
    require(records == 100, "Exactly 100 validation records must be inferred")
    return np.concatenate(predictions, axis=0), historical_total / records


def selected_metric_rows(cases, identities, baseline_metrics, candidate_metrics):
    rows = []
    for case in cases:
        index = case["validation_position"]
        require(identities[index]["ecg_id"] == case["ecg_id"], "Selected identity mismatch")
        for column, (scope, lead) in enumerate(SCOPES):
            row = {**identities[index], **case, "scope": scope, "lead": lead,
                   "selection_delta_overall_mse_mV2": case["candidate_minus_baseline_mse_mV2"],
                   "samples_per_record": 1000, "metrics_time_range_s": "full 10-second record"}
            for metric in METRICS:
                left, right = float(baseline_metrics[metric][index, column]), float(candidate_metrics[metric][index, column])
                row.update({f"baseline_{metric}": left, f"candidate_{metric}": right,
                            f"candidate_minus_baseline_{metric}": right - left})
            row["metric_change"] = mse_change(row["candidate_minus_baseline_mse_mV2"])
            rows.append(row)
    return rows


def voltage_limits(lead_i, targets, baseline, candidate):
    """Per-lead limits include all full-record curves; reused for both windows."""
    require(lead_i.shape == (1000,) and targets.shape == baseline.shape == candidate.shape == (11, 1000),
            "Invalid plotting shapes")
    limits = []
    for channel in range(12):
        values = lead_i if channel == 0 else np.concatenate((targets[channel-1], baseline[channel-1], candidate[channel-1]))
        require(np.isfinite(values).all(), "Nonfinite plotting data")
        low, high = float(np.min(values)), float(np.max(values))
        margin = max((high - low) * 0.08, 0.02)
        limits.append((low - margin, high + margin))
    return limits


def save_case_plot(path, case, identity, lead_i, targets, baseline, candidate, limits, window):
    time = np.arange(1000) / 100.0
    visible = (time >= window[0]) & (time <= window[1])
    fig, axes = plt.subplots(12, 1, figsize=(17, 25), sharex=True, constrained_layout=True)
    try:
        axes[0].plot(time[visible], lead_i[visible], color="tab:green", linewidth=1, label="Input Lead I (observed)")
        for channel, axis in enumerate(axes[1:]):
            axis.plot(time[visible], targets[channel, visible], color="black", linewidth=1.0, label="Ground truth")
            axis.plot(time[visible], baseline[channel, visible], color="tab:blue", linestyle="--", linewidth=0.9,
                      alpha=0.85, label="Baseline prediction")
            axis.plot(time[visible], candidate[channel, visible], color="tab:orange", linewidth=0.9,
                      alpha=0.85, label="Dilated prediction")
        for axis, lead, bound in zip(axes, LEADS, limits, strict=True):
            axis.set_ylabel(f"{lead} (mV)")
            axis.set_ylim(*bound)
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlim(*window)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Validation-selected case: {case['category']} | ECG {identity['ecg_id']} | "
                     f"patient {identity['patient_id']} | original index {identity['original_index']}\n"
                     f"Candidate - baseline overall MSE = {case['candidate_minus_baseline_mse_mV2']:.8g} mV^2 "
                     f"({case['overall_change']}) | {window[0]:g}-{window[1]:g} s\n{SELECTION_NOTE}", fontsize=12)
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160)
    finally:
        plt.close(fig)


def validated_inputs(args):
    arrays, masks, description, baseline, bc, br, candidate, cc, cr = prepare(args)
    require(bc["seed"] == cc["seed"] == 42 and
            bc["experiment"]["settings"]["seed"] == cc["experiment"]["settings"]["seed"] == 42,
            "This presentation script requires seed 42 checkpoints")
    validation = masks["validation"]
    require(int(validation.sum()) == 100 and np.array_equal(validation, arrays["split"].astype(str) == "validation"),
            "Expected original 100-record validation mask")
    require(arrays["ecg_id"][validation].tolist() == description["split_ecg_ids"]["validation"], "Validation order mismatch")
    require(description["lead_names"] == LEADS and description["sampling_rate_hz"] == 100, "Lead/rate mismatch")
    return arrays, validation, description, baseline, bc, br, candidate, cc, cr


def run(args):
    arrays, validation, description, baseline, bc, br, candidate, cc, cr = validated_inputs(args)
    if args.validate_only:
        print("Seed-42 validation metadata PASS; no waveform inference or output writes.")
        return
    device = device_for(args.device)
    indices = np.flatnonzero(validation)
    identities = [{"original_index": int(index), "ecg_id": int(arrays["ecg_id"][index]),
                   "patient_id": int(arrays["patient_id"][index]), "split": "validation"} for index in indices]
    x = arrays["X"][validation]
    targets = np.ascontiguousarray(arrays["Y"][validation].transpose(0, 2, 1))
    loader = make_loader(x, arrays["Y"][validation], SETTINGS["batch_size"], shuffle=False)
    del arrays
    output = OUTPUT_ROOT / new_id()
    output.mkdir(parents=True, exist_ok=False)
    execution = environment(device)
    write_json(output / "comparison_config.json", {
        "seed": 42, "split": "validation", "records": 100, "baseline": br, "candidate": cr,
        "baseline_epoch": bc["epoch"], "candidate_epoch": cc["epoch"],
        "dataset_sha256": description["dataset_sha256"], "validation_ecg_ids": description["split_ecg_ids"]["validation"],
        "lead_names": LEADS, "sampling_rate_hz": 100, "voltage_unit": "mV", "environment": execution,
        "selection_note": SELECTION_NOTE, "delta_definition": "candidate overall MSE - baseline overall MSE",
        "tie_break": "ecg_id ascending; independent criteria may select the same record",
        "selection_rules": ["minimum delta", "maximum delta", "nearest to median delta"],
        "metrics": "float64 over full 10 s; overall excludes input Lead I",
        "change_columns": {
            "overall_change": "Sign of candidate-minus-baseline record overall MSE, repeated across that record's rows",
            "metric_change": "Sign of candidate_minus_baseline_mse_mV2 for the current overall/per_lead row; not MAE or RMSE",
            "labels": "negative: improved; positive: worsened; exactly zero: equal; no tolerance"},
        "plot_windows_s": [[0, 10], [0, 2]], "axes": "all three curves share each target panel; same per-lead voltage limits for full and zoom",
        "source_sha256": {name: sha256_file(PROJECT / name) for name in
                          ("plot_validation_waveforms.py", "compare_validation_1000.py", "analyze_validation_records.py",
                           "train_dilated_1000.py", "train_1000.py", "model.py", "model_dilated.py")}})
    left, historical_left = infer_validation(baseline, loader, device)
    right, historical_right = infer_validation(candidate, loader, device)
    baseline_metrics, candidate_metrics = per_record_metrics(left, targets), per_record_metrics(right, targets)
    checks = {}
    for role, values, historical, checkpoint in (("baseline", baseline_metrics, historical_left, bc),
                                                 ("candidate", candidate_metrics, historical_right, cc)):
        pooled = [{"mse_mV2": float(values["mse_mV2"][:, 0].mean(dtype=np.float64))}]
        checks[role] = loss_check(pooled, historical, checkpoint, execution)
    write_json(output / "validation_loss_checks.json", checks)
    require(all(check["matches"] for check in checks.values()),
            f"Saved validation loss mismatch. See {output / 'validation_loss_checks.json'}; no cases plotted.")
    cases = select_cases([row["ecg_id"] for row in identities], baseline_metrics["mse_mV2"][:, 0], candidate_metrics["mse_mV2"][:, 0])
    selected = [{**identities[case["validation_position"]], **case} for case in cases]
    write_csv(output / "selected_records.csv", selected)
    write_csv(output / "selected_record_metrics.csv", selected_metric_rows(cases, identities, baseline_metrics, candidate_metrics))
    write_csv(output / "all_validation_record_mse.csv", [
        {**identity, "baseline_mse_mV2": float(baseline_metrics["mse_mV2"][i, 0]),
         "candidate_mse_mV2": float(candidate_metrics["mse_mV2"][i, 0]),
         "candidate_minus_baseline_mse_mV2": float(candidate_metrics["mse_mV2"][i, 0] - baseline_metrics["mse_mV2"][i, 0])}
        for i, identity in enumerate(identities)])
    print(SELECTION_NOTE)
    for case in cases:
        index = case["validation_position"]
        identity = identities[index]
        limits = voltage_limits(x[index, :, 0], targets[index], left[index], right[index])
        for window, suffix in (((0, 10), "full_10s"), ((0, 2), "zoom_0_2s")):
            path = output / f"{case['category']}_ecg_{case['ecg_id']}_{suffix}.png"
            save_case_plot(path, case, identity, x[index, :, 0], targets[index], left[index], right[index], limits, window)
        print(f"{case['category']}: {identity}; delta MSE={case['candidate_minus_baseline_mse_mV2']:.9g}; {case['overall_change']}")
    print(f"Results: {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--validate-only", action="store_true", help="Check metadata only; no waveform inference or writes.")
    return parser.parse_args(argv)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
