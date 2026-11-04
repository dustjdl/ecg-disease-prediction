"""Compare explicit baseline/candidate checkpoints on validation only.

No split selector exists. --validate-only checks metadata without waveform
inference or output files. Model selection remains lowest validation MSE.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from train import make_loader
from train_1000 import DATASET, PROJECT, SETTINGS, environment, load_and_select, new_id, require, write_json
from train_dilated_1000 import load_checked_checkpoint

MSE_ATOL = 1e-7
MSE_RTOL = 1e-5


def metric_rows(sums_abs, sums_squared, records):
    """Float64 pooled metrics, never an average of record RMSEs."""
    count = records * 1000
    from train_1000 import LEADS
    rows = []
    for lead, absolute, squared, elements in [
        ("ALL_11_TARGET_LEADS", sums_abs.sum(dtype=np.float64), sums_squared.sum(dtype=np.float64), count * 11),
        *[(name, sums_abs[i], sums_squared[i], count) for i, name in enumerate(LEADS[1:])],
    ]:
        mse = float(squared / elements)
        rows.append({"scope": "overall" if lead == "ALL_11_TARGET_LEADS" else "per_lead", "lead": lead,
                     "mse_mV2": mse, "mae_mV": float(absolute / elements), "rmse_mV": float(np.sqrt(mse)),
                     "records": records, "samples_per_record": 1000})
    return rows


def validation_metrics(model, loader, device):
    """Validation-only batched inference; cast before subtraction and summation."""
    model.to(device).eval()
    sums_abs, sums_squared = np.zeros(11, dtype=np.float64), np.zeros(11, dtype=np.float64)
    records = 0
    # Also reproduce the historical float32 MSE reduction for mismatch diagnosis.
    historical_loss = 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs)
            require(predicted.shape == targets.shape and predicted.shape[1:] == (11, 1000), "Invalid output shape")
            require(torch.isfinite(predicted).all().item(), "Nonfinite predictions")
            error = predicted.cpu().numpy().astype(np.float64) - targets.cpu().numpy().astype(np.float64)
            sums_abs += np.abs(error).sum(axis=(0, 2), dtype=np.float64)
            sums_squared += np.square(error).sum(axis=(0, 2), dtype=np.float64)
            historical_loss += torch.nn.functional.mse_loss(predicted, targets).item() * len(inputs)
            records += len(inputs)
    require(records == 100, f"Expected 100 validation records, got {records}")
    return metric_rows(sums_abs, sums_squared, records), historical_loss / records


def loss_check(rows, historical_loss, checkpoint, current_environment):
    recalculated, saved = rows[0]["mse_mV2"], float(checkpoint["validation_loss"])
    matches = bool(np.isclose(recalculated, saved, atol=MSE_ATOL, rtol=MSE_RTOL))
    historical_matches = bool(np.isclose(historical_loss, saved, atol=MSE_ATOL, rtol=MSE_RTOL))
    differences = {key: {"saved": value, "current": current_environment.get(key)}
                   for key, value in checkpoint["experiment"]["environment"].items()
                   if current_environment.get(key) != value}
    if matches:
        diagnosis = "within tolerance"
    elif historical_matches:
        diagnosis = "Float64 aggregation differs from historical float32 MSE reduction beyond tolerance."
    else:
        diagnosis = ("Both float64 and historical-style float32 MSE differ. Data/IDs/model structure were verified. "
                     "Device/library differences, numerical execution differences, or saved loss/state inconsistency "
                     "remain possible; the exact cause cannot be established automatically.")
    return {"saved_validation_loss": saved, "recalculated_mse_mV2_float64": recalculated,
            "historical_style_mse_mV2_float32": historical_loss,
            "absolute_difference": abs(recalculated - saved), "atol": MSE_ATOL, "rtol": MSE_RTOL,
            "matches": matches, "historical_style_matches": historical_matches,
            "environment_differences": differences, "diagnosis": diagnosis}


def compare_rows(baseline, candidate):
    rows = []
    for left, right in zip(baseline, candidate, strict=True):
        require(left["lead"] == right["lead"], "Metric lead order differs")
        row = {"scope": left["scope"], "lead": left["lead"], "records": 100, "samples_per_record": 1000}
        for metric in ("mse_mV2", "mae_mV", "rmse_mV"):
            delta = right[metric] - left[metric]
            row.update({f"baseline_{metric}": left[metric], f"candidate_{metric}": right[metric],
                        f"candidate_minus_baseline_{metric}": delta,
                        f"improved_{metric}": delta < 0,
                        f"change_{metric}": "improved" if delta < 0 else "worsened" if delta > 0 else "equal"})
        rows.append(row)
    return rows


def prepare(args):
    arrays, masks, description = load_and_select(args.dataset)
    baseline, baseline_checkpoint, baseline_reference = load_checked_checkpoint(args.baseline_checkpoint, "baseline", description)
    candidate, candidate_checkpoint, candidate_reference = load_checked_checkpoint(args.candidate_checkpoint, "candidate", description)
    reference = candidate_checkpoint["experiment"]["baseline_reference"]
    require(reference.get("sha256") == baseline_reference["sha256"] and
            reference.get("run_id") == baseline_reference["run_id"], "Candidate refers to a different baseline checkpoint")
    return arrays, masks, description, baseline, baseline_checkpoint, baseline_reference, candidate, candidate_checkpoint, candidate_reference


def compare(args):
    arrays, masks, description, baseline, bc, br, candidate, cc, cr = prepare(args)
    if args.validate_only:
        print("Metadata validation PASS; no validation/test inference or output writes.")
        return
    # Hard-coded validation mask; test cannot be selected through CLI.
    loader = make_loader(arrays["X"][masks["validation"]], arrays["Y"][masks["validation"]],
                         SETTINGS["batch_size"], shuffle=False)
    del arrays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_environment = environment(device)
    output = PROJECT / "results/validation_comparison_1000" / new_id("comparison")
    output.mkdir(parents=True, exist_ok=False)
    config = {"split": "validation", "validation_ecg_ids": description["split_ecg_ids"]["validation"],
              "dataset_sha256": description["dataset_sha256"], "baseline": br, "candidate": cr,
              "baseline_epoch": bc["epoch"], "candidate_epoch": cc["epoch"],
              "environment": current_environment, "batch_size": 8,
              "aggregation": "float64 errors and sums; pooled overall and per-lead; Lead I excluded",
              "selection_rule": "lowest validation MSE; MAE/RMSE are secondary comparisons",
              "mse_tolerance": {"atol": MSE_ATOL, "rtol": MSE_RTOL}}
    write_json(output / "comparison_config.json", config)
    baseline_rows, baseline_historical = validation_metrics(baseline, loader, device)
    candidate_rows, candidate_historical = validation_metrics(candidate, loader, device)
    checks = {"baseline": loss_check(baseline_rows, baseline_historical, bc, current_environment),
              "candidate": loss_check(candidate_rows, candidate_historical, cc, current_environment)}
    write_json(output / "validation_loss_checks.json", checks)
    rows = compare_rows(baseline_rows, candidate_rows)
    for row in rows:
        row["baseline_loss_matches"] = checks["baseline"]["matches"]
        row["candidate_loss_matches"] = checks["candidate"]["matches"]
        row["comparison_validated"] = all(item["matches"] for item in checks.values())
    with (output / "validation_metrics_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for role, metrics in (("baseline", baseline_rows), ("candidate", candidate_rows)):
        overall = metrics[0]
        print(f"{role} overall validation MSE={overall['mse_mV2']:.10g} mV^2; "
              f"MAE={overall['mae_mV']:.10g} mV; RMSE={overall['rmse_mV']:.10g} mV")
        print(f"{role} saved-loss check: {checks[role]}")
    print(f"Comparison results: {output}")
    if not all(check["matches"] for check in checks.values()):
        raise RuntimeError("Validation loss mismatch; comparison is unverified. Diagnostics were saved; do not select a model from these results.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-only", action="store_true", help="Validate checkpoint/data compatibility only; no inference or writes.")
    return parser.parse_args(argv)


def main():
    compare(parse_args())


if __name__ == "__main__":
    main()
