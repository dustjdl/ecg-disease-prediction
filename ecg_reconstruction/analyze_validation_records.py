"""Validation-only record distributions for explicit baseline/candidate checkpoints.

Reuse compare_validation_1000.prepare for dataset, ordered IDs, architecture,
state and baseline-reference validation. There is no test/split selector.
Every execution creates a new output folder; input artifacts are read-only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from compare_validation_1000 import MSE_ATOL, MSE_RTOL, prepare
from train import make_loader
from train_1000 import (DATASET, LEADS, PROJECT, SETTINGS, environment, new_id,
                        require, sha256_file, write_json)

METRICS = ("mse_mV2", "mae_mV", "rmse_mV")
SCOPES = [("overall", "ALL_11_TARGET_LEADS"), *[("per_lead", name) for name in LEADS[1:]]]


def per_record_metrics(predictions, targets):
    """Return [records, overall+11 leads] arrays; cast before subtraction."""
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.float64)
    require(predicted.shape == actual.shape and actual.ndim == 3
            and actual.shape[1:] == (11, 1000) and len(actual) > 0,
            "Expected matching nonempty [records, 11, 1000] arrays")
    require(np.isfinite(predicted).all() and np.isfinite(actual).all(), "Nonfinite waveforms")
    error = predicted - actual
    squared, absolute = np.square(error), np.abs(error)
    mse = np.column_stack((squared.mean(axis=(1, 2), dtype=np.float64),
                           squared.mean(axis=2, dtype=np.float64)))
    mae = np.column_stack((absolute.mean(axis=(1, 2), dtype=np.float64),
                           absolute.mean(axis=2, dtype=np.float64)))
    return {"mse_mV2": mse, "mae_mV": mae, "rmse_mV": np.sqrt(mse)}


def infer_validation_records(model, loader, device):
    model.to(device).eval()
    batches = {key: [] for key in METRICS}
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs.to(device))
            values = per_record_metrics(predictions.cpu().numpy(), targets.numpy())
            for key in METRICS:
                batches[key].append(values[key])
    combined = {key: np.concatenate(batches[key], axis=0) for key in METRICS}
    require(all(value.shape == (100, 12) for value in combined.values()),
            "Expected metrics for exactly 100 validation records")
    return combined


def validate_metric_arrays(values):
    shape = values["mse_mV2"].shape
    require(len(shape) == 2 and shape[0] > 0 and shape[1] == 12, "Invalid metric array shape")
    require(all(values[key].shape == shape and np.isfinite(values[key]).all()
                and (values[key] >= 0).all() for key in METRICS), "Invalid metric values")
    return shape[0]


def compare_records(baseline, candidate, identities):
    records = validate_metric_arrays(baseline)
    require(validate_metric_arrays(candidate) == records and len(identities) == records,
            "Record identities/metric shapes differ")
    rows = []
    for index, identity in enumerate(identities):
        for column, (scope, lead) in enumerate(SCOPES):
            row = {**identity, "scope": scope, "lead": lead, "samples_per_record": 1000}
            for key in METRICS:
                left, right = float(baseline[key][index, column]), float(candidate[key][index, column])
                delta = right - left
                row.update({f"baseline_{key}": left, f"candidate_{key}": right,
                            f"candidate_minus_baseline_{key}": delta,
                            f"change_{key}": "improved" if delta < 0 else "worsened" if delta > 0 else "equal"})
            rows.append(row)
    delta = candidate["mse_mV2"][:, 0] - baseline["mse_mV2"][:, 0]
    counts = {"improved": int(np.count_nonzero(delta < 0)),
              "worsened": int(np.count_nonzero(delta > 0)),
              "equal": int(np.count_nonzero(delta == 0))}
    return rows, counts


def distribution_summary(values, model_name):
    """Record distributions and pooled metrics for fixed-length equal-size ECGs."""
    records = validate_metric_arrays(values)
    rows = []
    for column, (scope, lead) in enumerate(SCOPES):
        row = {"model": model_name, "scope": scope, "lead": lead, "records": records,
               "samples_per_record": 1000}
        for key in METRICS:
            distribution = values[key][:, column]
            row[f"record_mean_{key}"] = float(distribution.mean(dtype=np.float64))
            q25, median, q75, q90 = np.percentile(distribution, [25, 50, 75, 90], method="linear")
            row.update({f"record_p25_{key}": float(q25), f"record_median_{key}": float(median),
                        f"record_p75_{key}": float(q75), f"record_p90_{key}": float(q90)})
        # Every record contributes the same number of samples/leads. Thus mean
        # record MSE/MAE equal pooled MSE/MAE; pooled RMSE takes sqrt AFTER pooling.
        row["pooled_mse_mV2"] = row["record_mean_mse_mV2"]
        row["pooled_mae_mV"] = row["record_mean_mae_mV"]
        row["pooled_rmse_mV"] = float(np.sqrt(row["pooled_mse_mV2"]))
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    # Existing prepare validates both actual model classes, dilation/padding,
    # sources, dataset hash, ordered split IDs, exclusions and candidate lineage.
    arrays, masks, description, baseline, bc, br, candidate, cc, cr = prepare(args)
    validation = masks["validation"]
    require(np.array_equal(validation, arrays["split"].astype(str) == "validation")
            and int(validation.sum()) == 100, "Expected original 100-record validation mask")
    indices = np.flatnonzero(validation)
    require(arrays["ecg_id"][validation].tolist() == description["split_ecg_ids"]["validation"],
            "Validation ID order mismatch")
    identities = [{"ecg_id": int(arrays["ecg_id"][i]), "patient_id": int(arrays["patient_id"][i]),
                   "npz_index": int(i), "split": "validation"} for i in indices]
    loader = make_loader(arrays["X"][validation], arrays["Y"][validation],
                         SETTINGS["batch_size"], shuffle=False)
    del arrays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = new_id()
    output = PROJECT / "results/validation_record_analysis" / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "split": "validation", "records": 100,
              "validation_ecg_ids": description["split_ecg_ids"]["validation"],
              "dataset_path": str(args.dataset.resolve()), "dataset_sha256": description["dataset_sha256"],
              "baseline": br, "candidate": cr, "baseline_epoch": bc["epoch"], "candidate_epoch": cc["epoch"],
              "batch_size": SETTINGS["batch_size"], "environment": environment(device),
              "aggregation": "Cast predictions and targets to float64 before subtraction; Lead I excluded. Per-record overall pools 11 leads and time; per-lead pools time.",
              "distribution": "Across records; NumPy percentile method=linear; p25/median/p75/p90.",
              "pooled_definition": "Fixed 1000 samples per record: pooled MSE/MAE equal record means; pooled RMSE=sqrt(mean(record MSE)), not mean(record RMSE).",
              "tie_rule": "Exact equality of float64 overall MSE; negative candidate-baseline is improvement.",
              "source_sha256": {name: sha256_file(PROJECT / name) for name in
                                ("analyze_validation_records.py", "compare_validation_1000.py",
                                 "train_dilated_1000.py", "train_1000.py", "train.py")}}
    write_json(output / "analysis_config.json", config)
    baseline_values = infer_validation_records(baseline, loader, device)
    candidate_values = infer_validation_records(candidate, loader, device)
    record_rows, counts = compare_records(baseline_values, candidate_values, identities)
    summaries = distribution_summary(baseline_values, "baseline") + distribution_summary(candidate_values, "candidate")
    write_csv(output / "record_metrics_comparison.csv", record_rows)
    write_csv(output / "record_distribution_summary.csv", summaries)
    saved_loss_checks = {}
    for name, checkpoint in (("baseline", bc), ("candidate", cc)):
        pooled = next(row["pooled_mse_mV2"] for row in summaries if row["model"] == name and row["scope"] == "overall")
        saved = float(checkpoint["validation_loss"])
        saved_loss_checks[name] = {"saved_validation_loss": saved, "pooled_mse_mV2": pooled,
                                   "matches": bool(np.isclose(pooled, saved, atol=MSE_ATOL, rtol=MSE_RTOL)),
                                   "atol": MSE_ATOL, "rtol": MSE_RTOL}
    write_json(output / "analysis_summary.json", {"overall_mse_record_counts": counts,
               "tie_rule": config["tie_rule"], "saved_loss_checks": saved_loss_checks})
    print(f"Overall record MSE: improved={counts['improved']}, worsened={counts['worsened']}, equal={counts['equal']}")
    print("Record-mean RMSE and pooled RMSE are different aggregates; they may coincide only in special cases.")
    print("For these equal-size records, record-mean MSE/MAE equal pooled MSE/MAE.")
    for row in summaries:
        print(f"{row['model']} {row['scope']} {row['lead']}: record-mean MSE={row['record_mean_mse_mV2']:.9g} mV^2; "
              f"pooled MSE={row['pooled_mse_mV2']:.9g} mV^2")
        for key in ("mae_mV", "rmse_mV"):
            print(f"  {key}: record mean={row[f'record_mean_{key}']:.9g}; "
                  f"median={row[f'record_median_{key}']:.9g}; p25={row[f'record_p25_{key}']:.9g}; "
                  f"p75={row[f'record_p75_{key}']:.9g}; p90={row[f'record_p90_{key}']:.9g}; "
                  f"pooled={row[f'pooled_{key}']:.9g}")
    print(f"Saved-loss checks: {saved_loss_checks}\nResults: {output}")
    if not all(check["matches"] for check in saved_loss_checks.values()):
        raise RuntimeError("Recalculated pooled MSE differs from saved validation_loss. Results were saved but require investigation of numerical/device differences or saved state/loss consistency.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    return parser.parse_args(argv)


def main():
    analyze(parse_args())


if __name__ == "__main__":
    main()
