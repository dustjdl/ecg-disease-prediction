"""Evaluate six fixed best-validation checkpoints on the same 100 test ECGs.

Seeds 42/43/44 are required, each with an explicit baseline and candidate path.
--validate-only loads data/checkpoints and checks metadata/history, but creates
no test DataLoader, performs no forward pass, and writes no output files.
Test results never choose a model, seed, checkpoint, or waveform example.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_validation_1000 import compare_rows, metric_rows
from model import ECGModel
from model_dilated import ECGDilatedModel
from multiseed_common import load_checkpoint, order_digest, source_manifest, validate_pair
from train_1000 import (DATASET, LEADS, PREPROCESSING, PROJECT, environment,
                        load_and_select, new_id, require, sha256_file, write_json)
from train_dilated_1000 import structure

SEEDS = (42, 43, 44)
ROLES = ("baseline", "candidate")
METRICS = ("mse_mV2", "mae_mV", "rmse_mV")
OUTPUT_ROOT = PROJECT / "results/test_comparison_multiseed"
POLICY = {
    "checkpoint_selection": "Each explicit checkpoint was selected by lowest validation MSE, before this test evaluation.",
    "test_selection": "No model, seed, checkpoint or waveform case is selected using these test results.",
    "prior_test_exposure": "The existing seed-42 baseline test result was already inspected, as disclosed by the user; this is not a wholly unseen-test workflow.",
    "seed_interpretation": "Three training seeds on the same data split and the same 100 test ECGs; not three independent test cohorts.",
    "seed_mean_rmse": "Arithmetic mean of three seed-specific pooled RMSE values; SD uses ddof=1.",
    "all_seed_pooled_rmse": "sqrt(total squared error / total elements across three seeds); repeats the same 100 ECGs three times and differs from mean seed RMSE.",
    "aggregation": "Convert predictions and targets separately to float64 before subtraction; float64 sums; overall excludes observed Lead I; RMSE=sqrt(MSE).",
}


def write_csv(path, rows):
    require(bool(rows), "Cannot save an empty metric table")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    require(name in ("cpu", "cuda"), "Unsupported device")
    require(name != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    return torch.device(name)


def check_runtime_structure(model, role):
    """Check executable classes and convolution properties, never a forward pass."""
    require(role in ROLES, "Unsupported model role")
    require(type(model) is (ECGModel if role == "baseline" else ECGDilatedModel), "Wrong runtime model class")
    layers = list(model.network.children())
    require(len(layers) == 7, "Expected four convolutions and three ReLUs")
    dilation = (1, 1, 1, 1) if role == "baseline" else (1, 2, 4, 1)
    padding = (3, 2, 2, 0) if role == "baseline" else (3, 4, 8, 0)
    channels = ((1, 32), (32, 64), (64, 64), (64, 11))
    for i, (incoming, outgoing) in enumerate(channels):
        layer = layers[2 * i]
        require(type(layer) is nn.Conv1d and layer.in_channels == incoming and layer.out_channels == outgoing
                and layer.kernel_size == ((7, 5, 5, 1)[i],) and layer.stride == (1,)
                and layer.dilation == (dilation[i],) and layer.padding == (padding[i],)
                and layer.bias is not None and layer.groups == 1 and layer.padding_mode == "zeros",
                f"Wrong executable convolution for {role}, layer {i + 1}")
        if i < 3:
            require(type(layers[2 * i + 1]) is nn.ReLU and not layers[2 * i + 1].inplace, "Wrong activation")
    actual = structure(model)
    require(actual["parameter_count"] == 31819 and actual["receptive_field_samples"] == (15 if role == "baseline" else 31),
            "Parameter count/receptive field mismatch")
    return actual


def check_best_history(checkpoint, role):
    """Verify the checkpoint's selected epoch against saved training history only."""
    metadata = checkpoint["experiment"]
    legacy = "training_protocol" not in metadata
    group = ("baseline_1000" if role == "baseline" else "dilated_1000") if legacy else (
        "baseline_multiseed" if role == "baseline" else "dilated_multiseed")
    path = PROJECT / "results" / group / metadata["run_id"] / "training_history.csv"
    before = sha256_file(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(before == sha256_file(path), "Training history changed while reading")
    require([int(row["epoch"]) for row in rows] == list(range(1, 31)), "Incomplete training history")
    losses = [float(row["validation_loss"]) for row in rows]
    require(np.isfinite(losses).all() and min(losses) >= 0, "Invalid validation history")
    best = int(np.argmin(losses))  # First minimum matches strictly-lower selection.
    require(checkpoint["epoch"] == best + 1 and
            np.isclose(float(checkpoint["validation_loss"]), losses[best], rtol=1e-12, atol=1e-12),
            "Checkpoint does not match the saved best-validation epoch/loss")
    return {"path": str(path.resolve()), "sha256": before, "best_epoch": best + 1,
            "best_validation_loss": losses[best], "verified": True}


def prepare(args):
    """Finish all six metadata checks before making any test loader or output."""
    arrays, masks, description = load_and_select(args.dataset)
    test = masks["test"]
    expected_ids = description["split_ecg_ids"]["test"]
    require(np.array_equal(test, arrays["split"].astype(str) == "test") and int(test.sum()) == 100,
            "Expected the original unchanged test 100 records")
    require(arrays["ecg_id"][test].tolist() == expected_ids and len(set(expected_ids)) == 100,
            "Invalid ordered test IDs")
    loaded, common_settings = {}, None
    for seed in SEEDS:
        for role in ROLES:
            path = getattr(args, f"{role}_checkpoint_{seed}")
            require(path.name == "best_model.pth", "Explicit best_model.pth checkpoints are required")
            model, checkpoint, reference = load_checkpoint(path, role, description, expected_seed=seed, allow_legacy=True)
            actual = check_runtime_structure(model, role)
            metadata = checkpoint["experiment"]
            require(metadata["split_ecg_ids"]["test"] == expected_ids and
                    metadata["dataset_sha256"] == description["dataset_sha256"], "Test IDs/data hash mismatch")
            settings = {key: value for key, value in metadata["settings"].items() if key != "seed"}
            if common_settings is None:
                common_settings = settings
            require(settings == common_settings, "Non-seed training settings differ across experiments")
            loaded[(seed, role)] = {"model": model, "checkpoint": checkpoint, "reference": reference,
                                    "actual_structure": actual, "best_history": check_best_history(checkpoint, role)}
        validate_pair(loaded[(seed, "baseline")]["checkpoint"], loaded[(seed, "candidate")]["checkpoint"],
                      loaded[(seed, "baseline")]["reference"])
    print("PASS: seeds 42/43/44 pairs, actual classes/dilations, best-validation histories, data hash and ordered test IDs.")
    return arrays, test, description, loaded


def error_sums(predictions, targets):
    """Pure array reducer shared by runtime and synthetic checks; no inference."""
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.float64)
    require(predicted.shape == actual.shape and actual.ndim == 3 and actual.shape[1:] == (11, 1000)
            and len(actual) > 0, "Expected matching nonempty [records,11,1000] arrays")
    require(np.isfinite(predicted).all() and np.isfinite(actual).all(), "Nonfinite waveform values")
    error = predicted - actual
    absolute = np.abs(error).sum(axis=(0, 2), dtype=np.float64)
    squared = np.square(error).sum(axis=(0, 2), dtype=np.float64)
    require(np.isfinite(absolute).all() and np.isfinite(squared).all(), "Nonfinite error sums")
    return absolute, squared


def make_test_loader(arrays, test, batch_size, device):
    x = np.ascontiguousarray(arrays["X"][test].transpose(0, 2, 1))
    y = np.ascontiguousarray(arrays["Y"][test].transpose(0, 2, 1))
    ids = np.ascontiguousarray(arrays["ecg_id"][test].astype(np.int64))
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ids)),
                      batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
                      pin_memory=device.type == "cuda")


def infer_test_metrics(model, loader, device, expected_ids):
    model.to(device).eval()
    absolute, squared = np.zeros(11, dtype=np.float64), np.zeros(11, dtype=np.float64)
    observed = []
    with torch.inference_mode():
        for inputs, targets, ids in loader:
            output = model(inputs.to(device, non_blocking=True))
            batch_absolute, batch_squared = error_sums(output.cpu().numpy(), targets.numpy())
            absolute += batch_absolute
            squared += batch_squared
            observed.extend(ids.tolist())
    require(observed == expected_ids and len(observed) == 100, "Inference did not consume the exact ordered test 100")
    require(np.isfinite(absolute).all() and np.isfinite(squared).all(), "Nonfinite aggregate errors")
    return metric_rows(absolute, squared, len(observed)), absolute, squared, order_digest(observed)


def pair_comparison(seed, baseline, candidate):
    rows = compare_rows(baseline, candidate)
    for row in rows:
        base = row["baseline_mse_mV2"]
        row.update(seed=seed, mse_reduction_percent=(100.0 * (base - row["candidate_mse_mV2"]) / base
                                                   if base > 0 else float("nan")),
                   mse_reduction_status="ok" if base > 0 else "undefined_baseline_mse_zero")
    return rows


def seed_summary(metrics):
    require(set(metrics) == {(seed, role) for seed in SEEDS for role in ROLES}, "Exactly six seed/model metric sets required")
    expected = [("overall", "ALL_11_TARGET_LEADS"), *[("per_lead", lead) for lead in LEADS[1:]]]
    for rows in metrics.values():
        require([(r["scope"], r["lead"]) for r in rows] == expected and
                all(r["records"] == 100 and r["samples_per_record"] == 1000 for r in rows), "Metric scope/count mismatch")
    output = []
    for role in ROLES:
        for column, (scope, lead) in enumerate(expected):
            row = {"model": role, "scope": scope, "lead": lead, "seeds": "42;43;44", "n_seeds": 3,
                   "ddof": 1, "unique_test_records": 100, "aggregation": "mean and sample SD of seed-specific pooled metrics"}
            for metric in METRICS:
                values = np.asarray([metrics[(seed, role)][column][metric] for seed in SEEDS], dtype=np.float64)
                require(np.isfinite(values).all() and (values >= 0).all(), "Invalid seed metrics")
                row.update({f"mean_{metric}": float(values.mean(dtype=np.float64)),
                            f"sample_sd_{metric}": float(values.std(ddof=1, dtype=np.float64))})
            output.append(row)
    return output


def prior_baseline_exposure(run_id):
    root = PROJECT / "results/baseline_1000" / run_id
    return {"already_inspected": True, "basis": "User disclosure before final test evaluation", "note": POLICY["prior_test_exposure"],
            "existing_result_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)}
                                      for path in sorted(root.glob("evaluation_*/test_metrics.csv"))]}


def run(args):
    arrays, test, description, loaded = prepare(args)
    device = device_for(args.device)
    if args.validate_only:
        print("Validation only: no forward, no test DataLoader creation/iteration, no output writes.")
        return
    # All model choices and checks finish before this point. No test-based selection.
    expected_ids = description["split_ecg_ids"]["test"]
    indices = np.flatnonzero(test).tolist()
    test_patients = arrays["patient_id"][test].tolist()
    loader = make_test_loader(arrays, test, args.batch_size, device)
    del arrays
    run_id = new_id()
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for (seed, role), item in loaded.items():
        checkpoint, reference = item["checkpoint"], item["reference"]
        require(sha256_file(Path(reference["path"])) == reference["sha256"], "Checkpoint changed after validation")
        metadata = checkpoint["experiment"]
        manifest.append({"seed": seed, "model": role, "checkpoint": reference,
                         "epoch": checkpoint["epoch"], "validation_loss": checkpoint["validation_loss"],
                         "settings": metadata["settings"], "training_protocol": metadata.get("training_protocol", "legacy_seed42"),
                         "saved_structure": metadata["model_structure"], "actual_structure": item["actual_structure"],
                         "best_history": item["best_history"], "training_source_sha256": metadata["source_sha256"],
                         "training_environment": metadata["environment"], "baseline_reference": metadata.get("baseline_reference"),
                         "test_ecg_ids": metadata["split_ecg_ids"]["test"], "dataset_sha256": metadata["dataset_sha256"]})
    write_json(output / "evaluation_config.json", {
        "run_id": run_id, "split": "test", "seeds": list(SEEDS), "models": manifest,
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": description["dataset_sha256"],
        "original_split_counts": description["original_split_counts"], "used_split_counts": description["used_split_counts"],
        "test_ecg_ids": expected_ids, "test_original_indices": indices, "test_patient_ids": test_patients,
        "test_ecg_id_order_sha256": order_digest(expected_ids), "lead_names": LEADS, "sampling_rate_hz": 100,
        "voltage_unit": "mV", "preprocessing": PREPROCESSING, "exclusion_policy": loaded[(42, "baseline")]["checkpoint"]["experiment"]["exclusion_policy"],
        "inference": {"batch_size": args.batch_size, "shuffle": False, "drop_last": False, "num_workers": 0,
                      "eval": True, "inference_mode": True, "device": str(device)},
        "environment": environment(device), "policy": POLICY,
        "prior_seed42_baseline_test_exposure": prior_baseline_exposure(loaded[(42, "baseline")]["reference"]["run_id"]),
        "source_sha256": {**source_manifest(), "evaluate_test_multiseed.py": sha256_file(Path(__file__))}})
    metrics, totals = {}, {role: [np.zeros(11, dtype=np.float64), np.zeros(11, dtype=np.float64)] for role in ROLES}
    rows, orders = [], []
    for (seed, role), item in loaded.items():
        values, absolute, squared, observed_hash = infer_test_metrics(item["model"], loader, device, expected_ids)
        metrics[(seed, role)] = values
        rows.extend({"seed": seed, "model": role, **row} for row in values)
        totals[role][0] += absolute
        totals[role][1] += squared
        orders.append({"seed": seed, "model": role, "records": 100, "ecg_id_order_sha256": observed_hash,
                       "matches_expected": observed_hash == order_digest(expected_ids)})
        item["model"].to("cpu")
        print(f"seed {seed} {role}: {values[0]}")
    comparisons = [row for seed in SEEDS for row in pair_comparison(seed, metrics[(seed, "baseline")], metrics[(seed, "candidate")])]
    pooled = [{"model": role, **row, "n_seeds": 3, "unique_test_records": 100,
               "aggregation": "all-seed pooled errors: 300 record predictions on the same 100 ECGs; not seed mean"}
              for role in ROLES for row in metric_rows(*totals[role], records=300)]
    write_csv(output / "test_metrics.csv", rows)
    write_csv(output / "test_metrics_comparison.csv", comparisons)
    write_csv(output / "seed_metric_summary.csv", seed_summary(metrics))
    write_csv(output / "all_seed_pooled_metrics.csv", pooled)
    write_json(output / "completion.json", {"complete": True, "model_count": 6, "inference_orders": orders,
                                            "policy": POLICY, "output_files": ["test_metrics.csv", "test_metrics_comparison.csv",
                                            "seed_metric_summary.csv", "all_seed_pooled_metrics.csv"]})
    print(POLICY["seed_mean_rmse"])
    print(POLICY["all_seed_pooled_rmse"])
    print(f"Results: {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        for role in ROLES:
            parser.add_argument(f"--{role}-checkpoint-{seed}", type=Path, required=True,
                                help=f"Explicit seed {seed} {role} best_model.pth path.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true", help="Metadata/history checks only; no forward, test loader or writes.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
