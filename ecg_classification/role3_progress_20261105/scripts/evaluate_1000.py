"""Evaluate an explicitly selected baseline_1000 checkpoint on 100 held-out ECGs.

Each evaluation writes a new folder under its training run. --validate-only
checks compatibility without test inference or output files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import torch

from evaluate import (calculate_metrics, load_compatible_model, run_inference,
                      save_first_comparison_plot, save_metrics_csv)
from train_1000 import (DATASET, EXCLUSION_POLICY, PREPROCESSING, PROJECT, SETTINGS,
                        environment, load_and_select, model_structure, new_id, require,
                        sha256_file, write_json)


def validate_experiment(checkpoint, model, description):
    metadata = checkpoint.get("experiment")
    require(isinstance(metadata, dict), "Missing baseline_1000 experiment metadata")
    require(metadata.get("schema_version") == 1, "Unsupported experiment schema")
    run_id = metadata.get("run_id", "")
    require(isinstance(run_id, str) and re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{32}", run_id) is not None,
            "Invalid experiment run_id")
    for key in ("dataset_sha256", "original_split_counts", "used_split_counts", "split_ecg_ids",
                "excluded_records", "lead_names", "sampling_rate_hz", "plot_selection"):
        require(metadata.get(key) == description[key], f"Checkpoint/data mismatch: {key}")
    require(metadata.get("model_structure") == model_structure(model), "Model structure/source mismatch")
    require(metadata.get("exclusion_policy") == EXCLUSION_POLICY, "Exclusion policy mismatch")
    require(metadata.get("preprocessing") == PREPROCESSING, "Preprocessing mismatch")
    require(metadata.get("settings") == SETTINGS, "Baseline settings mismatch")
    for key in ("seed", "batch_size", "learning_rate"):
        require(checkpoint.get(key) == SETTINGS[key], f"Checkpoint setting mismatch: {key}")
    require(isinstance(checkpoint.get("epoch"), int) and 1 <= checkpoint["epoch"] <= 30, "Invalid best epoch")
    require(isinstance(checkpoint.get("validation_loss"), (float, int)) and
            np.isfinite(checkpoint["validation_loss"]), "Invalid validation loss")
    print(f"Experiment compatibility PASS: {run_id}; SHA-256, model, leads, rate and ordered ECG IDs match")
    return metadata


def evaluate_checkpoint(args):
    arrays, masks, description = load_and_select(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_digest = sha256_file(args.checkpoint)
    model, checkpoint = load_compatible_model(args.checkpoint, device)
    require(sha256_file(args.checkpoint) == checkpoint_digest, "Checkpoint changed during loading")
    metadata = validate_experiment(checkpoint, model, description)
    if args.validate_only:
        print("Validation only: no test inference or output writes.")
        return
    # This mask is the original, unchanged 100-record test split in NPZ order.
    test_mask = masks["test"]
    x_test = np.ascontiguousarray(arrays["X"][test_mask].transpose(0, 2, 1))
    y_test = np.ascontiguousarray(arrays["Y"][test_mask].transpose(0, 2, 1))
    del arrays
    run_dir = PROJECT / "results/baseline_1000" / metadata["run_id"]
    require(run_dir.is_dir(), f"Training result folder missing: {run_dir}")
    evaluation_id = new_id("evaluation")
    output = run_dir / evaluation_id
    output.mkdir(exist_ok=False)
    write_json(output / "evaluation_config.json", {
        "run_id": metadata["run_id"], "evaluation_id": evaluation_id,
        "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": checkpoint["epoch"], "validation_loss": checkpoint["validation_loss"],
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": description["dataset_sha256"],
        "test_ecg_ids": description["split_ecg_ids"]["test"],
        "plot_selection": description["plot_selection"], "batch_size": args.batch_size,
        "preprocessing": PREPROCESSING, "environment": environment(device),
        "metric_aggregation": "Existing evaluate.calculate_metrics: pooled overall and pooled per-lead; Lead I excluded",
        "source_sha256": {name: sha256_file(PROJECT / name) for name in ("evaluate_1000.py", "evaluate.py")},
    })
    # Existing function uses eval() + inference_mode(); no optimizer/backward.
    predictions = run_inference(model, x_test, args.batch_size, device)
    require(np.isfinite(predictions).all(), "Nonfinite model predictions")
    rows = calculate_metrics(predictions, y_test, tuple(description["lead_names"]))
    for row in rows:
        # Existing zero-variance reasons are retained. Handle any other undefined correlation too.
        if not np.isfinite(row["pearson_correlation"]) and row["correlation_status"] == "ok":
            row["pearson_correlation"] = float("nan")
            row["correlation_status"] = "undefined_numerical_correlation"
    save_metrics_csv(rows, output / "test_metrics.csv")
    save_first_comparison_plot(x_test, y_test, predictions, tuple(description["lead_names"]),
                               description["sampling_rate_hz"], output / "test_prediction_comparison.png")
    print(f"Overall: {rows[0]}\nEvaluation results: {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Explicit best_model.pth created by train_1000.py.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true", help="Validate data/checkpoint only; no test inference or writes.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main():
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
