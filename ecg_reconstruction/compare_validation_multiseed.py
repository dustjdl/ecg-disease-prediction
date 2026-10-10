"""Same-seed validation-only comparison; supports original seed-42 pairs intact."""

import argparse
import csv
from pathlib import Path

import torch

from compare_validation_1000 import (MSE_ATOL, MSE_RTOL, compare_rows, loss_check, validation_metrics)
from multiseed_common import PROTOCOL, load_checkpoint, source_manifest, validate_pair
from train import make_loader
from train_1000 import DATASET, PROJECT, environment, load_and_select, new_id, write_json


def prepare(args):
    arrays, masks, description = load_and_select(args.dataset)
    baseline, bc, br = load_checkpoint(args.baseline_checkpoint, "baseline", description, allow_legacy=True)
    candidate, cc, cr = load_checkpoint(args.candidate_checkpoint, "candidate", description,
                                       expected_seed=bc["seed"], allow_legacy=True)
    validate_pair(bc, cc, br)
    return arrays, masks, description, baseline, bc, br, candidate, cc, cr


def compare(args):
    arrays, masks, description, baseline, bc, br, candidate, cc, cr = prepare(args)
    if args.validate_only:
        print("Same-seed pair metadata PASS; no validation/test inference or writes.")
        return
    settings = bc["experiment"]["settings"]
    loader = make_loader(arrays["X"][masks["validation"]], arrays["Y"][masks["validation"]],
                         settings["batch_size"], shuffle=False)
    del arrays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    execution = environment(device)
    output = PROJECT / "results/validation_comparison_multiseed" / new_id("comparison")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison_config.json", {
        "seed": bc["seed"], "settings": settings, "split": "validation",
        "validation_ecg_ids": description["split_ecg_ids"]["validation"],
        "dataset_sha256": description["dataset_sha256"], "baseline": br, "candidate": cr,
        "baseline_epoch": bc["epoch"], "candidate_epoch": cc["epoch"], "environment": execution,
        "source_sha256": source_manifest(),
        "aggregation": "float64 errors and sums; pooled overall/per-lead; Lead I excluded",
        "initialization_and_orders_verified": bc["experiment"].get("training_protocol") == PROTOCOL,
        "legacy_limitation": "Original seed-42 runs did not store actual initial weight hashes or all epoch orders.",
        "mse_tolerance": {"atol": MSE_ATOL, "rtol": MSE_RTOL}})
    left, left_historical = validation_metrics(baseline, loader, device)
    right, right_historical = validation_metrics(candidate, loader, device)
    checks = {"baseline": loss_check(left, left_historical, bc, execution),
              "candidate": loss_check(right, right_historical, cc, execution)}
    write_json(output / "validation_loss_checks.json", checks)
    rows = compare_rows(left, right)
    for row in rows:
        row.update(seed=bc["seed"], baseline_loss_matches=checks["baseline"]["matches"],
                   candidate_loss_matches=checks["candidate"]["matches"],
                   comparison_validated=all(check["matches"] for check in checks.values()))
    with (output / "validation_metrics_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Seed {bc['seed']} overall: {rows[0]}\nLoss checks: {checks}\nResults: {output}")
    if not all(check["matches"] for check in checks.values()):
        raise RuntimeError("Saved validation loss mismatch; diagnostics saved, comparison is unverified.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main():
    compare(parse_args())


if __name__ == "__main__":
    main()
