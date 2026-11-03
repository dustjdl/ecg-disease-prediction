"""Frozen A/B/C classification comparison. Train all models before any test evaluation.

See docs/ECG_CLASSIFICATION_EXPERIMENT.md. No source inputs are overwritten.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import analyze_disease_reconstruction as audit_tools
from model import ECGModel
from model_dilated import ECGDilatedModel

PROJECT = Path(__file__).resolve().parent
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
INPUTS = {"A": "actual Lead I", "B": "actual 12 leads", "C": "actual Lead I + reconstructed 11 leads"}
DEFAULT_OUT = PROJECT / "results/classification_abc/experiment_20261101"
PROTOCOL = {
    "seeds": [42, 43, 44], "epochs": 40, "batch_size": 32,
    "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
    "dropout": .2, "loss": "BCEWithLogitsLoss with train-only neg/pos weights",
    "checkpoint_selection": "first strictly lowest weighted validation BCE in 40 fixed epochs",
    "threshold_selection": "per output maximize validation F1 on 0.05..0.95 step .01; ties closest to .5, then higher",
    "normalization": "per-input/per-channel train-eligible population mean and SD over records/time; no record centering/filtering",
    "unclassified": "exclude from supervised loss and metrics; preserve split and save predictions with NA targets",
    "multilabel_mixed": "retain all provided labels including NORM+disease",
    "binary": "any disease MI/STTC/CD/HYP positive, NORM-only negative; mixed positive; unclassified excluded",
    "test_policy": "finish all 18 models and lock all thresholds/checkpoints before any test probability/metric calculation",
    "architecture": "Conv1d(C,32,7,stride=2,pad=3)-ReLU-Conv1d(32,64,5,2,2)-ReLU-Conv1d(64,128,5,2,2)-ReLU; global mean+max; Linear(256,64)-ReLU-Dropout(.2)-Linear(64,outputs)",
    "AP_definition": "AUPRC = non-interpolated average precision (sklearn.average_precision_score), not trapezoidal PR area",
    "seed_interpretation": "same records/splits/reconstruction NPZ repeated; SD ddof=1, not independent cohorts",
}


def write_json(path, data):
    # JSON null rather than nonstandard NaN.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv(path, frame):
    pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def stamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


class Classifier(nn.Module):
    def __init__(self, channels, outputs, seed):
        super().__init__()
        # Separate RNG scopes keep deeper-layer initialization identical across A/B/C.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100)
            self.first = nn.Conv1d(channels, 32, 7, stride=2, padding=3)
            torch.manual_seed(seed + 200)
            self.body = nn.Sequential(nn.ReLU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
                                      nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, outputs))

    def forward(self, x):
        z = self.body(self.first(x))
        return self.head(torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1))


def targets(frame, task):
    y = frame[LABELS].to_numpy(dtype=np.float32)
    eligible = y.sum(axis=1) > 0
    if task == "binary":
        y = (y[:, 1:].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    return y, eligible


def binary_metrics(y, probability, predicted):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    tp, fp = int(((y == 1) & (predicted == 1)).sum()), int(((y == 0) & (predicted == 1)).sum())
    tn, fn = int(((y == 0) & (predicted == 0)).sum()), int(((y == 1) & (predicted == 0)).sum())
    result = {"n": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    def ratio(name, numerator, denominator, reason):
        result[name] = numerator / denominator if denominator else np.nan
        result[name + "_na_reason"] = "" if denominator else reason
    ratio("precision", tp, tp + fp, "no predicted positives")
    ratio("recall", tp, tp + fn, "no actual positives")
    ratio("sensitivity", tp, tp + fn, "no actual positives")
    ratio("specificity", tn, tn + fp, "no actual negatives")
    ratio("f1", 2 * tp, 2 * tp + fp + fn, "no actual or predicted positives")
    ratio("accuracy", tp + tn, len(y), "empty cohort")
    result["auroc"] = roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan
    result["auroc_na_reason"] = "" if len(np.unique(y)) == 2 else "requires both positive and negative labels"
    result["auprc"] = average_precision_score(y, probability) if y.sum() else np.nan
    result["auprc_na_reason"] = "" if y.sum() else "no actual positives"
    return result


def choose_thresholds(y, probability):
    thresholds, details = [], []
    grid = np.round(np.arange(.05, .951, .01), 2)
    for k in range(y.shape[1]):
        if len(np.unique(y[:, k])) < 2:
            thresholds.append(.5)
            details.append({"threshold": .5, "reason": "validation lacks both classes; fixed .5 fallback", "validation_f1": None})
            continue
        scores = []
        for t in grid:
            pred = probability[:, k] >= t
            tp = np.sum(pred & (y[:, k] == 1))
            fp, fn = np.sum(pred & (y[:, k] == 0)), np.sum(~pred & (y[:, k] == 1))
            scores.append(2 * tp / (2 * tp + fp + fn))
        best = max(range(len(grid)), key=lambda i: (scores[i], -round(abs(grid[i] - .5), 8), grid[i]))
        thresholds.append(float(grid[best]))
        details.append({"threshold": float(grid[best]), "reason": "maximum validation F1", "validation_f1": float(scores[best])})
    return np.asarray(thresholds), details


def normalization(values, mask):
    train = values[mask].astype(np.float64)
    mean, sd = train.mean(axis=(0, 2)), train.std(axis=(0, 2), ddof=0)
    return mean, np.where(sd > 1e-8, sd, 1.), sd <= 1e-8


def prepare(out):
    folder = out / "input_audit"
    folder.mkdir()
    args = audit_tools.parse_args([])
    audit = {}
    frame, actual, reconstructed, leads, rate = audit_tools.load_inputs(args, folder, audit)
    write_json(folder / "audit.json", audit)
    overlaps = []
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        common = sorted(set(frame[frame.split == a].patient_id) & set(frame[frame.split == b].patient_id))
        overlaps.append({"split_a": a, "split_b": b, "n_shared_patients": len(common), "patient_ids": common})
    write_json(out / "patient_overlap.json", overlaps)
    csv(out / "patient_overlap.csv", [{**x, "patient_ids": "|".join(map(str, x["patient_ids"]))} for x in overlaps])
    # Reuse source-label cross checks; ensure canonical numeric target columns.
    for label in LABELS:
        frame[label] = pd.to_numeric(frame[label]).astype(int)
    eligible = frame[LABELS].sum(axis=1) > 0
    frame["classification_eligible"] = eligible
    frame["exclusion_reason_classification"] = np.where(eligible, "", "no diagnostic superclass; not assumed normal/negative")
    csv(out / "cohort.csv", frame)
    support = []
    for split in ("train", "validation", "test"):
        mask = (frame.split == split) & eligible
        for label in LABELS:
            support.append({"task": "multilabel", "split": split, "label": label, "n_original": int((frame.split == split).sum()),
                            "n_eligible": int(mask.sum()), "positive": int(frame.loc[mask, label].sum()), "negative": int(mask.sum() - frame.loc[mask, label].sum())})
        disease = (frame[LABELS[1:]].sum(axis=1) > 0)
        support.append({"task": "binary", "split": split, "label": "ABNORMAL", "n_original": int((frame.split == split).sum()),
                        "n_eligible": int(mask.sum()), "positive": int(disease[mask].sum()), "negative": int((~disease[mask]).sum())})
    csv(out / "label_support.csv", support)
    i = leads.index("I")
    # C explicitly combines actual I with the supplied 11 reconstructed channels.
    c = reconstructed.copy()
    c[:, :, i] = actual[:, :, i]
    values = {"A": actual[:, :, [i]].transpose(0, 2, 1).copy(),
              "B": actual.transpose(0, 2, 1).copy(), "C": c.transpose(0, 2, 1).copy()}
    for v in values.values():
        audit_tools.require(np.isfinite(v).all(), "Classification inputs must be finite")
    return frame, values, audit


def provenance(out, frame, values):
    rows = []
    dataset_hash = audit_tools.sha256(PROJECT / "data/processed/ptbxl_100hz_1000records.npz")
    # Numerical consistency is evidence, not an undocumented NPZ-generation log.
    for path in sorted((PROJECT / "checkpoints").glob("*/*/best_model.pth")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        e = checkpoint.get("experiment", {})
        name = e.get("model_structure", {}).get("class")
        if name not in ("model.ECGModel", "model_dilated.ECGDilatedModel"):
            continue
        model = ECGModel() if name == "model.ECGModel" else ECGDilatedModel()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        diffs = []
        with torch.inference_mode():
            for start in range(0, len(frame), 32):
                prediction = model(torch.from_numpy(values["A"][start:start+32])).numpy()
                diffs.append(np.abs(prediction.astype(np.float64) - values["C"][start:start+32, 1:, :].astype(np.float64)))
        difference = np.concatenate(diffs)
        ids = e.get("split_ecg_ids", {})
        train_ids = set(ids.get("train", []))
        val_test = set(frame[frame.split != "train"].ecg_id)
        metadata_matches = all(set(ids.get(s, [])) == set(frame[frame.split == s].ecg_id) for s in ("train", "validation", "test"))
        row = {"checkpoint": str(path.relative_to(PROJECT)), "checkpoint_sha256": audit_tools.sha256(path), "model_class": name,
               "epoch": int(checkpoint["epoch"]), "seed": int(checkpoint["seed"]), "max_abs_prediction_difference_mV": float(difference.max()),
               "mean_abs_prediction_difference_mV": float(difference.mean()), "consistent_at_2e_5_mV": bool(difference.max() <= 2e-5),
               "train_id_count": len(train_ids), "train_overlap_with_current_validation_test": len(train_ids & val_test),
               "split_ids_match_supplied": metadata_matches, "reference_dataset_hash_matches": e.get("dataset_sha256") == dataset_hash,
               "saved_test_usage": e.get("test_usage", "not recorded")}
        rows.append(row)
        print("Reconstruction provenance:", path.parent.name, "max error", row["max_abs_prediction_difference_mV"], flush=True)
    csv(out / "reconstruction_checkpoint_consistency.csv", rows)
    matches = [r for r in rows if r["consistent_at_2e_5_mV"]]
    write_json(out / "reconstruction_provenance.json", {
        "npz_generator_script_or_embedded_checkpoint_metadata": "unverified: no generator script/log or checkpoint ID found in supplied NPZ/repository",
        "numerically_consistent_checkpoints": matches,
        "evidence_interpretation": "Numerical consistency with a checkpoint and saved training/source metadata; not proof of the exact generation command or full training history",
        "training_source": "Saved matching checkpoint metadata identifies the original PTB-XL 1000 NPZ and train-only 798 IDs; source training loop consumes train loader for optimizer steps and validation for selection",
        "validation_test_training_use": "For numerically matching checkpoint, stored train IDs are disjoint from validation/test; saved protocol says no validation/test optimizer steps. Full historical execution/generator logs remain unverified.",
        "limitations": ["Classifier C train inputs are in-sample reconstructions from a reconstructor trained on the same train ECGs; no out-of-fold reconstructions supplied",
                        "Reconstruction checkpoint was selected with validation; upstream validation involvement is shared across classification seeds",
                        "C is a deterministic representation of Lead I, not 11 additional independent measured leads",
                        "Prior test reconstruction metrics and labels have been inspected; not a wholly unseen external test workflow"]})


def probabilities(model, x, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            result.append(model(torch.from_numpy(x[start:start+128]).to(device)).sigmoid().cpu().numpy())
    return np.concatenate(result)


def train_one(out, task, inp, seed, frame, values, device):
    folder = out / "models" / task / inp / f"seed_{seed}"
    folder.mkdir(parents=True)
    y, eligible = targets(frame, task)
    train = (frame.split.to_numpy() == "train") & eligible
    validation = (frame.split.to_numpy() == "validation") & eligible
    mean, sd, constant = normalization(values, train)
    # No test normalization/forward pass until every model and threshold is locked.
    xtrain = np.ascontiguousarray((values[train] - mean[None, :, None]) / sd[None, :, None], dtype=np.float32)
    xval = np.ascontiguousarray((values[validation] - mean[None, :, None]) / sd[None, :, None], dtype=np.float32)
    positive, negative = y[train].sum(axis=0), len(y[train]) - y[train].sum(axis=0)
    weights = np.divide(negative, positive, out=np.ones_like(positive), where=positive > 0)
    weights = np.where(negative > 0, weights, 1.)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Classifier(values.shape[1], y.shape[1], seed).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(y[train]), torch.tensor(frame.ecg_id.to_numpy()[train])),
                        batch_size=PROTOCOL["batch_size"], shuffle=True, generator=generator, num_workers=0)
    config = {"task": task, "input": inp, "input_description": INPUTS[inp], "seed": seed, "labels": LABELS if task == "multilabel" else ["ABNORMAL"],
              "protocol": PROTOCOL, "parameter_count": sum(p.numel() for p in model.parameters()), "structure": repr(model),
              "n_train": int(train.sum()), "n_validation": int(validation.sum()), "train_ids": frame.ecg_id.to_numpy()[train].tolist(),
              "validation_ids": frame.ecg_id.to_numpy()[validation].tolist(), "normalization_mean": mean.tolist(), "normalization_sd": sd.tolist(),
              "constant_train_channels": constant.tolist(), "train_positive": positive.tolist(), "train_negative": negative.tolist(), "pos_weight": weights.tolist()}
    write_json(folder / "config.json", config)
    np.savez(folder / "normalization.npz", mean=mean, sd=sd, pos_weight=weights)
    history, order_rows, best, best_epoch = [], [], float("inf"), None
    start_time = time.monotonic()
    for epoch in range(1, PROTOCOL["epochs"] + 1):
        model.train(); total, count, ids = 0., 0, []
        for bx, by, ecg_id in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            total += loss.item() * len(bx); count += len(bx); ids.extend(ecg_id.tolist())
        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(torch.from_numpy(xval).to(device)), torch.from_numpy(y[validation]).to(device)).item()
        audit_tools.require(np.isfinite(val_loss) and np.isfinite(total), "Nonfinite classification loss")
        history.append({"epoch": epoch, "train_loss": total / count, "validation_loss": val_loss})
        order_rows.append({"epoch": epoch, "train_ecg_id_order_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()})
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "validation_loss": best, "config": config}, folder / "best_model.pth")
        if epoch % 10 == 0:
            print(f"{task} {inp} seed={seed} epoch={epoch}/40 train={total/count:.4f} val={val_loss:.4f}", flush=True)
    csv(folder / "history.csv", history); csv(folder / "train_order_digests.csv", order_rows)
    saved = torch.load(folder / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    pval = probabilities(model, xval, device)
    thresholds, details = choose_thresholds(y[validation], pval)
    for d, label in zip(details, config["labels"]):
        d["label"] = label
    write_json(folder / "thresholds.json", details)
    csv(folder / "validation_probabilities.csv", [{"ecg_id": int(e), "label": label, "target": int(yval[k]), "probability": float(prob[k])}
         for e, yval, prob in zip(frame.ecg_id.to_numpy()[validation], y[validation], pval) for k, label in enumerate(config["labels"])])
    selection = {"task": task, "input": inp, "seed": seed, "best_epoch": best_epoch, "best_validation_loss": best,
                 "thresholds": thresholds.tolist(), "checkpoint_sha256": audit_tools.sha256(folder / "best_model.pth"),
                 "thresholds_sha256": audit_tools.sha256(folder / "thresholds.json"), "training_seconds": time.monotonic() - start_time,
                 "path": str(folder.relative_to(out)), "selected_at_kst": stamp()}
    write_json(folder / "selection.json", selection)
    print(f"FINISHED {task} {inp} seed={seed} best_epoch={best_epoch} seconds={selection['training_seconds']:.1f}", flush=True)
    return selection


def evaluate(out, frame, values, selections, device):
    rows, predictions = [], []
    for selection in selections:
        task, inp, seed = (selection[k] for k in ("task", "input", "seed"))
        folder = out / selection["path"]
        audit_tools.require(audit_tools.sha256(folder / "best_model.pth") == selection["checkpoint_sha256"], "Checkpoint changed after selection freeze")
        audit_tools.require(audit_tools.sha256(folder / "thresholds.json") == selection["thresholds_sha256"], "Thresholds changed after freeze")
        ckpt = torch.load(folder / "best_model.pth", map_location=device, weights_only=True)
        config = ckpt["config"]
        model = Classifier(values[inp].shape[1], len(config["labels"]), seed).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        mean, sd = np.asarray(config["normalization_mean"]), np.asarray(config["normalization_sd"])
        x = np.ascontiguousarray((values[inp] - mean[None, :, None]) / sd[None, :, None], dtype=np.float32)
        probability = probabilities(model, x, device)
        thresholds = np.asarray(selection["thresholds"])
        predicted = (probability >= thresholds).astype(int)
        y, eligible = targets(frame, task)
        np.savez(folder / "predictions.npz", ecg_id=frame.ecg_id.to_numpy(), split=frame.split.to_numpy(dtype=str),
                 labels=np.array(config["labels"]), target=np.where(eligible[:, None], y, np.nan), probability=probability,
                 predicted=predicted, eligible=eligible, thresholds=thresholds, seed=seed)
        for i, record in frame.iterrows():
            for k, label in enumerate(config["labels"]):
                predictions.append({"task": task, "input": inp, "seed": seed, "split": record.split, "ecg_id": int(record.ecg_id),
                                    "patient_id": int(record.patient_id), "label": label, "eligible": bool(eligible[i]),
                                    "target": int(y[i, k]) if eligible[i] else np.nan, "probability": float(probability[i, k]),
                                    "predicted_label": int(predicted[i, k]), "threshold": float(thresholds[k])})
        for split in ("train", "validation", "test"):
            mask = (frame.split.to_numpy() == split) & eligible
            base = {"task": task, "input": inp, "seed": seed, "split": split, "n_records": int(mask.sum())}
            per_class = []
            for k, label in enumerate(config["labels"]):
                m = binary_metrics(y[mask, k], probability[mask, k], predicted[mask, k])
                per_class.append(m); rows.append({**base, "label": label, "threshold": float(thresholds[k]), **m})
            if task == "multilabel":
                rows.append({**base, "label": "micro", "threshold": np.nan, **binary_metrics(y[mask].ravel(), probability[mask].ravel(), predicted[mask].ravel())})
                macro = {**base, "label": "macro", "n": int(mask.sum()), "positive": np.nan, "negative": np.nan, "threshold": np.nan}
                for key in ("precision", "recall", "sensitivity", "specificity", "f1", "accuracy", "auroc", "auprc"):
                    v = np.array([m[key] for m in per_class])
                    macro[key] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                    macro[key + "_valid_classes"] = int(np.isfinite(v).sum())
                    macro[key + "_na_reason"] = "; ".join(f"{label}: {m[key+'_na_reason']}" for label, m in zip(config["labels"], per_class) if m[key+"_na_reason"])
                macro["subset_accuracy"] = float((y[mask] == predicted[mask]).all(axis=1).mean())
                rows.append(macro)
        print("EVALUATED", task, inp, seed, flush=True)
    metrics = pd.DataFrame(rows)
    csv(out / "metrics_by_seed.csv", metrics)
    csv(out / "predictions_long.csv", predictions)
    return metrics


def summarize(out, metrics):
    numeric = ["auroc", "auprc", "precision", "recall", "f1", "sensitivity", "specificity", "accuracy"]
    summaries = []
    for keys, f in metrics.groupby(["task", "input", "split", "label"], sort=False):
        row = dict(zip(["task", "input", "split", "label"], keys))
        row.update(n_records=int(f.n_records.iloc[0]), positive=f.positive.iloc[0], negative=f.negative.iloc[0])
        for key in numeric:
            v = f[key].dropna()
            row.update({key + "_mean": v.mean(), key + "_sd": v.std(ddof=1), key + "_n_seeds": len(v)})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    csv(out / "abc_comparison_mean_sd.csv", summary)
    delta = []
    for task, label in (("multilabel", "macro"), ("multilabel", "micro"), ("binary", "ABNORMAL")):
        f = metrics[(metrics.task == task) & (metrics.label == label) & (metrics.split == "test")]
        for a, b in (("C", "A"), ("C", "B"), ("B", "A")):
            for metric in numeric:
                left, right = f[f.input == a].set_index("seed")[metric], f[f.input == b].set_index("seed")[metric]
                values = left - right
                delta.append({"task": task, "label": label, "comparison": a + "-" + b, "metric": metric,
                              "paired_mean_difference": values.mean(), "paired_sd_difference": values.std(ddof=1),
                              **{f"seed_{int(s)}_difference": v for s, v in values.items()}})
    csv(out / "test_paired_differences.csv", delta)
    confusion = metrics[(metrics.task == "binary") & (metrics.split.isin(["validation", "test"]))]
    csv(out / "binary_confusion_matrices.csv", confusion[["input", "seed", "split", "n_records", "positive", "negative", "tn", "fp", "fn", "tp", "threshold"]])
    issues = []
    for r in metrics.to_dict("records"):
        for key in numeric:
            if pd.isna(r[key]) or r.get(key + "_na_reason"):
                issues.append({k: r[k] for k in ("task", "input", "seed", "split", "label")} | {"metric": key, "reason": r.get(key + "_na_reason", "not defined")})
    csv(out / "metric_na_reasons.csv", pd.DataFrame(issues, columns=["task", "input", "seed", "split", "label", "metric", "reason"]))
    for task, label in (("multilabel", "macro"), ("binary", "ABNORMAL")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        for ax, metric in zip(axes, ("auroc", "auprc", "f1")):
            f = summary[(summary.task == task) & (summary.label == label) & (summary.split == "test")].set_index("input").loc[list(INPUTS)]
            ax.bar(range(3), f[metric + "_mean"], yerr=f[metric + "_sd"], capsize=5, color=["#2563eb", "#64748b", "#d97706"])
            for i, (m, sd) in enumerate(zip(f[metric + "_mean"], f[metric + "_sd"])):
                ax.text(i, m + sd + .025, f"{m:.3f}", ha="center")
            ax.set_xticks(range(3), ["A: Lead I", "B: actual 12", "C: reconstructed"])
            ax.set_ylim(0, 1.08); ax.set_title(metric.upper()); ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Test {task} {label}: mean +/- sample SD, seeds 42/43/44 (same test cohort)")
        fig.savefig(out / f"test_{task}_comparison.png", dpi=180); plt.close(fig)
    f = summary[(summary.task == "multilabel") & (summary.split == "test") & summary.label.isin(LABELS)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("auroc", "auprc", "f1")):
        for offset, inp in enumerate(INPUTS):
            d = f[f.input == inp].set_index("label").loc[LABELS]
            ax.bar(np.arange(5) + (offset - 1) * .24, d[metric + "_mean"], width=.24, label=inp, yerr=d[metric + "_sd"], capsize=2)
        ax.set_xticks(range(5), LABELS); ax.set_ylim(0, 1.08); ax.set_title(metric.upper()); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "test_multilabel_by_class.png", dpi=180); plt.close(fig)
    histories = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for i, task in enumerate(("multilabel", "binary")):
        for j, inp in enumerate(INPUTS):
            for seed in PROTOCOL["seeds"]:
                h = pd.read_csv(out / "models" / task / inp / f"seed_{seed}/history.csv")
                axes[i, j].plot(h.epoch, h.validation_loss, label=str(seed))
            axes[i, j].set_title(task + " " + inp); axes[i, j].set_xlabel("epoch"); axes[i, j].set_ylabel("weighted validation BCE"); axes[i, j].legend()
    fig.savefig(out / "validation_learning_curves.png", dpi=160); plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--phase", choices=("all", "audit"), default="all")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"started_at_kst": stamp(), "protocol": PROTOCOL, "code_sha256": audit_tools.sha256(Path(__file__)),
              "audit_code_sha256": audit_tools.sha256(PROJECT / "analyze_disease_reconstruction.py"),
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "policy_review": {"status": "No finalized role-2 policy found in local repository", "evidence": ["ecg-disease-prediction/docs,src,tests contain only .gitkeep", "ECG_INTERFACE.md section '분류 담당자와 앞으로 합의할 사항 — 미확정'", "repository README says classification integration incomplete"],
                                "resolution": "User-authorized default five-label task plus auxiliary abnormal-any binary task"}}
    write_json(out / "experiment_config.json", config)
    frame, values, audit = prepare(out)
    provenance(out, frame, values)
    if args.phase == "audit":
        print("AUDIT COMPLETE", out); return
    selections = []
    for task in ("multilabel", "binary"):
        for seed in PROTOCOL["seeds"]:
            for inp in INPUTS:
                selections.append(train_one(out, task, inp, seed, frame, values[inp], device))
    # Selection file is written before evaluate() can see any test predictions.
    lock = {"frozen_at_kst": stamp(), "config_sha256": audit_tools.sha256(out / "experiment_config.json"), "models": selections}
    write_json(out / "selection_frozen_before_test.json", lock)
    csv(out / "selected_models.csv", selections)
    metrics = evaluate(out, frame, values, selections, device)
    summarize(out, metrics)
    write_json(out / "completion.json", {"status": "complete", "finished_at_kst": stamp(), "models_trained": len(selections),
                                          "selection_lock_sha256": audit_tools.sha256(out / "selection_frozen_before_test.json"),
                                          "test_evaluated_only_after_all_selections_locked": True})
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
