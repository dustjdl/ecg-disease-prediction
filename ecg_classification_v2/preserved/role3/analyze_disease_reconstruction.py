"""Audit supplied reconstructions and describe errors by diagnostic superclass.

No torch import, training, classification, inference, or split generation.
Run --help; see docs/DISEASE_RECONSTRUCTION_ANALYSIS.md for input and methods.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / "results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json"
GROUPS = ("normal_class_only", "MI", "STTC", "CD", "HYP", "normal_disease_mixed", "unclassified")
CLASSES = {"NORM", "MI", "STTC", "CD", "HYP"}
METRICS = ("mae", "rmse", "pearson_r")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def strings(values):
    a = np.asarray(values)
    if a.dtype.kind == "O":
        require(all(isinstance(x, (str, bytes)) for x in a.flat), "Text metadata contains missing or non-text values")
        a = np.asarray([x.decode("utf-8") if isinstance(x, bytes) else x for x in a])
    require(a.dtype.kind in "US", "Text metadata must be Unicode or bytes, not object/pickle")
    return np.char.decode(a, "utf-8") if a.dtype.kind == "S" else a.astype(str)


def ids(values, name):
    a = np.asarray(values)
    require(a.ndim == 1 and len(a) > 0, f"{name}: expected nonempty one-dimensional IDs")
    parsed = []
    for value in a:
        token = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        require(re.fullmatch(r"[0-9]+", token) is not None, f"{name}: invalid ID {token!r}")
        parsed.append(int(token))
    return np.asarray(parsed, dtype=np.int64)


def splits(values):
    a = strings(values)
    # Spelling alias only; ID membership is always checked against frozen config.
    result = np.array([{"val": "validation", "valid": "validation"}.get(x.strip().lower(), x.strip().lower()) for x in a])
    require(set(result) <= {"train", "validation", "test"}, "Unknown split label")
    return result


def parse_classes(value):
    """Parse explicit superclass labels, never infer labels from inclusion flags."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "nan", "none", "null", "[]", "{}", "set()"):
            return set()
        if value[:1] in "[{(":
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Malformed class list: {value!r}") from exc
        else:
            value = re.split(r"[;,|\s]+", value)
    if isinstance(value, dict):
        value = [k for k, v in value.items() if float(v) > 0]
    require(isinstance(value, (list, tuple, set)), f"Expected class labels, got {value!r}")
    labels = {str(x).strip().upper() for x in value}
    require(labels <= CLASSES, f"Unknown superclass labels: {sorted(labels - CLASSES)}; use superclass labels, not SCP codes")
    return labels


def boolean(value):
    token = str(value).strip().lower()
    require(token in {"true", "false", "1", "0", "1.0", "0.0"}, f"Invalid boolean: {value!r}")
    return token in {"true", "1", "1.0"}


def group_memberships(labels):
    diseases = sorted(labels - {"NORM"})
    if labels == {"NORM"}:
        return ["normal_class_only"]
    if "NORM" in labels and diseases:
        return ["normal_disease_mixed"]
    return diseases or ["unclassified"]


def assign_classes(frame, column=None):
    candidates = [c for c in ("diagnostic_superclass", "diagnostic_superclasses", "diagnostic_classes", "superclasses", "scp_classes", "classes") if c in frame]
    if column:
        require(column in frame, f"Class column not found: {column}")
        chosen = column
    else:
        require(len(candidates) <= 1, f"Ambiguous class columns {candidates}; set --classes-column")
        chosen = candidates[0] if candidates else None
    if chosen:
        labels = [parse_classes(v) for v in frame[chosen]]
    else:
        require(CLASSES <= set(frame), "Need explicit superclass list (--classes-column) or NORM/MI/STTC/CD/HYP binary columns")
        labels = [{c for c in CLASSES if boolean(row[c])} for _, row in frame.iterrows()]
    if chosen and CLASSES <= set(frame):
        binary = [{c for c in CLASSES if boolean(row[c])} for _, row in frame.iterrows()]
        require(labels == binary, "Class list disagrees with binary superclass columns")
    if "normal_class_only" in frame:
        require([boolean(v) for v in frame.normal_class_only] == [x == {"NORM"} for x in labels],
                "normal_class_only disagrees with explicit superclass labels")
    result = frame.copy()
    result["analysis_classes"] = ["|".join(sorted(x)) for x in labels]
    result["analysis_groups"] = ["|".join(group_memberships(x)) for x in labels]
    return result, chosen or "binary NORM/MI/STTC/CD/HYP"


def metric_arrays(actual, reconstructed):
    """Float64 [record,time,lead] metrics; invalid r stays NaN with a reason."""
    actual = np.asarray(actual, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    require(actual.shape == reconstructed.shape and actual.ndim == 3 and actual.shape[1] >= 2, "Waveform shape mismatch")
    finite_a = np.isfinite(actual).all(axis=1)
    finite_b = np.isfinite(reconstructed).all(axis=1)
    finite = finite_a & finite_b
    constant_a = np.max(actual, axis=1) == np.min(actual, axis=1)
    constant_b = np.max(reconstructed, axis=1) == np.min(reconstructed, axis=1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        delta = reconstructed - actual
        mae = np.mean(np.abs(delta), axis=1)
        mse = np.mean(delta ** 2, axis=1)
        rmse = np.sqrt(mse)
        mean_a, mean_b = actual.mean(axis=1), reconstructed.mean(axis=1)
        ca, cb = actual - mean_a[:, None, :], reconstructed - mean_b[:, None, :]
        # Scale before norms/dot products to avoid unnecessary under/overflow.
        sa, sb = np.max(np.abs(ca), axis=1), np.max(np.abs(cb), axis=1)
        ua, ub = ca / sa[:, None, :], cb / sb[:, None, :]
        na, nb = np.sqrt(np.sum(ua ** 2, axis=1)), np.sqrt(np.sum(ub ** 2, axis=1))
        r = np.sum((ua / na[:, None, :]) * (ub / nb[:, None, :]), axis=1)
        near_a = (sa > 0) & (sa < 1e-13 * np.abs(mean_a) / na)
        near_b = (sb > 0) & (sb < 1e-13 * np.abs(mean_b) / nb)
    status = np.full(finite.shape, "valid", dtype=object)
    status[finite & (near_a | near_b)] = "near_constant"
    status[finite & constant_a] = "constant_actual"
    status[finite & constant_b] = "constant_reconstructed"
    status[finite & constant_a & constant_b] = "constant_both"
    status[~finite_a] = "nonfinite_actual"
    status[finite_a & ~finite_b] = "nonfinite_reconstructed"
    status[finite & (status == "valid") & ~np.isfinite(r)] = "numerical_failure"
    mae[~finite], mse[~finite], rmse[~finite] = np.nan, np.nan, np.nan
    require(np.isfinite(mae[finite]).all() and np.isfinite(mse[finite]).all(), "Metric overflow")
    r = np.clip(r, -1, 1)
    r[status != "valid"] = np.nan
    return {"mae": mae, "mse": mse, "rmse": rmse, "pearson_r": r, "correlation_status": status}


def ntc(array, name, count, layout):
    require(array.dtype.kind in "fiu" and array.ndim == 3 and array.shape[0] == count, f"{name}: expected real [N,T,12] or [N,12,T]")
    if layout == "auto":
        axes = [axis for axis in (1, 2) if array.shape[axis] == 12]
        require(len(axes) == 1, f"{name}: ambiguous lead axis; set --waveform-layout")
        axis = axes[0]
    else:
        axis = 2 if layout == "NTC" else 1
    require(array.shape[axis] == 12, f"{name}: lead axis must have length 12")
    result = array if axis == 2 else array.transpose(0, 2, 1)
    require(result.shape[1] >= 2, "At least two samples required")
    return result


def load_inputs(args, out, audit):
    paths = {"labels": args.labels, "reconstructions": args.reconstructions,
             "reference_dataset": args.reference_dataset, "split_config": args.split_config}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    audit["missing_files"] = missing
    require(not missing, "Missing input files: " + ", ".join(missing))
    audit["inputs"] = {key: {"path": str(p.resolve()), "sha256": sha256(p)} for key, p in paths.items()}
    config = json.loads(args.split_config.read_text(encoding="utf-8-sig"))
    require(audit["inputs"]["reference_dataset"]["sha256"] == config["dataset_sha256"], "Reference dataset hash differs from frozen experiment")
    expected = {int(e): s for s, values in config["split_ecg_ids"].items() for e in values}
    require(len(expected) == sum(map(len, config["split_ecg_ids"].values())), "Duplicate IDs in split config")
    csv = pd.read_csv(args.labels, dtype={"ecg_id": "string"}, encoding="utf-8-sig")
    audit["csv_columns"] = csv.columns.tolist()
    require({"ecg_id", "split"} <= set(csv), "CSV needs ecg_id and split")
    csv["ecg_id"] = ids(csv.ecg_id.to_numpy(), "CSV")
    csv["csv_split_raw"] = csv["split"]
    csv["split"] = splits(csv["split"].to_numpy())
    with np.load(args.reconstructions, allow_pickle=False) as z:
        audit["npz_keys"] = z.files
        required = {"ecg_id", "split", "lead_names", "actual_12lead", "reconstructed_12lead"}
        require(required <= set(z.files), f"Missing NPZ keys: {sorted(required - set(z.files))}")
        ecg_ids = ids(z["ecg_id"], "NPZ")
        raw_split = strings(z["split"])
        split = splits(raw_split)
        require(split.shape == ecg_ids.shape, "NPZ ID/split length mismatch")
        leads = strings(z["lead_names"]).tolist()
        require(len(leads) == 12 and len({s.upper() for s in leads}) == 12, "Expected 12 unique lead_names")
        require({s.upper() for s in leads} == {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}, "Unsupported lead names")
        actual = ntc(z["actual_12lead"], "actual_12lead", len(ecg_ids), args.waveform_layout)
        reconstructed = ntc(z["reconstructed_12lead"], "reconstructed_12lead", len(ecg_ids), args.waveform_layout)
        require(actual.shape == reconstructed.shape, "Actual/reconstructed shapes differ")
        rate = float(np.asarray(z["sampling_rate_hz"]).item()) if "sampling_rate_hz" in z else None
        audit["npz_shapes"] = {"actual_12lead": list(actual.shape), "reconstructed_12lead": list(reconstructed.shape)}
        audit["npz_sampling_rate_hz"] = rate
        npz_patient = z["patient_id"] if "patient_id" in z else None
        if "voltage_unit" in z:
            unit = str(strings(np.asarray(z["voltage_unit"]).reshape(1))[0])
            audit["npz_voltage_unit"] = unit
            require(unit == "mV", f"Unsupported voltage unit: {unit}")
        if "lead_i" in z:
            input_i = z["lead_i"]
            lead_i_index = next(i for i, s in enumerate(leads) if s.upper() == "I")
            require(input_i.shape == (len(ecg_ids), actual.shape[1], 1), "lead_i shape mismatch")
            require(np.array_equal(input_i[:, :, 0], actual[:, :, lead_i_index]), "lead_i disagrees with actual Lead I")
            audit["lead_i_matches_actual_I"] = True
    duplicate_rows = []
    for name, values in (("csv", csv.ecg_id), ("npz", ecg_ids)):
        counts = pd.Series(values).value_counts()
        duplicate_rows.extend({"source": name, "ecg_id": int(k), "count": int(v)} for k, v in counts.items() if v > 1)
    save_csv(pd.DataFrame(duplicate_rows, columns=["source", "ecg_id", "count"]), out / "duplicate_ids.csv")
    audit["duplicate_ids"] = duplicate_rows
    coverage = []
    csv_set, npz_set = set(csv.ecg_id), set(ecg_ids)
    for e in sorted(set(expected) | csv_set | npz_set):
        coverage.append({"ecg_id": e, "in_csv": e in csv_set, "in_npz": e in npz_set, "in_frozen_experiment": e in expected, "expected_split": expected.get(e, "")})
    coverage = pd.DataFrame(coverage)
    save_csv(coverage, out / "id_coverage.csv")
    audit["coverage_counts"] = {"csv_unique": len(csv_set), "npz_unique": len(npz_set), "frozen_experiment": len(expected),
                                "missing_csv": sorted(set(expected) - csv_set), "missing_npz": sorted(set(expected) - npz_set),
                                "extra_csv": sorted(csv_set - set(expected)), "extra_npz": sorted(npz_set - set(expected))}
    require(not duplicate_rows, "Duplicate IDs; no automatic deduplication")
    require(csv_set == npz_set == set(expected), "Missing/extra IDs; inspect id_coverage.csv. No automatic inner-join/drop")
    meta = pd.DataFrame({"ecg_id": ecg_ids, "npz_index": np.arange(len(ecg_ids)), "npz_split": split, "npz_split_raw": raw_split})
    joined = meta.merge(csv, on="ecg_id", how="left", validate="one_to_one", sort=False)
    joined["expected_split"] = joined.ecg_id.map(expected)
    bad = joined[(joined.split != joined.npz_split) | (joined.split != joined.expected_split)]
    save_csv(bad, out / "split_mismatches.csv")
    audit["split_mismatch_count"] = len(bad)
    require(bad.empty, "CSV/NPZ/frozen split mismatch")
    with np.load(args.reference_dataset, allow_pickle=False) as ref:
        reference_ids = ids(ref["ecg_id"], "reference")
        require(len(set(reference_ids)) == len(reference_ids), "Duplicate reference IDs")
        index = {int(e): i for i, e in enumerate(reference_ids)}
        order = [index[int(e)] for e in ecg_ids]
        reference_split = splits(ref["split"])[order]
        require(np.array_equal(reference_split, split), "NPZ split disagrees with original reference")
        audit["original_reference_split_counts"] = {str(k): int(v) for k, v in pd.Series(splits(ref["split"])).value_counts().items()}
        audit["original_ids_not_in_supplied_files"] = sorted(set(reference_ids.tolist()) - npz_set)
        ref_leads = strings(ref["lead_names"]).tolist()
        ref_full = np.concatenate((ref["X"][order], ref["Y"][order]), axis=2)
        lead_order = [ref_leads.index(next(x for x in ref_leads if x.upper() == s.upper())) for s in leads]
        ref_full = ref_full[:, :, lead_order]
        require(ref_full.shape == actual.shape, "Actual shape disagrees with reference")
        require(np.isfinite(actual).all(), "Actual waveform contains nonfinite values")
        max_difference = float(np.max(np.abs(actual.astype(np.float64) - ref_full.astype(np.float64))))
        audit["actual_reference_max_abs_difference_mV"] = max_difference
        require(np.allclose(actual, ref_full, rtol=0, atol=1e-6), "Actual waveforms differ from original mV reference (>1e-6)")
        reference_rate = float(ref["sampling_rate_hz"].item())
        require(rate is None or rate == reference_rate, "Sampling rate disagrees with reference")
        for key in ("patient_id", "strat_fold"):
            if key in joined:
                require(np.array_equal(pd.to_numeric(joined[key]).to_numpy(), ref[key][order]), f"CSV {key} disagrees with reference")
        if npz_patient is not None:
            require(np.array_equal(npz_patient, ref["patient_id"][order]), "NPZ patient_id disagrees with original/CSV")
            audit["npz_patient_ids_match_reference"] = True
        audit["sampling_rate_hz"] = reference_rate
        audit["sampling_rate_source"] = "NPZ checked against original" if rate else "original NPZ (missing in supplied NPZ)"
    joined, label_source = assign_classes(joined, args.classes_column)
    audit["class_source"] = label_source
    audit["include_in_analysis_policy"] = "Retained for cross-tabulation only; never used to assign groups or drop records"
    audit["lead_names"] = leads
    audit["target_leads"] = [s for s in leads if s.upper() != "I"]
    audit["split_counts"] = {str(k): int(v) for k, v in joined.split.value_counts().items()}
    audit["nonfinite_reconstructed_values"] = int((~np.isfinite(reconstructed)).sum())
    i = next(i for i, s in enumerate(leads) if s.upper() == "I")
    audit["reconstructed_input_I_equals_actual"] = bool(np.array_equal(actual[:, :, i], reconstructed[:, :, i]))
    save_csv(joined, out / "joined_records.csv")
    if "include_in_analysis" in joined:
        save_csv(joined.groupby(["split", "analysis_groups", "include_in_analysis"], dropna=False).size().reset_index(name="n_records"), out / "include_flag_crosstab.csv")
    for key, path in paths.items():
        require(sha256(path) == audit["inputs"][key]["sha256"], f"Input changed while reading: {path}")
    audit["status"] = "validated"
    return joined, actual, reconstructed, leads, reference_rate


def make_metrics(joined, actual, reconstructed, leads):
    # All delivered ECGs get raw metrics; only held-out splits enter group summaries.
    selected = joined.copy()
    indices = selected.npz_index.to_numpy(dtype=int)
    target = [i for i, s in enumerate(leads) if s.upper() != "I"]
    calculated = metric_arrays(actual[indices][:, :, target], reconstructed[indices][:, :, target])
    rows = []
    for n, (_, record) in enumerate(selected.iterrows()):
        for l, lead in enumerate([leads[i] for i in target]):
            row = {"ecg_id": int(record.ecg_id), "split": record.split, "lead": lead,
                   "analysis_classes": record.analysis_classes, "analysis_groups": record.analysis_groups,
                   "n_samples": actual.shape[1]}
            row.update({k: v[n, l] for k, v in calculated.items()})
            rows.append(row)
    metrics = pd.DataFrame(rows)
    record_rows = []
    for (split, ecg_id), f in metrics.groupby(["split", "ecg_id"], sort=False):
        valid = f.pearson_r.notna()
        finite = f.mae.notna()
        record_rows.append({"ecg_id": ecg_id, "split": split, "analysis_classes": f.analysis_classes.iloc[0], "analysis_groups": f.analysis_groups.iloc[0],
                            "mae": f.mae.mean() if finite.all() else np.nan,
                            "rmse": np.sqrt(f.mse.mean()) if finite.all() else np.nan,
                            "pearson_r": f.pearson_r.mean(), "n_valid_r_leads": int(valid.sum()), "n_error_leads": int(finite.sum())})
    return metrics, pd.DataFrame(record_rows)


def summarize(metrics, records, leads):
    counts, overlap, summaries = [], [], []
    for split in ("validation", "test"):
        subset = records[records.split == split]
        memberships = {g: set(subset[subset.analysis_groups.str.split("|").map(lambda x: g in x).astype(bool)].ecg_id) for g in GROUPS}
        for g in GROUPS:
            selected = memberships[g]
            counts.append({"split": split, "group": g, "n_records": len(selected), "small_group_lt_10": len(selected) < 10,
                           "policy": "disease groups overlap; NORM+disease isolated"})
            for other in GROUPS:
                overlap.append({"split": split, "group_a": g, "group_b": other, "n_shared_records": len(selected & memberships[other])})
            for lead in ["OVERALL"] + leads:
                f = metrics[(metrics.split == split) & metrics.ecg_id.isin(selected)]
                if lead != "OVERALL":
                    f = f[f.lead == lead]
                row = {"split": split, "group": g, "lead": lead, "n_records": len(selected), "n_record_lead_pairs": len(f),
                       "n_valid_error_pairs": int(f.mae.notna().sum()), "n_valid_r_pairs": int(f.pearson_r.notna().sum()),
                       "n_invalid_r_pairs": int(f.pearson_r.isna().sum()), "small_group_lt_10": len(selected) < 10}
                for metric in METRICS:
                    values = f[metric].dropna()
                    row.update({f"{metric}_mean": values.mean(), f"{metric}_sd": values.std(ddof=1), f"{metric}_median": values.median(),
                                f"{metric}_q25": values.quantile(.25), f"{metric}_q75": values.quantile(.75)})
                row["pooled_rmse"] = np.sqrt(f.mse.mean())
                # Overall SD/quantiles below use ECGs as units, not 11 pseudo-independent leads.
                overall_records = records[(records.split == split) & records.ecg_id.isin(selected)]
                if lead == "OVERALL":
                    for metric in METRICS:
                        v = overall_records[metric].dropna()
                        for stat, value in (("mean", v.mean()), ("sd", v.std(ddof=1)), ("median", v.median()), ("q25", v.quantile(.25)), ("q75", v.quantile(.75))):
                            row[f"{metric}_{stat}"] = value
                    row["n_complete_error_records"] = int((overall_records.n_error_leads == 11).sum())
                    row["n_records_with_valid_r"] = int((overall_records.n_valid_r_leads > 0).sum())
                summaries.append(row)
    return pd.DataFrame(counts), pd.DataFrame(overlap), pd.DataFrame(summaries)


def comparison_plots(out, summary, records, leads):
    for split in ("validation", "test"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
        for ax, metric in zip(axes, METRICS):
            values = [records[(records.split == split) & records.analysis_groups.str.split("|").map(lambda x: g in x).astype(bool)][metric].dropna().to_numpy() for g in GROUPS]
            positions = [i + 1 for i, v in enumerate(values) if len(v)]
            if positions:
                ax.boxplot([v for v in values if len(v)], positions=positions, widths=.55, showfliers=True)
            ax.set_xticks(range(1, 8), [f"{g}\n(n={len(v)})" for g, v in zip(GROUPS, values)], rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("mV" if metric != "pearson_r" else "r")
            ax.set_title(metric + " (ECG-level)")
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(f"{split}: reconstruction errors; overlapping disease groups; n = valid ECG metrics")
        fig.savefig(out / f"{split}_group_comparison.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
        for ax, metric in zip(axes, METRICS):
            s = summary[(summary.split == split) & (summary.lead != "OVERALL")]
            matrix = s.pivot(index="group", columns="lead", values=f"{metric}_mean").reindex(index=GROUPS, columns=leads).to_numpy()
            im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis", **({"vmin": -1, "vmax": 1} if metric == "pearson_r" else {}))
            for y in range(len(GROUPS)):
                for x in range(len(leads)):
                    ax.text(x, y, f"{matrix[y,x]:.2f}" if np.isfinite(matrix[y,x]) else "NA", ha="center", va="center", fontsize=7, color="white", bbox={"facecolor": "black", "alpha": .25, "edgecolor": "none", "pad": .3})
            ax.set_xticks(range(len(leads)), leads, rotation=45)
            ax.set_yticks(range(len(GROUPS)), GROUPS, fontsize=8)
            ax.set_title(metric + " mean" + (" (mV)" if metric != "pearson_r" else ""))
            fig.colorbar(im, ax=ax, shrink=.75)
        fig.suptitle(f"{split}: per-lead means; NA = no valid observations")
        fig.savefig(out / f"{split}_lead_heatmaps.png", dpi=160)
        plt.close(fig)


def examples(out, joined, records, actual, reconstructed, leads, rate):
    chosen = []
    # Predefined ID-only policy: avoids choosing attractive or extreme test errors.
    for split in ("validation", "test"):
        for group in GROUPS:
            members = records[(records.split == split) & records.analysis_groups.str.split("|").map(lambda x: group in x).astype(bool)]
            if members.empty:
                continue
            ecg_id = int(members.ecg_id.min())
            record = joined[joined.ecg_id == ecg_id].iloc[0]
            name = f"waveforms/{split}_{group}_ecg_{ecg_id}.png"
            chosen.append({"split": split, "group": group, "ecg_id": ecg_id, "selection": "minimum ecg_id in group, independent of errors", "group_n": len(members), "path": name})
            index = int(record.npz_index)
            fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True, layout="constrained")
            time = np.arange(actual.shape[1]) / rate
            for ax, lead, l in zip(axes.flat, leads, range(12)):
                ax.plot(time, actual[index, :, l], lw=.8, label="Actual")
                ax.plot(time, reconstructed[index, :, l], lw=.8, alpha=.8, label="Reconstructed")
                ax.set_title(lead + (" (observed input; excluded from metrics)" if lead.upper() == "I" else ""), fontsize=9)
                ax.set_ylabel("mV")
                ax.set_xlabel("s")
                ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"{split} | {group} | ecg_id={ecg_id} | labels={record.analysis_classes} | min-ID example")
            fig.savefig(out / name, dpi=140)
            plt.close(fig)
    return pd.DataFrame(chosen, columns=["split", "group", "ecg_id", "selection", "group_n", "path"])


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |\n|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join("NA" if pd.isna(v) else f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return header + "\n" + "\n".join(rows)


def report(out, audit, summary, chosen):
    sections = ["# 질환별 ECG 재구성 성능 분석", "", "입력 파일의 SHA-256, 스키마, ID·split·원본 파형 대조 결과는 [audit.json](audit.json)에 기록했다.",
                "이 결과는 전달된 재구성 배열의 파형 비교이며 질환 분류 실험이 아니다. 재구성 모델/체크포인트의 출처는 두 파일만으로 보증할 수 없다.",
                "", "## 그룹과 집계", "", "정상은 클래스 집합이 정확히 {NORM}인 기록이다. MI/STTC/CD/HYP는 NORM이 없는 질환 기록을 해당 질환마다 중복 포함한다. NORM+질환은 normal_disease_mixed에만, 클래스가 비어 있으면 unclassified에만 배정한다. include_in_analysis는 그룹 결정이나 제외에 사용하지 않는다.",
                "", "입력 I을 제외한 NPZ 순서의 11개 유도에서 float64로 전체 시간 샘플을 비교했다. 유도별 표의 mean/SD/median/IQR은 ECG별 지표의 분포다. OVERALL의 MAE는 ECG 내 11유도 MAE 평균, RMSE는 ECG 내 11유도 MSE 평균의 제곱근, r은 유효 유도 r의 산술평균을 먼저 구한 뒤 ECG 간 집계한다. pooled_rmse는 전체 유효 ECG×유도 쌍의 MSE를 평균한 후 제곱근을 취하며 mean RMSE와 다르다. r을 파형 연결 후 계산하거나 Fisher z로 평균하지 않았다.",
                "", "상수·준상수·비유한 파형의 r은 빈 값과 사유로 표시했다. 오차는 유한한 쌍에서만 계산한다. 전체 ECG 오차는 11개 유도 모두 유효해야 계산한다. r은 유효 유도만 평균하고 분모를 남긴다. SD는 ddof=1이며 n<2이면 NA다. 빈 그룹도 표에 남긴다."]
    for split in ("validation", "test"):
        s = summary[(summary.split == split) & (summary.lead == "OVERALL")]
        sections.extend(["", f"## {split}", "", markdown_table(s[["group", "n_records", "mae_mean", "rmse_mean", "pooled_rmse", "pearson_r_mean", "n_valid_r_pairs"]]), "", f"![그룹 비교]({split}_group_comparison.png)", "", f"![유도별 비교]({split}_lead_heatmaps.png)"])
        normal = s[s.group == "normal_class_only"].iloc[0]
        for g in ("MI", "STTC", "CD", "HYP"):
            r = s[s.group == g].iloc[0]
            if r.n_records and normal.n_records and np.isfinite(r.mae_mean) and np.isfinite(normal.mae_mean):
                sections.extend(["", f"{g} (n={int(r.n_records)}): 정상군(n={int(normal.n_records)}) 대비 관찰된 평균 MAE 차이는 {r.mae_mean-normal.mae_mean:+.6f} mV다. 겹치는 질환과 표본 구성이 달라 질환 자체의 효과로 해석할 수 없다."])
        small = s[(s.n_records > 0) & (s.n_records < 10)]
        sections.extend(["", "표본 10개 미만(주의 표시용 기준이며 통계적 기준은 아님): " + (", ".join(f"{r.group}={r.n_records}" for r in small.itertuples()) or "없음") + "."])
    sections.extend(["", "## 파형 예시", "", "각 split·그룹에서 ecg_id가 가장 작은 기록을 선정했다. 오차나 시각적 유사도를 보고 선택하지 않았다. 겹치는 질환 그룹은 동일 ECG가 재등장할 수 있다. 대표성 또는 최악/최선 사례를 의미하지 않는다.", "", markdown_table(chosen)])
    for r in chosen.itertuples():
        sections.extend(["", f"![{r.split} {r.group} ECG {r.ecg_id}]({r.path})"])
    sections.extend(["", "## 해석과 한계", "", "MAE/RMSE가 작을수록 진폭 오차가 작고 r이 클수록 선형 형태 유사도가 높다. r이 높아도 진폭·기준선 오차가 클 수 있다. 전체 10초 평균은 국소 ST 변화, QRS 형태 등의 보존을 보증하지 않는다.",
                     "", "질환 다중 라벨로 그룹이 독립적이지 않으며 합계가 고유 ECG 수를 넘을 수 있다. 작은 그룹의 평균과 표준편차는 불안정하다. 유의성 검정, 환자군 보정, 임상 유효성 평가를 수행하지 않았다. validation은 기존 모델 선택에 사용된 자료이므로 test와 분리해 기술한다. 같은 분할·단일 전달 재구성 배열의 기술 분석이며 다른 seed나 외부 환자군으로 일반화하지 않는다.",
                     "", "정상/질환 라벨의 정확도와 전달 재구성의 생성 과정은 이 분석만으로 검증할 수 없다. 원본 actual·ID·split 일치가 재구성의 완전한 독립 추론 또는 학습 누출 부재를 보증하지 않는다. 파형 재구성 지표를 질환 분류 정확도·민감도·특이도·AUC로 해석하지 않는다.", "", "실행 방법과 입력 규격: [분석 안내](../../../docs/DISEASE_RECONSTRUCTION_ANALYSIS.md)."])
    (out / "REPORT.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=PROJECT / "data/received/ecg_labels_analysis_998.csv")
    p.add_argument("--reconstructions", type=Path, default=PROJECT / "data/received/ecg_for_classification.npz")
    p.add_argument("--reference-dataset", type=Path, default=PROJECT / "data/processed/ptbxl_100hz_1000records.npz")
    p.add_argument("--split-config", type=Path, default=CONFIG)
    p.add_argument("--classes-column", help="Explicit superclass list column; inclusion flags are not class labels")
    p.add_argument("--waveform-layout", choices=("auto", "NTC", "NCT"), default="auto")
    p.add_argument("--output", type=Path, help="New directory only; existing results never overwritten")
    p.add_argument("--validate-only", action="store_true", help="Write input audit only; no metric calculations")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    out = args.output or PROJECT / "results/disease_reconstruction" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    audit = {"status": "started", "created_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
             "source_sha256": sha256(Path(__file__)), "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
             "metric_scope": "All supplied ECGs; train metrics are descriptive only. Group summaries and examples use validation/test separately.",
             "groups_policy": "NORM only; non-NORM diseases overlap; NORM+disease isolated; empty classes unclassified; include_in_analysis never filters",
             "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        joined, actual, reconstructed, leads, rate = load_inputs(args, out, audit)
        if not args.validate_only:
            metrics, records = make_metrics(joined, actual, reconstructed, leads)
            targets = [s for s in leads if s.upper() != "I"]
            counts, overlap, summary = summarize(metrics, records, targets)
            for name, frame in (("ecg_lead_metrics", metrics), ("ecg_overall_metrics", records), ("group_counts", counts), ("group_overlap", overlap), ("group_lead_metrics", summary),
                                ("correlation_issues", metrics[metrics.correlation_status != "valid"])):
                save_csv(frame, out / f"{name}.csv")
            comparison_plots(out, summary, records, targets)
            (out / "waveforms").mkdir()
            chosen = examples(out, joined, records, actual, reconstructed, leads, rate)
            save_csv(chosen, out / "waveform_examples.csv")
            report(out, audit, summary, chosen)
            audit["status"] = "complete"
        save_json(out / "audit.json", audit)
        print(f"{audit['status']}: {out}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        audit.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        save_json(out / "audit.json", audit)
        print(f"BLOCKED: {exc}\nAudit: {out / 'audit.json'}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
