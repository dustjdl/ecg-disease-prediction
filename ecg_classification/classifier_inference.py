"""Role-4 classifier interface: NumPy + PyTorch only; no training data access.

CLI accepts signal [1000,C] or [N,1000,C] with explicit lead/unit/rate metadata.
Import ECGClassifierPredictor for in-process integration. See CLASSIFIER_INTERFACE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class InferenceCNN(nn.Module):
    """Exact trained architecture, independently usable without training imports."""
    def __init__(self, channels, outputs):
        super().__init__()
        # Construction must not change the embedding application's torch RNG.
        with torch.random.fork_rng(devices=[]):
            self.first = nn.Conv1d(channels, 32, 7, stride=2, padding=3)
            self.body = nn.Sequential(nn.ReLU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
                                      nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, outputs))

    def forward(self, x):
        z = self.body(self.first(x))
        return self.head(torch.cat((z.mean(dim=2), z.amax(dim=2)), dim=1))


def text_array(value):
    a = np.asarray(value)
    require(a.dtype.kind in "US", "Expected Unicode/bytes metadata; no object/pickle")
    return np.char.decode(a, "utf-8") if a.dtype.kind == "S" else a.astype(str)


def identify_input_kind(lead_names, declared=None, is_estimated_lead=None):
    names = [str(s).upper() for s in lead_names]
    require(names and len(names) == len(set(names)), "Missing or duplicate lead names")
    inferred = None
    if is_estimated_lead is not None:
        flags = np.asarray(is_estimated_lead)
        require(flags.shape == (len(names),) and flags.dtype.kind == "b", "is_estimated_lead must be boolean per lead")
        if names == ["I"] and not flags.any():
            inferred = "A"
        elif set(names) == set(LEADS):
            if not flags.any():
                inferred = "B"
            elif not flags[names.index("I")] and all(flags[i] for i, n in enumerate(names) if n != "I"):
                inferred = "C"
        require(inferred is not None, "Estimated/observed lead flags do not match A, B, or C")
    if declared is not None:
        require(declared in ("A", "B", "C"), "input_kind must be A, B, or C")
        require(inferred is None or declared == inferred, "input_kind contradicts is_estimated_lead")
    kind = declared or inferred
    require(kind is not None, "Declare input_kind or supply is_estimated_lead; 12-lead shape cannot distinguish B/C")
    require(set(names) == ({"I"} if kind == "A" else set(LEADS)), "Input kind and lead names disagree")
    return kind


class ECGClassifierPredictor:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = Path(model_dir)
        require(device in ("cpu", "cuda"), "device must be cpu or cuda")
        require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
        self.device = torch.device(device)
        checkpoint_path = self.model_dir / "best_model.pth"
        threshold_path = self.model_dir / "thresholds.json"
        selection = json.loads((self.model_dir / "selection.json").read_text(encoding="utf-8"))
        self.checkpoint_sha256 = sha256(checkpoint_path)
        self.threshold_sha256 = sha256(threshold_path)
        require(self.checkpoint_sha256 == selection["checkpoint_sha256"], "Checkpoint does not match frozen selection hash")
        require(self.threshold_sha256 == selection["thresholds_sha256"], "Thresholds do not match frozen selection hash")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        self.kind, self.task, self.seed = cfg["input"], cfg["task"], int(cfg["seed"])
        require(self.kind in ("A", "B", "C") and self.task in ("multilabel", "binary"), "Unsupported checkpoint task/input")
        require(all(selection[k] == cfg[k] for k in ("task", "input", "seed")), "Selection/checkpoint identity mismatch")
        require(checkpoint["epoch"] == selection["best_epoch"], "Checkpoint epoch mismatch")
        self.labels = cfg["labels"]
        require(self.labels == (LABELS if self.task == "multilabel" else ["ABNORMAL"]), "Unexpected output label order")
        self.expected_leads = ["I"] if self.kind == "A" else LEADS
        self.mean = np.asarray(cfg["normalization_mean"], dtype=np.float64)
        self.sd = np.asarray(cfg["normalization_sd"], dtype=np.float64)
        require(self.mean.shape == self.sd.shape == (len(self.expected_leads),), "Normalization shape mismatch")
        require(np.isfinite(self.mean).all() and np.isfinite(self.sd).all() and (self.sd > 0).all(), "Invalid saved normalization")
        details = json.loads(threshold_path.read_text(encoding="utf-8"))
        require([r["label"] for r in details] == self.labels, "Threshold label order mismatch")
        self.thresholds = np.asarray([r["threshold"] for r in details], dtype=np.float64)
        require(np.isfinite(self.thresholds).all() and ((self.thresholds >= 0) & (self.thresholds <= 1)).all(), "Invalid thresholds")
        require(self.thresholds.tolist() == selection["thresholds"], "Threshold file and selection disagree")
        self.model = InferenceCNN(len(self.expected_leads), len(self.labels))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        require(all(torch.isfinite(v).all().item() for v in self.model.state_dict().values()), "Nonfinite model weights")
        self.model.to(self.device).eval()
        self.epoch = int(checkpoint["epoch"])

    def predict(self, signal, *, lead_names, sampling_rate_hz, voltage_unit, input_kind=None,
                is_estimated_lead=None, record_ids=None):
        require(float(sampling_rate_hz) == 100., "Expected 100 Hz; resampling is not performed")
        require(voltage_unit == "mV", "Expected mV; unit conversion is not performed")
        require(isinstance(lead_names, (list, tuple, np.ndarray)), "lead_names must be a vector")
        require(np.asarray(lead_names).ndim == 1, "lead_names must be one-dimensional")
        names = [str(s).upper() for s in lead_names]
        kind = identify_input_kind(names, input_kind, is_estimated_lead)
        require(kind == self.kind, f"Model expects input {self.kind}, received {kind}")
        a = np.asarray(signal)
        require(a.dtype.kind in "fiu" and a.ndim in (2, 3), "signal must be real numeric [1000,C] or [N,1000,C]")
        if a.ndim == 2:
            a = a[None, :, :]
        require(a.shape[0] > 0 and a.shape[1:] == (1000, len(names)), "Expected 1000 time samples and channels matching lead_names")
        require(np.isfinite(a).all(), "Nonfinite waveform values")
        if record_ids is not None:
            require(len(record_ids) == len(a), "record_ids length must match batch size")
        order = [names.index(name) for name in self.expected_leads]
        ordered = a[:, :, order].transpose(0, 2, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.ascontiguousarray((ordered.astype(np.float64) - self.mean[None, :, None]) / self.sd[None, :, None], dtype=np.float32)
        require(np.isfinite(x).all(), "Input overflows normalized float32 range")
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(x), 128):
                outputs.append(self.model(torch.from_numpy(x[start:start+128]).to(self.device)).sigmoid().cpu().numpy())
        probability = np.concatenate(outputs)
        require(np.isfinite(probability).all(), "Nonfinite classifier output")
        decisions = probability >= self.thresholds
        records = []
        for i, (scores, pred) in enumerate(zip(probability, decisions)):
            record = {"record_index": i,
                      "scores": [{"label": label, "probability": float(scores[k]), "threshold": float(self.thresholds[k]), "predicted": bool(pred[k])}
                                 for k, label in enumerate(self.labels)],
                      "predicted_labels": [label for label, active in zip(self.labels, pred) if active]}
            if record_ids is not None:
                value = record_ids[i]
                record["record_id"] = value.item() if isinstance(value, np.generic) else value
            if self.task == "binary":
                record["binary_decision"] = "ABNORMAL" if pred[0] else "NORMAL"
            records.append(record)
        return {"schema_version": 1, "task": self.task, "input_kind": self.kind, "seed": self.seed,
                "labels": self.labels, "n_records": len(a), "best_epoch": self.epoch,
                "checkpoint_sha256": self.checkpoint_sha256, "thresholds_sha256": self.threshold_sha256,
                "score_type": "sigmoid; not probability-calibrated", "label_policy_status": "label criteria applied to this experiment",
                "records": records}


def load_input(path):
    with np.load(path, allow_pickle=False) as z:
        require({"signal", "lead_names", "sampling_rate_hz", "voltage_unit"} <= set(z.files), "NPZ needs signal, lead_names, sampling_rate_hz, voltage_unit")
        require(z["sampling_rate_hz"].shape == () and z["voltage_unit"].shape == (), "Rate/unit metadata must be scalar")
        names = text_array(z["lead_names"])
        require(names.ndim == 1, "lead_names must be one-dimensional")
        return {"signal": z["signal"], "lead_names": names.tolist(), "sampling_rate_hz": float(z["sampling_rate_hz"].item()),
                "voltage_unit": str(text_array(z["voltage_unit"]).item()),
                "input_kind": str(text_array(z["input_kind"]).item()) if "input_kind" in z else None,
                "is_estimated_lead": z["is_estimated_lead"] if "is_estimated_lead" in z else None,
                "record_ids": np.atleast_1d(z["record_ids"]).tolist() if "record_ids" in z else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing files are not overwritten")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        model = ECGClassifierPredictor(args.model_dir, args.device)
        result = model.predict(**load_input(args.input))
        result["input_file_sha256"] = sha256(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(f"Saved {result['n_records']} predictions.")
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        parser.exit(2, f"Inference failed: {exc}\n")


if __name__ == "__main__":
    main()
