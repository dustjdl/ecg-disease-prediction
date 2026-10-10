"""Infer 11 leads from one explicitly labelled 100 Hz, mV Lead I .npy file.

Only this input and the requested checkpoint are read. No resampling, signal
normalization, filtering, clipping, target comparison or metric calculation.
Outputs are created in a unique results/ecg_predictions/run_* directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ECGModel
from model_dilated import ECGDilatedModel

PROJECT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT / "results/ecg_predictions"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PREPROCESSING = {"normalization": "none", "clipping": "none", "filtering": "none",
                 "augmentation": "none", "axis_conversion": "NTC -> NCT", "voltage_unit": "mV"}
ESTIMATED = np.asarray([False] + [True] * 11, dtype=np.bool_)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    require(name in ("cpu", "cuda"), "Supported devices: auto, cpu, cuda")
    require(name != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    return torch.device(name)


def load_lead_i(path, sampling_rate_hz, unit):
    require(sampling_rate_hz == 100 and unit == "mV", "Only 100 Hz and mV are supported")
    require(path.suffix.lower() == ".npy", "Input must be a .npy file")
    digest = sha256_file(path)
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
        if not isinstance(values, np.ndarray):
            values.close()
            raise ValueError("Input must contain one .npy array, not an archive")
    require(sha256_file(path) == digest, "Input changed while being loaded")
    require(values.shape in ((1000,), (1000, 1)), f"Unsupported Lead I shape: {values.shape}")
    require(np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating),
            f"Lead I must be real numeric data, got {values.dtype}")
    require(np.isfinite(values).all(), "Lead I contains NaN/inf before float32 conversion")
    with np.errstate(over="ignore", invalid="ignore"):
        lead_i = np.array(values.reshape(1000), dtype=np.float32, order="C", copy=True)
    require(np.isfinite(lead_i).all(), "Lead I contains NaN/inf after float32 conversion")
    return lead_i, {"path": str(path.resolve()), "sha256": digest,
                    "original_shape": list(values.shape), "original_dtype": str(values.dtype),
                    "model_input_dtype": "float32", "sampling_rate_hz": 100, "voltage_unit": "mV",
                    "conversion": "float32 cast and reshape only; source float64/integer precision may round"}


def describe_structure(model):
    require(type(model) in (ECGModel, ECGDilatedModel), "Unsupported model class")
    dilated = type(model) is ECGDilatedModel
    expected_dilations, expected_paddings = ([1, 2, 4, 1], [3, 4, 8, 0]) if dilated else ([1]*4, [3, 2, 2, 0])
    expected_channels, kernels = [(1, 32), (32, 64), (64, 64), (64, 11)], [7, 5, 5, 1]
    require(len(model.network) == 7, "Expected exactly four convolutions and three ReLUs")
    layers, receptive = [], 1
    for index, layer in enumerate(model.network):
        if index % 2:
            require(type(layer) is nn.ReLU and not layer.inplace, "Expected unchanged ReLU")
            layers.append({"name": str(index), "type": "ReLU", "inplace": layer.inplace})
            continue
        conv = index // 2
        require(type(layer) is nn.Conv1d, "Expected Conv1d")
        require((layer.in_channels, layer.out_channels) == expected_channels[conv]
                and layer.kernel_size == (kernels[conv],) and layer.stride == (1,)
                and layer.dilation == (expected_dilations[conv],)
                and layer.padding == (expected_paddings[conv],)
                and layer.bias is not None and layer.groups == 1 and layer.padding_mode == "zeros",
                f"Unsupported runtime convolution configuration: layer {index}")
        receptive += (kernels[conv] - 1) * expected_dilations[conv]
        layers.append({"name": str(index), "type": "Conv1d", "in_channels": layer.in_channels,
                       "out_channels": layer.out_channels, "kernel_size": list(layer.kernel_size),
                       "stride": list(layer.stride), "dilation": list(layer.dilation),
                       "padding": list(layer.padding), "bias": True, "groups": layer.groups,
                       "padding_mode": layer.padding_mode})
    count = sum(p.numel() for p in model.parameters())
    require(count == 31819, "Unexpected parameter count")
    source = "model_dilated.py" if dilated else "model.py"
    return {"class": f"{type(model).__module__}.{type(model).__name__}", "representation": repr(model),
            "model_source_sha256": sha256_file(PROJECT / source),
            "state_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
            "parameter_count": count, "layers": layers, "receptive_field_samples": receptive,
            "receptive_field_ms_at_100hz": receptive * 10,
            "first_to_last_sample_ms_at_100hz": (receptive - 1) * 10}


def model_from_checkpoint(checkpoint):
    """Select from explicit class metadata, never from state tensor shapes."""
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary")
    metadata = checkpoint.get("experiment")
    require(isinstance(metadata, dict) and metadata.get("schema_version") == 1,
            "Missing/unsupported experiment metadata; architecture cannot be verified")
    saved = metadata.get("model_structure")
    require(isinstance(saved, dict), "Missing model structure metadata")
    name = saved.get("class")
    if name == "model.ECGModel":
        require(metadata.get("experiment_type") in (None, "baseline_1000"), "Conflicting baseline experiment type")
        model = ECGModel()
        expected = describe_structure(model)
        # Observed baseline schema has these five fields, but no layers list.
        # Its exact repr explicitly identifies all convolutions (default d=1),
        # and its source hash must match the actual imported model implementation.
        fields = ("class", "representation", "model_source_sha256", "state_shapes", "parameter_count")
        require(all(saved.get(key) == expected[key] for key in fields),
                "Baseline model representation/source/shape metadata mismatch")
        for key in set(saved).intersection(expected).difference(fields):
            require(saved[key] == expected[key], f"Baseline structure mismatch: {key}")
        verification = "exact stored representation/source hash plus explicit runtime layer checks"
    elif name == "model_dilated.ECGDilatedModel":
        require(metadata.get("experiment_type") == "dilated_1000", "Conflicting dilated experiment type")
        model = ECGDilatedModel()
        expected = describe_structure(model)
        require(saved == expected, "Dilated structure/source mismatch, including kernel/dilation/padding")
        verification = "exact full structure metadata and explicit runtime layer checks"
    else:
        raise ValueError(f"Unsupported or missing model class: {name!r}; refusing architecture inference")
    require(metadata.get("preprocessing") == PREPROCESSING, "Unsupported/missing preprocessing metadata")
    require(metadata.get("lead_names") == LEADS, "Unsupported/missing lead order")
    require(metadata.get("sampling_rate_hz") == 100, "Unsupported/missing checkpoint sampling rate")
    state = checkpoint.get("model_state_dict")
    expected_state = model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected_state), "State dictionary keys differ")
    for key, target in expected_state.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.shape == target.shape
                and value.dtype == target.dtype and torch.isfinite(value).all().item(),
                f"Invalid state shape/dtype/value: {key}")
    model.load_state_dict(state, strict=True)
    return model, expected, {"training_run_id": metadata.get("run_id"),
                             "checkpoint_epoch": checkpoint.get("epoch"), "structure_verification": verification}


def load_model(path):
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(sha256_file(path) == digest, "Checkpoint changed while being loaded")
    model, structure, info = model_from_checkpoint(checkpoint)
    return model, structure, {"path": str(path.resolve()), "sha256": digest, **info}


def predict_signal(model, lead_i, device):
    require(lead_i.shape == (1000,) and lead_i.dtype == np.float32 and np.isfinite(lead_i).all(),
            "Expected finite float32 Lead I with length 1000")
    inputs = torch.from_numpy(lead_i.copy()).reshape(1, 1, 1000).to(device)
    model.to(device).eval()
    with torch.inference_mode():
        output = model(inputs)
    require(output.shape == (1, 11, 1000) and output.dtype == torch.float32
            and torch.isfinite(output).all().item(), "Invalid output shape/dtype or nonfinite prediction")
    signal = np.concatenate((lead_i[:, None], output[0].cpu().numpy().T), axis=1)
    require(signal.shape == (1000, 12) and signal.dtype == np.float32 and np.isfinite(signal).all(),
            "Invalid 12-lead signal")
    require(signal[:, 0].tobytes() == lead_i.tobytes(), "Input Lead I was not preserved exactly after float32 conversion")
    return signal


def save_outputs(signal, input_info, checkpoint_info, model_structure, device):
    require(signal.shape == (1000, 12) and signal.dtype == np.float32 and np.isfinite(signal).all(),
            "Cannot save invalid signal")
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{stamp}_{uuid.uuid4().hex}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "estimated_12lead.npz").open("xb") as stream:
        np.savez_compressed(stream, signal=signal, lead_names=np.asarray(LEADS, dtype="U4"),
                            sampling_rate_hz=np.asarray(100, dtype=np.int64), voltage_unit=np.asarray("mV"),
                            is_estimated_lead=ESTIMATED)
    fig, axes = plt.subplots(12, 1, figsize=(16, 22), sharex=True, constrained_layout=True)
    try:
        time = np.arange(1000) / 100
        for index, (axis, lead) in enumerate(zip(axes, LEADS, strict=True)):
            kind = "Input Lead I" if index == 0 else f"Estimated {lead}"
            axis.plot(time, signal[:, index], color="tab:blue" if index == 0 else "tab:orange",
                      linewidth=0.8, label=kind)
            axis.set_ylabel(f"{lead} (mV)")
            axis.margins(y=0.1)
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0, 10)
        fig.suptitle("Input Lead I and 11 estimated leads")
        fig.savefig(output / "estimated_12lead.png", dpi=160)
    finally:
        plt.close(fig)
    execution = {"python": platform.python_version(), "pytorch": str(torch.__version__),
                 "numpy": str(np.__version__), "matplotlib": str(matplotlib.__version__),
                 "platform": platform.platform(), "device": str(device),
                 "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                 "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                 "cudnn_deterministic": torch.backends.cudnn.deterministic,
                 "cudnn_benchmark": torch.backends.cudnn.benchmark}
    metadata = {"run_id": run_id, "input": input_info, "checkpoint": checkpoint_info,
                "model_structure": model_structure, "preprocessing": PREPROCESSING,
                "input_handling": "float32 conversion and reshape only; no resampling",
                "lead_i_preservation": "bit-exact to the validated float32 input; checked before saving",
                "signal_shape": [1000, 12], "signal_dtype": "float32", "lead_names": LEADS,
                "sampling_rate_hz": 100, "voltage_unit": "mV", "is_estimated_lead": ESTIMATED.tolist(),
                "environment": execution, "program_sha256": sha256_file(Path(__file__).resolve())}
    with (output / "prediction_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sampling-rate-hz", type=float, choices=[100.0], required=True)
    parser.add_argument("--unit", choices=["mV"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def predict(args):
    lead_i, input_info = load_lead_i(args.input, args.sampling_rate_hz, args.unit)
    model, structure, checkpoint_info = load_model(args.checkpoint)
    device = select_device(args.device)
    signal = predict_signal(model, lead_i, device)
    output = save_outputs(signal, input_info, checkpoint_info, structure, device)
    print(f"Model: {structure['class']}; device: {device}")
    print(f"Saved (1000, 12): Lead I=input, remaining 11 leads=estimated\nResults: {output}")
    return output


def main():
    predict(parse_args())


if __name__ == "__main__":
    main()
