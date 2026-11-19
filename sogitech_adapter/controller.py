"""Strict loader for trained v1-64 Sogitech parameter controllers.

The controller predicts twelve continuous synthesis controls and two categorical
waveform selections. Note identity/velocity/pressure are event passthroughs and
glideTime is a rule output, so the network does not waste targets on them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import math
import re
import numpy as np

from .ranges import ParamRanges, REQUIRED_18

CONTROLLER_CONTRACT = "sogitech.controller.v1-64"
FEATURE_CONTRACT = "v1-64"
PARAMETER_CONTRACT = "sogitech.params.v1"
ENUM_NAMES = ("osc1Wave", "osc2Wave")
PASSTHROUGH_NAMES = ("midiNote", "velocity", "pressure")
RULE_NAMES = ("glideTime",)
CONTINUOUS_NAMES = tuple(
    name for name in REQUIRED_18
    if name not in ENUM_NAMES + PASSTHROUGH_NAMES + RULE_NAMES
)
CONTROLLER_OUTPUT_NAMES = tuple(
    name for name in REQUIRED_18 if name not in PASSTHROUGH_NAMES + RULE_NAMES
)


class ControllerError(RuntimeError):
    """Raised when a trained controller checkpoint is incompatible or invalid."""


def build_network(torch_module=None):
    """Build the deterministic architecture; import PyTorch only when requested."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise ControllerError("PyTorch is required; install the neural extra") from exc

    class Network(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch_module.nn.Sequential(
                torch_module.nn.Linear(64, 128),
                torch_module.nn.ReLU(),
                torch_module.nn.Linear(128, 128),
                torch_module.nn.ReLU(),
            )
            self.continuous = torch_module.nn.Sequential(
                torch_module.nn.Linear(128, len(CONTINUOUS_NAMES)),
                torch_module.nn.Sigmoid(),
            )
            self.osc1_wave = torch_module.nn.Linear(128, 4)
            self.osc2_wave = torch_module.nn.Linear(128, 4)

        def forward(self, features):
            hidden = self.trunk(features)
            return self.continuous(hidden), self.osc1_wave(hidden), self.osc2_wave(hidden)

    return Network()


class TrainedParameterController:
    """Inference wrapper implementing the adapter's predict_dict interface."""

    def __init__(self, network, ranges: ParamRanges, torch_module):
        self.network = network
        self.ranges = ranges
        self.torch = torch_module
        self.network.eval()

    def predict_dict(self, features: np.ndarray) -> Dict[str, float]:
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if values.shape != (64,):
            raise ValueError(f"controller expects 64 features, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("controller features must be finite")
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(values, dtype=self.torch.float32).reshape(1, 64)
            continuous, osc1_logits, osc2_logits = self.network(tensor)
            continuous_values = continuous.squeeze(0).cpu().numpy()
            osc1_index = int(self.torch.argmax(osc1_logits, dim=-1).item())
            osc2_index = int(self.torch.argmax(osc2_logits, dim=-1).item())

        if (continuous_values.shape != (len(CONTINUOUS_NAMES),)
                or not np.all(np.isfinite(continuous_values))):
            raise ControllerError("controller produced invalid continuous outputs")

        result = {
            name: self.ranges.from_norm(name, float(continuous_values[index]))
            for index, name in enumerate(CONTINUOUS_NAMES)
        }
        result["osc1Wave"] = self.ranges.clamp("osc1Wave", osc1_index)
        result["osc2Wave"] = self.ranges.clamp("osc2Wave", osc2_index)
        if set(result) != set(CONTROLLER_OUTPUT_NAMES):
            raise ControllerError("controller output does not match the synthesis-control schema")
        return result


def _validate_checkpoint_metadata(checkpoint: dict, ranges: ParamRanges) -> None:
    expected = {
        "controller_contract": CONTROLLER_CONTRACT,
        "feature_contract": FEATURE_CONTRACT,
        "parameter_contract": PARAMETER_CONTRACT,
        "parameter_names": list(REQUIRED_18),
        "output_names": list(CONTROLLER_OUTPUT_NAMES),
        "enum_names": list(ENUM_NAMES),
        "continuous_names": list(CONTINUOUS_NAMES),
        "range_source_commit": ranges.source_commit,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ControllerError(
                f"checkpoint {key} mismatch: expected {value!r}, got {checkpoint.get(key)!r}"
            )

    digest = checkpoint.get("dataset_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ControllerError("checkpoint dataset_sha256 must be a 64-character SHA-256 digest")
    provenance = checkpoint.get("dataset_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ControllerError("checkpoint must contain non-empty dataset provenance")

    count_fields = ("sample_count", "training_sample_count",
                    "validation_sample_count", "source_group_count")
    counts = {}
    for field in count_fields:
        value = checkpoint.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ControllerError(f"checkpoint {field} must be a positive integer")
        counts[field] = value
    if counts["sample_count"] != counts["training_sample_count"] + counts["validation_sample_count"]:
        raise ControllerError("checkpoint train/validation counts do not sum to sample_count")
    if counts["sample_count"] < 10 or counts["source_group_count"] < 2:
        raise ControllerError("checkpoint dataset is below the minimum sample/group requirement")

    metrics = checkpoint.get("validation_metrics")
    metric_names = {
        "loss", "continuous_mae_normalized",
        "osc1_wave_accuracy", "osc2_wave_accuracy",
    }
    if not isinstance(metrics, dict) or set(metrics) != metric_names:
        raise ControllerError("checkpoint validation_metrics schema is invalid")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ControllerError(f"checkpoint validation metric {name} must be finite")
    if metrics["loss"] < 0 or not 0 <= metrics["continuous_mae_normalized"] <= 1:
        raise ControllerError("checkpoint loss/MAE metrics are outside their valid ranges")
    for name in ("osc1_wave_accuracy", "osc2_wave_accuracy"):
        if not 0 <= metrics[name] <= 1:
            raise ControllerError(f"checkpoint {name} must be in [0, 1]")


def load_controller(path: str | Path, ranges: ParamRanges) -> TrainedParameterController:
    """Load only a checkpoint explicitly trained for this feature/range contract."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"controller checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("controller checkpoint must use .pt or .pth")

    try:
        import torch
    except ImportError as exc:
        raise ControllerError("PyTorch is required to load controller checkpoints; install the neural extra") from exc

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ControllerError(f"unable to read controller checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ControllerError("checkpoint must contain a state_dict and versioned metadata")
    _validate_checkpoint_metadata(checkpoint, ranges)

    network = build_network(torch)
    try:
        network.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ControllerError(f"checkpoint weights do not match controller architecture: {exc}") from exc
    network.eval()
    return TrainedParameterController(network, ranges, torch)
