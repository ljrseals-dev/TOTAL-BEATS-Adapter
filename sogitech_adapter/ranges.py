"""Parameter range loader and utilities for Sogitech ↔ TOTAL BEATS adapter."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

REQUIRED_18 = [
    "midiNote", "velocity", "pressure", "osc1Wave", "osc2Wave",
    "osc1Level", "osc2Level", "subLevel", "noiseLevel", "additiveBlend",
    "powerSagDepth", "filterLP", "filterHP", "filterRes", "keyTrack",
    "velMod", "glideTime", "residualGain",
]
SUPPORTED_SCALES = {"lin", "log", "log1p", "enum"}


class RangeError(RuntimeError):
    """Raised when ranges.json is incomplete, unverified, or malformed."""


class ParamRanges:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise RangeError(f"ranges.json not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RangeError(f"cannot read ranges file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RangeError("ranges.json root must be an object")

        raw = dict(raw)
        meta = raw.pop("_meta", {})
        self.guards = raw.pop("_guards", {})
        self.contract = meta.get("contract", "unknown") if isinstance(meta, dict) else "unknown"
        self.fields: Dict[str, Dict[str, Any]] = raw
        self._validate(meta)

    @staticmethod
    def _finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def _validate(self, meta: dict) -> None:
        if not isinstance(meta, dict):
            raise RangeError("_meta must be an object")
        if self.contract != "sogitech.params.v1":
            raise RangeError(f"unsupported parameter contract: {self.contract!r}")
        if meta.get("verified") is not True:
            raise RangeError("ranges.json is not marked verified")
        missing = [name for name in REQUIRED_18 if name not in self.fields]
        extra = [name for name in self.fields if name not in REQUIRED_18]
        if missing or extra or len(self.fields) != len(REQUIRED_18):
            raise RangeError(f"parameter keys mismatch; missing={missing}, extra={extra}")

        for name in REQUIRED_18:
            spec = self.fields[name]
            if not isinstance(spec, dict):
                raise RangeError(f"{name}: specification must be an object")
            lo, hi, scale = spec.get("min"), spec.get("max"), spec.get("scale")
            if not self._finite_number(lo) or not self._finite_number(hi) or lo >= hi:
                raise RangeError(f"{name}: min/max must be finite numbers with min < max")
            if scale not in SUPPORTED_SCALES:
                raise RangeError(f"{name}: unsupported scale {scale!r}")
            if spec.get("status") != "verified":
                raise RangeError(f"{name}: field is not marked verified")
            if scale == "log" and lo <= 0:
                raise RangeError(f"{name}: logarithmic scale requires min > 0")
            if scale == "log1p" and lo <= -1:
                raise RangeError(f"{name}: log1p scale requires min > -1")
            if scale == "enum":
                states = spec.get("enum_states")
                if (not isinstance(states, list) or not states
                        or not all(isinstance(item, str) and item for item in states)
                        or not float(lo).is_integer() or not float(hi).is_integer()
                        or int(hi - lo + 1) != len(states)):
                    raise RangeError(f"{name}: enum_states must match its integer range")

        if not isinstance(self.guards, dict):
            raise RangeError("_guards must be an object")
        ratio = self.guards.get("hp_lt_lp_ratio")
        min_total = self.guards.get("min_total_level")
        fallback = self.guards.get("fallback_osc1_level")
        for key, value in (("hp_lt_lp_ratio", ratio), ("min_total_level", min_total),
                           ("fallback_osc1_level", fallback),
                           ("power_sag_musical_ceiling", self.guards.get("power_sag_musical_ceiling")),
                           ("residual_musical_ceiling", self.guards.get("residual_musical_ceiling"))):
            if not self._finite_number(value):
                raise RangeError(f"_guards.{key} must be a finite number")
        if not 0.0 < ratio <= 1.0:
            raise RangeError("_guards.hp_lt_lp_ratio must be in (0, 1]")
        if not 0.0 <= min_total <= 4.0:
            raise RangeError("_guards.min_total_level must be in [0, 4]")
        if not self.fields["osc1Level"]["min"] <= fallback <= self.fields["osc1Level"]["max"]:
            raise RangeError("_guards.fallback_osc1_level is outside osc1Level range")
        for key, field_name in (("power_sag_musical_ceiling", "powerSagDepth"),
                                ("residual_musical_ceiling", "residualGain")):
            ceiling = self.guards[key]
            if not self.fields[field_name]["min"] <= ceiling <= self.fields[field_name]["max"]:
                raise RangeError(f"_guards.{key} is outside {field_name} range")
        coupled = self.guards.get("res_soft_couple_additive")
        if not isinstance(coupled, dict):
            raise RangeError("_guards.res_soft_couple_additive must be an object")
        trigger, ceiling = coupled.get("trigger_above"), coupled.get("res_ceiling")
        if not self._finite_number(trigger) or not 0 <= trigger <= 1:
            raise RangeError("_guards.res_soft_couple_additive.trigger_above must be in [0, 1]")
        if (not self._finite_number(ceiling)
                or not self.fields["filterRes"]["min"] <= ceiling <= self.fields["filterRes"]["max"]):
            raise RangeError("_guards.res_soft_couple_additive.res_ceiling is invalid")

    def _value(self, name: str, value: float) -> float:
        if name not in self.fields:
            raise KeyError(name)
        if not self._finite_number(value):
            raise ValueError(f"{name} must be finite")
        return float(value)

    def clamp(self, name: str, value: float) -> float:
        value = self._value(name, value)
        spec = self.fields[name]
        bounded = max(float(spec["min"]), min(float(spec["max"]), value))
        if spec["scale"] == "enum":
            bounded = float(round(bounded))
        return bounded

    def to_norm(self, name: str, value: float) -> float:
        """Map a physical value to [0, 1], safely handling range endpoints."""
        value = self.clamp(name, value)
        spec = self.fields[name]
        lo, hi = float(spec["min"]), float(spec["max"])
        if spec["scale"] == "log":
            return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
        if spec["scale"] == "log1p":
            return (math.log1p(value) - math.log1p(lo)) / (math.log1p(hi) - math.log1p(lo))
        return (value - lo) / (hi - lo)

    def from_norm(self, name: str, value: float) -> float:
        """Map a finite normalised value to its physical parameter range."""
        value = self._value(name, value)
        u = max(0.0, min(1.0, value))
        spec = self.fields[name]
        lo, hi = float(spec["min"]), float(spec["max"])
        if spec["scale"] == "log":
            physical = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        elif spec["scale"] == "log1p":
            physical = math.expm1(math.log1p(lo) + u * (math.log1p(hi) - math.log1p(lo)))
        else:
            physical = lo + u * (hi - lo)
        return self.clamp(name, physical)

    def apply_musical_ceilings(self, frame: dict) -> dict:
        """Apply documented musical ceilings without mutating the input."""
        out = dict(frame)
        if "powerSagDepth" in out:
            out["powerSagDepth"] = min(out["powerSagDepth"], self.guards["power_sag_musical_ceiling"])
        if "residualGain" in out:
            out["residualGain"] = min(out["residualGain"], self.guards["residual_musical_ceiling"])
        return out
