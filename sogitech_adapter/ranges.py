"""Parameter range loader and utilities for Sogitech ↔ TOTAL BEATS adapter.

Fails loudly if ranges.json still contains TODOs or is unverified.
Same discipline as Sogitech refusing random controller weights.
"""
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


class RangeError(RuntimeError):
    """Raised when ranges.json is incomplete, unverified, or malformed."""
    pass


class ParamRanges:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise RangeError(f"ranges.json not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        meta = raw.pop("_meta", {})
        self.guards = raw.pop("_guards", {})
        self.contract = meta.get("contract", "unknown")
        self.fields: Dict[str, Dict[str, Any]] = {}
        for name, spec in raw.items():
            self.fields[name] = spec

        self._validate(meta)

    def _validate(self, meta: dict) -> None:
        if not meta.get("verified", False):
            todos = [k for k, v in self.fields.items() if v.get("status") == "TODO"]
            raise RangeError(
                f"ranges.json not verified ({len(todos)} TODO fields: {todos[:5]}...). "
                f"Fill from repo constants and set _meta.verified=true before running."
            )
        missing = [f for f in REQUIRED_18 if f not in self.fields]
        if missing:
            raise RangeError(f"missing fields: {missing}")
        if len(self.fields) != 18:
            raise RangeError(f"expected 18 fields, got {len(self.fields)}")

    def clamp(self, name: str, v: float) -> float:
        s = self.fields[name]
        if s["scale"] == "enum":
            return int(round(max(s["min"], min(s["max"], v))))
        return max(s["min"], min(s["max"], v))

    def to_norm(self, name: str, v: float) -> float:
        """physical → [0,1], for loss computation."""
        s = self.fields[name]
        if s["scale"] == "log":
            lo, hi = math.log(s["min"]), math.log(s["max"])
            return (math.log(max(v, s["min"])) - lo) / (hi - lo)
        if s["scale"] == "log1p":
            lo, hi = math.log1p(s["min"]), math.log1p(s["max"])
            return (math.log1p(max(v, s["min"])) - lo) / (hi - lo)
        return (v - s["min"]) / (s["max"] - s["min"])

    def from_norm(self, name: str, u: float) -> float:
        """[0,1] → physical, for the controller output layer."""
        s = self.fields[name]
        u = max(0.0, min(1.0, u))
        if s["scale"] == "log":
            return math.exp(math.log(s["min"]) + u * (math.log(s["max"]) - math.log(s["min"])))
        if s["scale"] == "log1p":
            return math.expm1(math.log1p(s["min"]) + u * (math.log1p(s["max"]) - math.log1p(s["min"])))
        return s["min"] + u * (s["max"] - s["min"])

    def apply_musical_ceilings(self, frame: dict) -> dict:
        """Apply guard-layer musical ceilings (powerSag, residual)."""
        out = dict(frame)
        if "powerSagDepth" in out:
            ceiling = self.guards.get("power_sag_musical_ceiling", 0.6)
            out["powerSagDepth"] = min(out["powerSagDepth"], ceiling)
        if "residualGain" in out:
            ceiling = self.guards.get("residual_musical_ceiling", 0.3)
            out["residualGain"] = min(out["residualGain"], ceiling)
        return out
