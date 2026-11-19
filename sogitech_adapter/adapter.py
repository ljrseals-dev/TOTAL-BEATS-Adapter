"""Core adapter: MusicalEvent + context → 18-field Sogitech parameter frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import copy

from .features import EventStub, ContextStub, build_features, CONTRACT_VERSION
from .ranges import ParamRanges


@dataclass
class FramePolicy:
    """Timing / emission policy."""
    mode: str = "static"          # "static" | "evolving"
    control_rate_hz: float = 100.0
    smooth_tau_ms: float = 30.0   # exponential smoothing time constant


class SogitechAdapter:
    """
    Translates TOTAL BEATS compositional output into Sogitech 18-field frames.

    Invariants:
    - Composition never emits synthesis parameters.
    - The 64-dim feature vector and 18-field frame are versioned contracts.
    """

    def __init__(
        self,
        ranges: ParamRanges,
        controller=None,
        policy: Optional[FramePolicy] = None,
    ):
        self.ranges = ranges
        self.ctl = controller          # must be a loaded, non-random controller
        self.policy = policy or FramePolicy()
        self.prev_frame: Optional[Dict[str, float]] = None
        self.last_note: Optional[int] = None
        self.brightness_hist = [0.5] * 8

    def on_event(self, ev: EventStub, ctx: ContextStub, t: float = 0.0) -> Dict[str, float]:
        # ---- pass-through fields
        frame: Dict[str, float] = {
            "midiNote": float(ev.midiNote),
            "velocity": float(ev.velocity),
            "pressure": float(ev.pressure if ev.pressure is not None else 0.0),
        }

        # ---- update context helpers
        ctx.last_note = self.last_note
        ctx.brightness_history = list(self.brightness_hist)
        self.brightness_hist = (self.brightness_hist + [ev.brightness])[-8:]

        # ---- features
        f = build_features(ev, ctx)
        assert len(f) == 64

        # ---- controller (or safe defaults if none loaded)
        if self.ctl is not None:
            pred = self.ctl.predict_dict(f)  # expects 64-dim → 18-param dict
        else:
            # Safe musical defaults when controller is not yet trained
            pred = {
                "osc1Wave": 1.0, "osc2Wave": 2.0,
                "osc1Level": 0.72, "osc2Level": 0.42,
                "subLevel": 0.25, "noiseLevel": 0.015,
                "additiveBlend": 0.25, "powerSagDepth": 0.2,
                "filterLP": 0.72, "filterHP": 0.04, "filterRes": 0.2,
                "keyTrack": 0.5, "velMod": 0.5,
                "glideTime": 0.03, "residualGain": 0.0,
            }

        # ---- rule-derived glide
        pred["glideTime"] = self._glide_rule(ev, ctx)

        # ---- merge
        for k, v in pred.items():
            if k not in frame:
                frame[k] = v

        # ---- clamp to legal ranges
        for k in list(frame.keys()):
            if k in self.ranges.fields:
                frame[k] = self.ranges.clamp(k, frame[k])

        # ---- musical guards
        frame = self.ranges.apply_musical_ceilings(frame)
        frame = self._guard_hp_lp(frame)
        frame = self._guard_levels(frame)
        frame = self._rate_limit(frame)

        self.prev_frame = copy.deepcopy(frame)
        self.last_note = ev.midiNote
        return frame

    def _glide_rule(self, ev: EventStub, ctx: ContextStub) -> float:
        if self.last_note is None:
            return 0.0
        leap = abs(ev.midiNote - self.last_note)
        sparse = ctx.rhythmic_density < 0.3
        if sparse and leap >= 5:
            return min(0.5, (leap / 12.0) * 0.25)
        return 0.0

    def _guard_hp_lp(self, frame: Dict[str, float]) -> Dict[str, float]:
        """Enforce HP < ratio * LP (normalised space)."""
        ratio = self.ranges.guards.get("hp_lt_lp_ratio", 0.5)
        lp = frame.get("filterLP", 0.7)
        hp = frame.get("filterHP", 0.05)
        if hp > ratio * lp:
            frame["filterHP"] = ratio * lp
        return frame

    def _guard_levels(self, frame: Dict[str, float]) -> Dict[str, float]:
        total = (
            frame.get("osc1Level", 0) + frame.get("osc2Level", 0) +
            frame.get("subLevel", 0) + frame.get("noiseLevel", 0)
        )
        min_total = self.ranges.guards.get("min_total_level", 0.01)
        if total < min_total:
            frame["osc1Level"] = self.ranges.guards.get("fallback_osc1_level", 0.05)
        return frame

    def _rate_limit(self, frame: Dict[str, float]) -> Dict[str, float]:
        if self.prev_frame is None:
            return frame
        max_delta = self.ranges.guards.get("max_delta_per_10ms", 0.08)
        out = {}
        for k, v in frame.items():
            prev = self.prev_frame.get(k, v)
            # simple hard clamp for now; exponential smoothing can replace this
            delta = v - prev
            if abs(delta) > max_delta:
                v = prev + max_delta * (1.0 if delta > 0 else -1.0)
            out[k] = v
        return out
