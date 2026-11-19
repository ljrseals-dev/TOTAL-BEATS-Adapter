"""Core adapter: MusicalEvent + context → 18-field Sogitech parameter frame."""
from __future__ import annotations

from typing import Dict, Optional
import copy

from .features import build_features, CONTRACT_VERSION
from .ranges import ParamRanges
from .events import MusicalEvent, GlobalContext
from .guards import apply_all_guards
from .timing import FramePolicy, FrameSmoother
from .validation import validate_event, validate_frame


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
        self.smoother = FrameSmoother(self.policy)
        self.last_note: Optional[int] = None
        self.brightness_hist = [0.5] * 8

    def on_event(self, ev: MusicalEvent, ctx: GlobalContext, t: float = 0.0) -> Dict[str, float]:
        problems = validate_event(ev)
        if problems:
            raise ValueError(f"invalid MusicalEvent: {problems}")

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

        # ---- features (still accepts EventStub/ContextStub shapes via duck typing)
        f = build_features(ev, ctx)
        assert len(f) == 64, f"feature contract broken: got {len(f)}"

        # ---- controller (or safe musical defaults if none loaded)
        if self.ctl is not None:
            pred = self.ctl.predict_dict(f)
        else:
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

        # ---- merge predictions
        for k, v in pred.items():
            if k not in frame:
                frame[k] = float(v)

        # ---- clamp to legal ranges
        for k in list(frame.keys()):
            if k in self.ranges.fields:
                frame[k] = self.ranges.clamp(k, frame[k])

        # ---- musical guards
        frame = apply_all_guards(frame, self.ranges)

        # ---- timing / smoothing
        frame = self.smoother.process(frame)

        # ---- final validation
        problems = validate_frame(frame, self.ranges)
        if problems:
            raise RuntimeError(f"guarded frame still invalid: {problems}")

        self.last_note = ev.midiNote
        return frame

    def _glide_rule(self, ev: MusicalEvent, ctx: GlobalContext) -> float:
        if self.last_note is None:
            return 0.0
        leap = abs(ev.midiNote - self.last_note)
        sparse = ctx.rhythmic_density < 0.3
        if sparse and leap >= 5:
            return min(0.5, (leap / 12.0) * 0.25)
        return 0.0
