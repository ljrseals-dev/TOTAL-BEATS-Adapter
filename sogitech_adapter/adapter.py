"""Core adapter: MusicalEvent + context → 18-field Sogitech parameter frame."""
from __future__ import annotations

from typing import Dict, Optional
import math

from .features import build_features
from .ranges import ParamRanges
from .events import MusicalEvent, GlobalContext
from .guards import apply_all_guards
from .timing import FramePolicy, FrameSmoother
from .validation import validate_event, validate_frame

DEMO_DEFAULTS = {
    "osc1Wave": 1.0, "osc2Wave": 2.0,
    "osc1Level": 0.72, "osc2Level": 0.42,
    "subLevel": 0.25, "noiseLevel": 0.015,
    "additiveBlend": 0.25, "powerSagDepth": 0.2,
    "filterLP": 0.72, "filterHP": 0.04, "filterRes": 0.2,
    "keyTrack": 0.5, "velMod": 0.5,
    "glideTime": 0.03, "residualGain": 0.0,
}


class SogitechAdapter:
    """Translate compositional events into validated Sogitech parameter frames.

    Production use requires a loaded, trained controller. allow_safe_defaults
    is an explicit demo-only escape hatch; it never claims to be neural inference.
    """

    def __init__(
        self,
        ranges: ParamRanges,
        controller=None,
        policy: Optional[FramePolicy] = None,
        *,
        allow_safe_defaults: bool = False,
    ):
        if controller is None and not allow_safe_defaults:
            raise ValueError(
                "a trained controller is required; pass allow_safe_defaults=True "
                "only for demos and adapter contract tests"
            )
        self.ranges = ranges
        self.ctl = controller
        self.allow_safe_defaults = allow_safe_defaults
        self.policy = policy or FramePolicy(
            max_delta_per_10ms=float(ranges.guards["max_delta_per_10ms"])
        )
        self.smoother = FrameSmoother(self.policy)
        self.last_note: Optional[int] = None
        self.brightness_hist = [0.5] * 8
        self._history_count = 0
        self._ma_velocity = 0.5
        self._ma_brightness = 0.5
        self._ma_abs_pitch = 60.0
        self._last_event_time: Optional[float] = None

    def reset(self) -> None:
        """Reset per-stream musical and smoothing state between independent renders."""
        self.last_note = None
        self.brightness_hist = [0.5] * 8
        self._history_count = 0
        self._ma_velocity = 0.5
        self._ma_brightness = 0.5
        self._ma_abs_pitch = 60.0
        self._last_event_time = None
        self.smoother.reset()

    def on_event(
        self, ev: MusicalEvent, ctx: GlobalContext, t: Optional[float] = None
    ) -> Dict[str, float]:
        problems = validate_event(ev)
        if problems:
            raise ValueError(f"invalid MusicalEvent: {problems}")

        timestamp = ev.startTime if t is None else t
        if (not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("event timestamp must be a finite non-negative number")
        if self._last_event_time is not None and timestamp < self._last_event_time:
            raise ValueError("events must be supplied in non-decreasing time order")

        frame: Dict[str, float] = {
            "midiNote": float(ev.midiNote),
            "velocity": float(ev.velocity),
            "pressure": float(ev.pressure if ev.pressure is not None else 0.0),
        }

        # Supply features with state before this event, then advance statistics.
        ctx.last_note = self.last_note
        ctx.brightness_history = list(self.brightness_hist)
        self._history_count += 1
        n = float(self._history_count)
        self._ma_velocity += (float(ev.velocity) - self._ma_velocity) / n
        self._ma_brightness += (float(ev.brightness) - self._ma_brightness) / n
        self._ma_abs_pitch += (abs(float(ev.midiNote)) - self._ma_abs_pitch) / n
        ctx.ma_velocity = self._ma_velocity
        ctx.ma_brightness = self._ma_brightness
        ctx.ma_abs_pitch = self._ma_abs_pitch

        features = build_features(ev, ctx)
        if len(features) != 64:
            raise RuntimeError(f"feature contract broken: expected 64, got {len(features)}")

        if self.ctl is not None:
            pred = self.ctl.predict_dict(features)
            if not isinstance(pred, dict):
                raise RuntimeError("controller.predict_dict must return a dictionary")
        else:
            pred = dict(DEMO_DEFAULTS)

        pred = dict(pred)
        pred["glideTime"] = self._glide_rule(ev, ctx)
        for key, value in pred.items():
            if key not in self.ranges.fields:
                raise RuntimeError(f"controller returned unknown parameter: {key}")
            if key not in frame:
                frame[key] = self.ranges.clamp(key, value)

        if set(frame) != set(self.ranges.fields):
            missing = sorted(set(self.ranges.fields) - set(frame))
            extra = sorted(set(frame) - set(self.ranges.fields))
            raise RuntimeError(f"parameter frame schema mismatch; missing={missing}, extra={extra}")

        frame = apply_all_guards(frame, self.ranges)
        dt_ms = (1000.0 / self.policy.control_rate_hz if self._last_event_time is None
                 else (timestamp - self._last_event_time) * 1000.0)
        frame = self.smoother.process(frame, dt_ms=dt_ms)
        frame = apply_all_guards(frame, self.ranges)

        problems = validate_frame(frame, self.ranges)
        if problems:
            raise RuntimeError(f"guarded frame still invalid: {problems}")

        self.last_note = ev.midiNote
        self.brightness_hist = (self.brightness_hist + [float(ev.brightness)])[-8:]
        self._last_event_time = float(timestamp)
        return frame

    def _glide_rule(self, ev: MusicalEvent, ctx: GlobalContext) -> float:
        if self.last_note is None:
            return 0.0
        leap = abs(ev.midiNote - self.last_note)
        if ctx.rhythmic_density < 0.3 and leap >= 5:
            return min(0.5, (leap / 12.0) * 0.25)
        return 0.0
