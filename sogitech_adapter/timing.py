"""Frame emission and timing policy with event-safe passthrough fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import copy
import math


@dataclass
class FramePolicy:
    """Controls frame interpolation; event identity and expression stay exact."""
    mode: str = "static"
    control_rate_hz: float = 100.0
    smooth_tau_ms: float = 30.0
    max_delta_per_10ms: float = 0.08

    def __post_init__(self) -> None:
        if self.mode not in {"static", "evolving"}:
            raise ValueError("mode must be 'static' or 'evolving'")
        for name, value in (("control_rate_hz", self.control_rate_hz),
                            ("smooth_tau_ms", self.smooth_tau_ms),
                            ("max_delta_per_10ms", self.max_delta_per_10ms)):
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, TypeError, ValueError):
                finite = False
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not finite or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")


class FrameSmoother:
    """Rate-limit and smooth only continuous synthesis controls.

    Note identity, note velocity/pressure, and oscillator waveform selections
    are event/discrete values and must not be interpolated.
    """
    INSTANT_FIELDS = {"midiNote", "velocity", "pressure", "osc1Wave", "osc2Wave"}

    def __init__(self, policy: FramePolicy):
        self.policy = policy
        self.prev: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self.prev = None

    def process(self, frame: Dict[str, float], dt_ms: Optional[float] = None) -> Dict[str, float]:
        if dt_ms is None:
            dt_ms = 1000.0 / self.policy.control_rate_hz
        try:
            dt_ms = float(dt_ms)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("dt_ms must be a finite non-negative number") from exc
        if not math.isfinite(dt_ms) or dt_ms < 0:
            raise ValueError("dt_ms must be a finite non-negative number")
        if self.prev is None or self.policy.mode == "static" or dt_ms == 0:
            self.prev = copy.deepcopy(frame)
            return dict(frame)

        alpha = 1.0 - math.exp(-dt_ms / self.policy.smooth_tau_ms)
        max_delta = self.policy.max_delta_per_10ms * (dt_ms / 10.0)
        out: Dict[str, float] = {}
        for key, target in frame.items():
            previous = self.prev.get(key, target)
            if key in self.INSTANT_FIELDS:
                out[key] = target
                continue
            delta = target - previous
            if abs(delta) > max_delta:
                target = previous + math.copysign(max_delta, delta)
            out[key] = previous + alpha * (target - previous)

        self.prev = copy.deepcopy(out)
        return out
