"""Frame emission and timing policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import copy


@dataclass
class FramePolicy:
    """Controls how and when parameter frames are emitted."""
    mode: str = "static"              # "static" | "evolving"
    control_rate_hz: float = 100.0
    smooth_tau_ms: float = 30.0       # exponential smoothing time constant
    max_delta_per_10ms: float = 0.08


class FrameSmoother:
    """Per-field exponential smoothing with rate limiting.

    Important: discrete fields (midiNote) and some expressive fields
    should not be smoothed the same way as continuous synthesis controls.
    """

    # Fields that must jump immediately (no smoothing)
    INSTANT_FIELDS = {"midiNote"}

    # Fields that may be lightly smoothed
    EXPRESSIVE_FIELDS = {"velocity", "pressure"}

    def __init__(self, policy: FramePolicy):
        self.policy = policy
        self.prev: Optional[Dict[str, float]] = None

    def process(self, frame: Dict[str, float], dt_ms: float = 10.0) -> Dict[str, float]:
        if self.prev is None or self.policy.mode == "static":
            self.prev = copy.deepcopy(frame)
            return frame

        out = {}
        alpha = self._alpha(dt_ms)
        max_delta = self.policy.max_delta_per_10ms * (dt_ms / 10.0)

        for k, v in frame.items():
            prev = self.prev.get(k, v)

            if k in self.INSTANT_FIELDS:
                out[k] = v
                continue

            # rate limit
            delta = v - prev
            if abs(delta) > max_delta:
                v = prev + max_delta * (1.0 if delta > 0 else -1.0)

            # exponential approach
            if k in self.EXPRESSIVE_FIELDS:
                # lighter smoothing for velocity / pressure
                out[k] = prev + 0.5 * (v - prev)
            else:
                out[k] = prev + alpha * (v - prev)

        self.prev = copy.deepcopy(out)
        return out

    def _alpha(self, dt_ms: float) -> float:
        # simple conversion from time-constant to mixing coefficient
        tau = max(self.policy.smooth_tau_ms, 1.0)
        return min(1.0, dt_ms / tau)
