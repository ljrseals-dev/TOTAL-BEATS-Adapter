"""Musical event and context data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math
import numpy as np


@dataclass
class MusicalEvent:
    """Contract A — compositional output from TOTAL BEATS / Neural Tide."""
    midiNote: int = 60
    velocity: float = 0.7
    duration: float = 1.0
    brightness: float = 0.5
    instrument: str = "pad"
    pan: float = 0.0
    startTime: float = 0.0
    pressure: Optional[float] = None

    @staticmethod
    def _finite(value, name: str) -> float:
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    def validate(self) -> None:
        if (not isinstance(self.midiNote, int) or isinstance(self.midiNote, bool)
                or not 0 <= self.midiNote <= 127):
            raise ValueError(f"midiNote must be an integer in [0, 127]: {self.midiNote!r}")
        velocity = self._finite(self.velocity, "velocity")
        duration = self._finite(self.duration, "duration")
        brightness = self._finite(self.brightness, "brightness")
        pan = self._finite(self.pan, "pan")
        start_time = self._finite(self.startTime, "startTime")
        if not 0.0 <= velocity <= 1.0:
            raise ValueError(f"velocity out of range: {velocity}")
        if duration < 0.0:
            raise ValueError(f"duration must be non-negative: {duration}")
        if not 0.0 <= brightness <= 1.0:
            raise ValueError(f"brightness out of range: {brightness}")
        if not -1.0 <= pan <= 1.0:
            raise ValueError(f"pan out of range: {pan}")
        if start_time < 0.0:
            raise ValueError(f"startTime must be non-negative: {start_time}")
        if self.pressure is not None:
            pressure = self._finite(self.pressure, "pressure")
            if not 0.0 <= pressure <= 1.0:
                raise ValueError(f"pressure out of range: {pressure}")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("instrument must be a non-empty string")


@dataclass
class GlobalContext:
    """Global / section-level context shared across events."""
    key_root: int = 0
    mode: str = "minor"
    tempo: float = 90.0
    bar_phase: float = 0.0
    section_index: int = 0
    section_progress: float = 0.0
    section_energy: float = 0.5
    atmosphere: str = "drift"
    rhythmic_density: float = 0.3
    ref_audio: Optional[np.ndarray] = None

    # Runtime history (filled by adapter)
    last_note: Optional[int] = None
    brightness_history: List[float] = field(default_factory=lambda: [0.5] * 8)
    ma_velocity: float = 0.5
    ma_brightness: float = 0.5
    ma_abs_pitch: float = 60.0
    novelty: float = 0.0
    chord_density: float = 0.3
    short_energy: float = 0.4
