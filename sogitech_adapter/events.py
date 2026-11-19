"""Musical event and context data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
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

    def validate(self) -> None:
        if not (0 <= self.midiNote <= 127):
            raise ValueError(f"midiNote out of range: {self.midiNote}")
        if not (0.0 <= self.velocity <= 1.0):
            raise ValueError(f"velocity out of range: {self.velocity}")
        if self.duration < 0:
            raise ValueError(f"duration must be non-negative: {self.duration}")
        if not (0.0 <= self.brightness <= 1.0):
            raise ValueError(f"brightness out of range: {self.brightness}")
        if not (-1.0 <= self.pan <= 1.0):
            raise ValueError(f"pan out of range: {self.pan}")


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
