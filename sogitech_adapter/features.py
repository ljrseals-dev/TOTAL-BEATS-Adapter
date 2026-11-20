"""64-dimensional feature constructor for Sogitech neural parameter controller.

Contract: sogitech.features.v1-64
Frozen layout. Any change requires bumping CONTRACT_VERSION and a projection adapter.
"""
from __future__ import annotations

from typing import List, Any
import math
import numpy as np

CONTRACT_VERSION = "v1-64"
DIM = 64

FEATURE_NAMES: List[str] = [
    "midi_note_norm", "velocity", "duration_log", "brightness",
    "register_low", "register_mid", "register_high",
    "pc_sin", "pc_cos",
    "inst_pad", "inst_bell", "inst_bass", "inst_rhodes", "inst_ocean", "inst_glass",
    "pan",
    "chord_density", "signed_interval", "scale_sin", "scale_cos",
    "tempo_norm", "bar_phase", "rhythmic_density", "short_energy",
    "novelty", "ma_velocity", "ma_brightness", "ma_abs_pitch",
    "section_progress", "section_energy",
    "key_sin", "key_cos", "mode_major", "mode_minor",
    "atmosphere_0", "atmosphere_1",
    "brightness_hist_0", "brightness_hist_1", "brightness_hist_2", "brightness_hist_3",
    "brightness_hist_4", "brightness_hist_5", "brightness_hist_6", "brightness_hist_7",
    "audio_emb_0", "audio_emb_1", "audio_emb_2", "audio_emb_3",
    "audio_emb_4", "audio_emb_5", "audio_emb_6", "audio_emb_7",
    "audio_emb_8", "audio_emb_9", "audio_emb_10", "audio_emb_11",
    "audio_emb_12", "audio_emb_13", "audio_emb_14", "audio_emb_15",
    "pool_mean", "pool_var", "pool_min", "pool_max",
]

assert len(FEATURE_NAMES) == DIM

# Identity values are safe placeholders only; these are not fitted corpus statistics.
NORM_STATS = {name: (0.0, 1.0) for name in FEATURE_NAMES}

INSTRUMENT_SLOTS = {
    "pad": 0, "bell": 1, "bass": 2, "rhodes": 3,
    "ocean-bass": 4, "ocean": 4, "glass-rhodes": 5, "glass": 5,
}


def _one_hot(index: int, size: int) -> List[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _register_band(midi: int) -> List[float]:
    if midi < 48:
        return [1.0, 0.0, 0.0]
    if midi < 72:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _pitch_class(midi: int) -> List[float]:
    angle = 2.0 * math.pi * (midi % 12) / 12.0
    return [math.sin(angle), math.cos(angle)]


def build_features(ev: Any, ctx: Any) -> np.ndarray:
    """Build the frozen 64-dim feature vector (duck-typed)."""
    midi = int(getattr(ev, "midiNote", 60))
    velocity = float(getattr(ev, "velocity", 0.7))
    duration = float(getattr(ev, "duration", 1.0))
    brightness = float(getattr(ev, "brightness", 0.5))
    instrument = str(getattr(ev, "instrument", "pad")).lower().strip()
    pan = float(getattr(ev, "pan", 0.0))

    midi_norm = midi / 127.0
    dur_log = math.log1p(max(duration, 0.0)) / math.log1p(16.0)
    reg = _register_band(midi)
    pc = _pitch_class(midi)
    inst_oh = _one_hot(INSTRUMENT_SLOTS.get(instrument, -1), 6)

    A = [
        midi_norm, float(np.clip(velocity, 0.0, 1.0)),
        float(np.clip(dur_log, 0.0, 1.0)), float(np.clip(brightness, 0.0, 1.0)),
        *reg, *pc, *inst_oh, float(np.clip(pan, -1.0, 1.0)),
    ]

    previous_note = getattr(ctx, "last_note", None)
    last = midi if previous_note is None else int(previous_note)
    interval = (midi - last) / 24.0
    scale_angle = 2.0 * math.pi * (midi % 12) / 12.0

    B = [
        float(np.clip(getattr(ctx, "chord_density", 0.3), 0.0, 1.0)),
        float(np.clip(interval, -1.0, 1.0)),
        math.sin(scale_angle), math.cos(scale_angle),
        float(np.clip(getattr(ctx, "tempo", 90.0) / 200.0, 0.0, 1.0)),
        float(np.clip(getattr(ctx, "bar_phase", 0.0), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "rhythmic_density", 0.3), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "short_energy", 0.4), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "novelty", 0.0), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "ma_velocity", 0.5), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "ma_brightness", 0.5), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "ma_abs_pitch", 60.0) / 127.0, 0.0, 1.0)),
    ]

    key_root = int(getattr(ctx, "key_root", 0))
    mode = str(getattr(ctx, "mode", "minor")).lower()
    key_angle = 2.0 * math.pi * (key_root % 12) / 12.0
    major = 1.0 if mode in ("major", "ionian", "lydian", "mixolydian") else 0.0
    minor = 1.0 if mode in ("minor", "aeolian", "dorian", "phrygian") else 0.0
    atmosphere = str(getattr(ctx, "atmosphere", "drift")).lower()
    atmosphere_0 = 1.0 if atmosphere in ("drift", "ambient", "pad") else 0.0
    atmosphere_1 = 1.0 if atmosphere in ("pulse", "rhythmic", "drive") else 0.0

    C = [
        float(np.clip(getattr(ctx, "section_progress", 0.0), 0.0, 1.0)),
        float(np.clip(getattr(ctx, "section_energy", 0.5), 0.0, 1.0)),
        math.sin(key_angle), math.cos(key_angle),
        major, minor, atmosphere_0, atmosphere_1,
    ]

    hist = list(getattr(ctx, "brightness_history", [0.5] * 8))[:8]
    while len(hist) < 8:
        hist.append(0.5)
    D = [float(np.clip(value, 0.0, 1.0)) for value in hist]

    ref_audio = getattr(ctx, "ref_audio", None)
    if ref_audio is not None and len(ref_audio) > 0:
        audio = np.asarray(ref_audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("reference audio must contain only finite samples")
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-8))
        peak = float(np.max(np.abs(audio)) + 1e-8)
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))))) if audio.size > 1 else 0.0
        E = [float(np.clip(value, -1.0, 1.0))
             for value in ([rms, peak, zcr] + [0.0] * 13)]
    else:
        E = [0.0] * 16

    core = A + B + C + D
    F = [float(np.mean(core)), float(np.var(core)), float(np.min(core)), float(np.max(core))]
    vector = np.asarray(A + B + C + D + E + F, dtype=np.float64)
    if vector.shape != (DIM,):
        raise RuntimeError(f"feature contract broken: expected {(DIM,)}, got {vector.shape}")
    return np.clip(vector, -1.0, 1.0)


class EventStub:
    def __init__(self, **kwargs):
        self.midiNote = kwargs.get("midiNote", 60)
        self.velocity = kwargs.get("velocity", 0.7)
        self.duration = kwargs.get("duration", 1.0)
        self.brightness = kwargs.get("brightness", 0.5)
        self.instrument = kwargs.get("instrument", "pad")
        self.pan = kwargs.get("pan", 0.0)
        self.startTime = kwargs.get("startTime", 0.0)
        self.pressure = kwargs.get("pressure", None)


class ContextStub:
    def __init__(self, **kwargs):
        self.key_root = kwargs.get("key_root", 0)
        self.mode = kwargs.get("mode", "minor")
        self.tempo = kwargs.get("tempo", 90.0)
        self.bar_phase = kwargs.get("bar_phase", 0.0)
        self.section_index = kwargs.get("section_index", 0)
        self.section_progress = kwargs.get("section_progress", 0.0)
        self.section_energy = kwargs.get("section_energy", 0.5)
        self.atmosphere = kwargs.get("atmosphere", "drift")
        self.rhythmic_density = kwargs.get("rhythmic_density", 0.3)
        self.ref_audio = kwargs.get("ref_audio", None)
        self.last_note = kwargs.get("last_note", None)
        self.brightness_history = kwargs.get("brightness_history", [0.5] * 8)
        self.ma_velocity = kwargs.get("ma_velocity", 0.5)
        self.ma_brightness = kwargs.get("ma_brightness", 0.5)
        self.ma_abs_pitch = kwargs.get("ma_abs_pitch", 60.0)
        self.novelty = kwargs.get("novelty", 0.0)
        self.chord_density = kwargs.get("chord_density", 0.3)
        self.short_energy = kwargs.get("short_energy", 0.4)
