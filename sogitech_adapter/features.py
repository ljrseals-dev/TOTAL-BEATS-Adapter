"""64-dimensional feature constructor for Sogitech neural parameter controller.

Contract: sogitech.features.v1-64
Frozen layout. Any change requires bumping CONTRACT_VERSION and a projection adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math
import numpy as np

CONTRACT_VERSION = "v1-64"
DIM = 64

FEATURE_NAMES: List[str] = [
    # A. Event (16)
    "midi_note_norm", "velocity", "duration_log", "brightness",
    "register_low", "register_mid", "register_high",
    "pc_sin", "pc_cos",
    "inst_pad", "inst_bell", "inst_bass", "inst_rhodes", "inst_ocean", "inst_glass",
    "pan",
    # B. Local context (12)
    "chord_density", "signed_interval", "scale_sin", "scale_cos",
    "tempo_norm", "bar_phase", "rhythmic_density", "short_energy",
    "novelty", "ma_velocity", "ma_brightness", "ma_abs_pitch",
    # C. Global / section (8)
    "section_progress", "section_energy",
    "key_sin", "key_cos", "mode_major", "mode_minor",
    "atmosphere_0", "atmosphere_1",
    # D. Temporal dynamics (8)
    "brightness_hist_0", "brightness_hist_1", "brightness_hist_2", "brightness_hist_3",
    "brightness_hist_4", "brightness_hist_5", "brightness_hist_6", "brightness_hist_7",
    # E. Audio embedding (16) — zeros when absent
    "audio_emb_0", "audio_emb_1", "audio_emb_2", "audio_emb_3",
    "audio_emb_4", "audio_emb_5", "audio_emb_6", "audio_emb_7",
    "audio_emb_8", "audio_emb_9", "audio_emb_10", "audio_emb_11",
    "audio_emb_12", "audio_emb_13", "audio_emb_14", "audio_emb_15",
    # F. Pool / reserved (4)
    "pool_mean", "pool_var", "pool_min", "pool_max",
]

assert len(FEATURE_NAMES) == DIM

# Frozen normalisation stats (placeholder means/stds — replace with dataset stats later)
NORM_STATS = {
    name: (0.0, 1.0) for name in FEATURE_NAMES
}

INSTRUMENT_SLOTS = {
    "pad": 0,
    "bell": 1,
    "bass": 2,
    "rhodes": 3,
    "ocean-bass": 4,
    "ocean": 4,
    "glass-rhodes": 5,
    "glass": 5,
}


@dataclass
class EventStub:
    midiNote: int = 60
    velocity: float = 0.7
    duration: float = 1.0
    brightness: float = 0.5
    instrument: str = "pad"
    pan: float = 0.0
    startTime: float = 0.0
    pressure: Optional[float] = None


@dataclass
class ContextStub:
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
    # history helpers (filled by adapter)
    last_note: Optional[int] = None
    brightness_history: List[float] = field(default_factory=lambda: [0.5] * 8)
    ma_velocity: float = 0.5
    ma_brightness: float = 0.5
    ma_abs_pitch: float = 60.0
    novelty: float = 0.0
    chord_density: float = 0.3
    short_energy: float = 0.4


def _one_hot(index: int, size: int) -> List[float]:
    v = [0.0] * size
    if 0 <= index < size:
        v[index] = 1.0
    return v


def _register_band(midi: int) -> List[float]:
    if midi < 48:
        return [1.0, 0.0, 0.0]
    if midi < 72:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _pitch_class(midi: int) -> List[float]:
    pc = midi % 12
    angle = 2.0 * math.pi * pc / 12.0
    return [math.sin(angle), math.cos(angle)]


def build_features(ev: EventStub, ctx: ContextStub) -> np.ndarray:
    """Build the frozen 64-dim feature vector.

    All values are clipped to roughly [-1, 1] or [0, 1].
    """
    # A. Event (16)
    midi_norm = ev.midiNote / 127.0
    dur_log = math.log1p(max(ev.duration, 0.0)) / math.log1p(16.0)  # assume max ~16s
    reg = _register_band(ev.midiNote)
    pc = _pitch_class(ev.midiNote)

    inst = ev.instrument.lower().strip()
    inst_idx = INSTRUMENT_SLOTS.get(inst, -1)
    inst_oh = _one_hot(inst_idx, 6)  # 6 known + unknown falls to all-zero

    A = [
        midi_norm,
        float(np.clip(ev.velocity, 0.0, 1.0)),
        float(np.clip(dur_log, 0.0, 1.0)),
        float(np.clip(ev.brightness, 0.0, 1.0)),
        *reg,
        *pc,
        *inst_oh,
        float(np.clip(ev.pan, -1.0, 1.0)),
    ]
    assert len(A) == 16

    # B. Local context (12)
    last = ctx.last_note if ctx.last_note is not None else ev.midiNote
    interval = (ev.midiNote - last) / 24.0
    scale_angle = 2.0 * math.pi * (ev.midiNote % 12) / 12.0  # simple proxy

    B = [
        float(np.clip(ctx.chord_density, 0.0, 1.0)),
        float(np.clip(interval, -1.0, 1.0)),
        math.sin(scale_angle),
        math.cos(scale_angle),
        float(np.clip(ctx.tempo / 200.0, 0.0, 1.0)),
        float(np.clip(ctx.bar_phase, 0.0, 1.0)),
        float(np.clip(ctx.rhythmic_density, 0.0, 1.0)),
        float(np.clip(ctx.short_energy, 0.0, 1.0)),
        float(np.clip(ctx.novelty, 0.0, 1.0)),
        float(np.clip(ctx.ma_velocity, 0.0, 1.0)),
        float(np.clip(ctx.ma_brightness, 0.0, 1.0)),
        float(np.clip(ctx.ma_abs_pitch / 127.0, 0.0, 1.0)),
    ]
    assert len(B) == 12

    # C. Global / section (8)
    key_angle = 2.0 * math.pi * (ctx.key_root % 12) / 12.0
    majorness = 1.0 if ctx.mode.lower() in ("major", "ionian", "lydian", "mixolydian") else 0.0
    minorness = 1.0 if ctx.mode.lower() in ("minor", "aeolian", "dorian", "phrygian") else 0.0
    # simple atmosphere embedding
    atm = ctx.atmosphere.lower()
    atm0 = 1.0 if atm in ("drift", "ambient", "pad") else 0.0
    atm1 = 1.0 if atm in ("pulse", "rhythmic", "drive") else 0.0

    C = [
        float(np.clip(ctx.section_progress, 0.0, 1.0)),
        float(np.clip(ctx.section_energy, 0.0, 1.0)),
        math.sin(key_angle),
        math.cos(key_angle),
        majorness,
        minorness,
        atm0,
        atm1,
    ]
    assert len(C) == 8

    # D. Temporal dynamics (8)
    hist = list(ctx.brightness_history)[:8]
    while len(hist) < 8:
        hist.append(0.5)
    D = [float(np.clip(h, 0.0, 1.0)) for h in hist]
    assert len(D) == 8

    # E. Audio embedding (16) — zeros when absent
    if ctx.ref_audio is not None and len(ctx.ref_audio) > 0:
        # Very cheap placeholder embedding: mean/std + a few spectral proxies
        # Real implementation should use a small trained encoder.
        audio = np.asarray(ctx.ref_audio, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-8))
        peak = float(np.max(np.abs(audio)) + 1e-8)
        # crude spectral centroid proxy via zero-crossing rate etc.
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))
        E = [rms, peak, zcr] + [0.0] * 13
        E = [float(np.clip(x, -1.0, 1.0)) for x in E[:16]]
    else:
        E = [0.0] * 16
    assert len(E) == 16

    # F. Pool stats (4)
    core = A + B + C + D
    pool_mean = float(np.mean(core))
    pool_var = float(np.var(core))
    pool_min = float(np.min(core))
    pool_max = float(np.max(core))
    F = [pool_mean, pool_var, pool_min, pool_max]
    assert len(F) == 4

    vec = np.asarray(A + B + C + D + E + F, dtype=np.float64)
    assert vec.shape == (DIM,)
    # Final safety clip
    vec = np.clip(vec, -1.0, 1.0)
    return vec
