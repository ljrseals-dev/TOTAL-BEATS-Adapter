# tests/test_contract_v1_64.py
import math
import numpy as np
import pytest
from sogitech_adapter.features import (
    build_features, CONTRACT_VERSION, FEATURE_NAMES, NORM_STATS,
    EventStub, ContextStub, DIM,
)

# ---------- fixtures ----------
@pytest.fixture
def ev():
    return EventStub(midiNote=60, velocity=0.7, duration=1.5, brightness=0.6,
                     instrument="pad", pan=0.0, startTime=0.0)

@pytest.fixture
def ctx():
    return ContextStub(key_root=0, mode="minor", tempo=90.0, bar_phase=0.25,
                       section_index=1, section_progress=0.4, section_energy=0.5,
                       atmosphere="drift", rhythmic_density=0.3,
                       ref_audio=None)

# ---------- the contract ----------
def test_version_is_pinned():
    assert CONTRACT_VERSION == "v1-64"

def test_dimension_is_exactly_64(ev, ctx):
    f = build_features(ev, ctx)
    assert len(f) == DIM
    assert np.asarray(f).shape == (DIM,)

def test_no_nan_or_inf(ev, ctx):
    f = np.asarray(build_features(ev, ctx), dtype=np.float64)
    assert np.all(np.isfinite(f)), f"non-finite at {np.where(~np.isfinite(f))}"

def test_all_values_bounded(ev, ctx):
    f = np.asarray(build_features(ev, ctx), dtype=np.float64)
    assert f.min() >= -1.0 - 1e-9, f"min {f.min()}"
    assert f.max() <=  1.0 + 1e-9, f"max {f.max()}"

def test_feature_names_are_stable():
    assert len(FEATURE_NAMES) == DIM
    assert FEATURE_NAMES[0]  == "midi_note_norm"
    assert FEATURE_NAMES[15] == "pan"
    assert FEATURE_NAMES[44] == "brightness_hist_7"
    assert FEATURE_NAMES[60] == "pool_mean"

def test_normalization_stats_frozen():
    for name, (mu, sd) in NORM_STATS.items():
        assert sd > 0, f"{name}: zero std"
        assert abs(mu) <= 5.0, f"{name}: implausible mean {mu}"

def test_zero_audio_embedding_when_absent(ev, ctx):
    f = np.asarray(build_features(ev, ctx))
    # E block is indices 44:60
    assert np.allclose(f[44:60], 0.0), "block E must be zeros when ref_audio is None"

def test_audio_embedding_populated_when_present(ev, ctx):
    ctx.ref_audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    f = np.asarray(build_features(ev, ctx))
    assert not np.allclose(f[44:60], 0.0)

def test_instrument_onehot_is_exclusive(ev, ctx):
    f = np.asarray(build_features(ev, ctx))
    block = f[9:15]
    assert block.sum() in (0.0, 1.0), f"instrument block not one-hot: {block}"
    assert block[0] == 1.0, "pad should be slot 0"

def test_unknown_instrument_does_not_crash(ev, ctx):
    ev.instrument = "theremin-2049"
    f = np.asarray(build_features(ev, ctx))
    assert np.all(np.isfinite(f))

def test_pitch_class_wraps(ev, ctx):
    ev.midiNote = 60
    f60 = np.asarray(build_features(ev, ctx))
    ev.midiNote = 72
    f72 = np.asarray(build_features(ev, ctx))
    assert np.allclose(f60[7:9], f72[7:9], atol=1e-6), "C and C+12 must match in pc block"
    assert not np.isclose(f60[0], f72[0]), "register/pitch dim must differ"

def test_monotone_in_velocity(ev, ctx):
    ev.velocity = 0.1
    lo = build_features(ev, ctx)[1]
    ev.velocity = 0.9
    hi = build_features(ev, ctx)[1]
    assert hi > lo
