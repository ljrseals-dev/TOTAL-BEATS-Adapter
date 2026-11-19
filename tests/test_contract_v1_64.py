# tests/test_contract_v1_64.py
import numpy as np
import pytest
from sogitech_adapter.features import (
    build_features, CONTRACT_VERSION, FEATURE_NAMES, NORM_STATS,
    EventStub, ContextStub, DIM,
)


@pytest.fixture
def ev():
    return EventStub(midiNote=60, velocity=0.7, duration=1.5, brightness=0.6,
                     instrument="pad", pan=0.0, startTime=0.0)


@pytest.fixture
def ctx():
    return ContextStub(key_root=0, mode="minor", tempo=90.0, bar_phase=0.25,
                       section_index=1, section_progress=0.4, section_energy=0.5,
                       atmosphere="drift", rhythmic_density=0.3, ref_audio=None)


def test_version_is_pinned():
    assert CONTRACT_VERSION == "v1-64"


def test_dimension_is_exactly_64(ev, ctx):
    features = build_features(ev, ctx)
    assert len(features) == DIM
    assert np.asarray(features).shape == (DIM,)


def test_no_nan_or_inf(ev, ctx):
    features = np.asarray(build_features(ev, ctx), dtype=np.float64)
    assert np.all(np.isfinite(features))


def test_all_values_bounded(ev, ctx):
    features = np.asarray(build_features(ev, ctx), dtype=np.float64)
    assert features.min() >= -1.0 - 1e-9
    assert features.max() <= 1.0 + 1e-9


def test_feature_names_are_stable():
    assert len(FEATURE_NAMES) == DIM
    assert FEATURE_NAMES[0] == "midi_note_norm"
    assert FEATURE_NAMES[15] == "pan"
    assert FEATURE_NAMES[43] == "brightness_hist_7"
    assert FEATURE_NAMES[44] == "audio_emb_0"
    assert FEATURE_NAMES[60] == "pool_mean"


def test_normalization_stats_are_explicit_identity_placeholders():
    # These are safe defaults only; fitted corpus statistics are not yet available.
    assert set(NORM_STATS) == set(FEATURE_NAMES)
    for mean, std in NORM_STATS.values():
        assert mean == 0.0
        assert std == 1.0


def test_zero_audio_embedding_when_absent(ev, ctx):
    features = np.asarray(build_features(ev, ctx))
    assert np.allclose(features[44:60], 0.0)


def test_audio_embedding_stub_is_nonzero_when_reference_audio_present(ev, ctx):
    ctx.ref_audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    features = np.asarray(build_features(ev, ctx))
    assert not np.allclose(features[44:60], 0.0)


def test_instrument_onehot_is_exclusive(ev, ctx):
    features = np.asarray(build_features(ev, ctx))
    block = features[9:15]
    assert block.sum() in (0.0, 1.0)
    assert block[0] == 1.0


def test_unknown_instrument_does_not_crash(ev, ctx):
    ev.instrument = "theremin-2049"
    features = np.asarray(build_features(ev, ctx))
    assert np.all(np.isfinite(features))
    assert np.allclose(features[9:15], 0.0)


def test_pitch_class_wraps(ev, ctx):
    ev.midiNote = 60
    c4 = np.asarray(build_features(ev, ctx))
    ev.midiNote = 72
    c5 = np.asarray(build_features(ev, ctx))
    assert np.allclose(c4[7:9], c5[7:9], atol=1e-6)
    assert not np.isclose(c4[0], c5[0])


def test_monotone_in_velocity(ev, ctx):
    ev.velocity = 0.1
    low = build_features(ev, ctx)[1]
    ev.velocity = 0.9
    high = build_features(ev, ctx)[1]
    assert high > low
