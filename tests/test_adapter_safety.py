import json
from pathlib import Path

import numpy as np
import pytest

from sogitech_adapter.adapter import SogitechAdapter
from sogitech_adapter.events import GlobalContext, MusicalEvent
from sogitech_adapter.features import build_features, EventStub, ContextStub
from sogitech_adapter.guards import apply_all_guards
from sogitech_adapter.ranges import ParamRanges, RangeError
from sogitech_adapter.timing import FramePolicy, FrameSmoother
from sogitech_adapter.validation import validate_event, validate_frame


ROOT = Path(__file__).resolve().parents[1]
RANGES = ROOT / "ranges.json"


def test_ranges_require_exact_verified_schema(tmp_path):
    raw = json.loads(RANGES.read_text(encoding="utf-8"))
    raw["unexpected"] = {"min": 0, "max": 1, "scale": "lin", "status": "verified"}
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RangeError, match="parameter keys mismatch"):
        ParamRanges(path)


def test_ranges_reject_nonfinite_or_inverted_bounds(tmp_path):
    raw = json.loads(RANGES.read_text(encoding="utf-8"))
    raw["filterLP"]["max"] = float("nan")
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RangeError, match="finite numbers"):
        ParamRanges(path)


def test_ranges_require_pinned_upstream_commit(tmp_path):
    raw = json.loads(RANGES.read_text(encoding="utf-8"))
    raw["_meta"]["source_commit"] = "main"
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RangeError, match="pinned 40-character"):
        ParamRanges(path)


def test_normalization_round_trip_and_enum():
    ranges = ParamRanges(RANGES)
    for name in ("velocity", "filterLP", "glideTime", "powerSagDepth"):
        for value in (ranges.fields[name]["min"], ranges.fields[name]["max"]):
            assert ranges.from_norm(name, ranges.to_norm(name, value)) == pytest.approx(value)
    assert ranges.clamp("osc1Wave", 1.6) == 2.0
    assert ranges.from_norm("osc1Wave", 0.5) == 2.0


def test_ranges_reject_nonfinite_values():
    ranges = ParamRanges(RANGES)
    with pytest.raises(ValueError, match="finite"):
        ranges.clamp("filterLP", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        ranges.from_norm("filterLP", float("inf"))


def test_production_adapter_requires_controller():
    with pytest.raises(ValueError, match="trained controller is required"):
        SogitechAdapter(ParamRanges(RANGES))


def test_demo_adapter_emits_exact_18_field_frame():
    adapter = SogitechAdapter(ParamRanges(RANGES), allow_safe_defaults=True)
    event = MusicalEvent(startTime=0.0)
    frame = adapter.on_event(event, GlobalContext())
    assert len(frame) == 18
    assert not validate_frame(frame, adapter.ranges)


def test_adapter_updates_rolling_context_statistics():
    adapter = SogitechAdapter(ParamRanges(RANGES), allow_safe_defaults=True)
    context = GlobalContext()
    adapter.on_event(MusicalEvent(velocity=0.7, brightness=0.4, midiNote=60), context)
    assert context.ma_velocity == pytest.approx(0.7)
    assert context.ma_brightness == pytest.approx(0.4)
    assert context.ma_abs_pitch == pytest.approx(60.0)
    adapter.on_event(MusicalEvent(velocity=0.9, brightness=0.8, midiNote=72, startTime=0.1), context)
    assert context.ma_velocity == pytest.approx(0.8)
    assert context.ma_brightness == pytest.approx(0.6)
    assert context.ma_abs_pitch == pytest.approx(66.0)


def test_guards_enforce_ceilings_filter_relationship_and_minimum_level():
    ranges = ParamRanges(RANGES)
    frame = {
        "midiNote": 60.0, "velocity": 0.8, "pressure": 0.0,
        "osc1Wave": 1.0, "osc2Wave": 2.0,
        "osc1Level": 0.0, "osc2Level": 0.0, "subLevel": 0.0, "noiseLevel": 0.0,
        "additiveBlend": 0.95, "powerSagDepth": 1.0,
        "filterLP": 0.4, "filterHP": 0.9, "filterRes": 0.95,
        "keyTrack": 0.5, "velMod": 0.5, "glideTime": 0.1, "residualGain": 1.0,
    }
    guarded = apply_all_guards(frame, ranges)
    assert guarded["powerSagDepth"] <= 0.6
    assert guarded["residualGain"] <= 0.3
    assert guarded["filterHP"] <= 0.5 * guarded["filterLP"]
    assert guarded["filterRes"] <= 0.6
    assert guarded["osc1Level"] == pytest.approx(0.05)
    assert not validate_frame(guarded, ranges)


def test_frame_validation_rejects_fractional_waveform_enum():
    ranges = ParamRanges(RANGES)
    frame = {
        "midiNote": 60.0, "velocity": 0.8, "pressure": 0.0,
        "osc1Wave": 1.5, "osc2Wave": 2.0,
        "osc1Level": 0.7, "osc2Level": 0.2, "subLevel": 0.1, "noiseLevel": 0.0,
        "additiveBlend": 0.2, "powerSagDepth": 0.2,
        "filterLP": 0.7, "filterHP": 0.1, "filterRes": 0.2,
        "keyTrack": 0.5, "velMod": 0.5, "glideTime": 0.0, "residualGain": 0.0,
    }
    assert any("enum value" in problem for problem in validate_frame(frame, ranges))


def test_event_validation_rejects_nan_and_noninteger_pitch():
    assert validate_event(MusicalEvent(velocity=float("nan")))
    assert validate_event(MusicalEvent(pressure=float("inf")))
    assert validate_event(MusicalEvent(midiNote=60.5))


def test_reference_audio_rejects_nonfinite_samples():
    context = ContextStub(ref_audio=np.array([0.0, float("nan")], dtype=np.float32))
    with pytest.raises(ValueError, match="finite samples"):
        build_features(EventStub(), context)


def test_frame_policy_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="mode"):
        FramePolicy(mode="automatic")
    with pytest.raises(ValueError, match="smooth_tau_ms"):
        FramePolicy(smooth_tau_ms=0.0)


def test_evolving_smoother_preserves_event_fields_and_rate_limits_controls():
    smoother = FrameSmoother(FramePolicy(mode="evolving", smooth_tau_ms=30.0))
    first = {
        "midiNote": 60.0, "velocity": 0.1, "pressure": 0.0,
        "osc1Wave": 0.0, "osc2Wave": 1.0, "filterLP": 0.2,
    }
    second = {
        "midiNote": 72.0, "velocity": 0.9, "pressure": 0.8,
        "osc1Wave": 3.0, "osc2Wave": 2.0, "filterLP": 0.9,
    }
    assert smoother.process(first, dt_ms=10.0) == first
    result = smoother.process(second, dt_ms=10.0)
    assert result["midiNote"] == 72.0
    assert result["velocity"] == 0.9
    assert result["pressure"] == 0.8
    assert result["osc1Wave"] == 3.0
    assert result["filterLP"] < 0.9
    assert result["filterLP"] <= 0.2 + 0.08


def test_adapter_rejects_out_of_order_events():
    adapter = SogitechAdapter(ParamRanges(RANGES), allow_safe_defaults=True)
    adapter.on_event(MusicalEvent(startTime=1.0), GlobalContext())
    with pytest.raises(ValueError, match="non-decreasing"):
        adapter.on_event(MusicalEvent(startTime=0.5), GlobalContext())


def test_adapter_reset_clears_stream_state():
    adapter = SogitechAdapter(ParamRanges(RANGES), allow_safe_defaults=True)
    adapter.on_event(MusicalEvent(midiNote=64, startTime=0.0), GlobalContext())
    adapter.reset()
    assert adapter.last_note is None
    assert adapter.brightness_hist == [0.5] * 8
    assert adapter._last_event_time is None
