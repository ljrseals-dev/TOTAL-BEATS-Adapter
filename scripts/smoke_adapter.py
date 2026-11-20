#!/usr/bin/env python3
"""Quick adapter smoke test using explicitly enabled, non-neural demo defaults."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sogitech_adapter.ranges import ParamRanges
from sogitech_adapter.events import MusicalEvent, GlobalContext
from sogitech_adapter.adapter import SogitechAdapter


def main():
    root = Path(__file__).resolve().parents[1]
    ranges = ParamRanges(root / "ranges.json")
    adapter = SogitechAdapter(
        ranges=ranges,
        controller=None,
        allow_safe_defaults=True,
    )

    event = MusicalEvent(
        midiNote=60,
        velocity=0.75,
        duration=2.0,
        brightness=0.65,
        instrument="pad",
        pan=-0.2,
        startTime=0.0,
    )
    context = GlobalContext(
        key_root=0,
        mode="minor",
        tempo=88.0,
        bar_phase=0.3,
        section_progress=0.4,
        section_energy=0.55,
        atmosphere="drift",
        rhythmic_density=0.25,
    )

    frame = adapter.on_event(event, context)
    assert len(frame) == 18
    print("Contract:", ranges.contract)
    print("Frame fields:", len(frame))
    print("midiNote     :", frame["midiNote"])
    print("velocity     :", frame["velocity"])
    print("additiveBlend:", round(frame["additiveBlend"], 3))
    print("powerSagDepth:", round(frame["powerSagDepth"], 3))
    print("filterLP     :", round(frame["filterLP"], 3))
    print("glideTime    :", round(frame["glideTime"], 3))
    print("OK — demo defaults produced a legal 18-field frame (not neural inference).")


if __name__ == "__main__":
    main()
