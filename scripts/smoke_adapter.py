#!/usr/bin/env python3
"""Quick smoke test of the adapter without a trained controller."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sogitech_adapter.ranges import ParamRanges
from sogitech_adapter.features import EventStub, ContextStub
from sogitech_adapter.adapter import SogitechAdapter

def main():
    ranges = ParamRanges(Path(__file__).resolve().parents[1] / "ranges.json")
    adapter = SogitechAdapter(ranges=ranges, controller=None)

    ev = EventStub(
        midiNote=60,
        velocity=0.75,
        duration=2.0,
        brightness=0.65,
        instrument="pad",
        pan=-0.2,
    )
    ctx = ContextStub(
        key_root=0,
        mode="minor",
        tempo=88.0,
        bar_phase=0.3,
        section_progress=0.4,
        section_energy=0.55,
        atmosphere="drift",
        rhythmic_density=0.25,
    )

    frame = adapter.on_event(ev, ctx)
    print("Contract:", ranges.contract)
    print("Frame keys:", sorted(frame.keys()))
    print("midiNote     :", frame["midiNote"])
    print("velocity     :", frame["velocity"])
    print("additiveBlend:", round(frame["additiveBlend"], 3))
    print("powerSagDepth:", round(frame["powerSagDepth"], 3))
    print("filterLP     :", round(frame["filterLP"], 3))
    print("glideTime    :", round(frame["glideTime"], 3))
    print("OK — adapter produced a legal 18-field frame.")

if __name__ == "__main__":
    main()
