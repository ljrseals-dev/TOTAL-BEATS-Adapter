"""Input and frame validation helpers."""
from __future__ import annotations

from typing import Dict, List
from .ranges import REQUIRED_18, ParamRanges
from .events import MusicalEvent


def validate_event(ev: MusicalEvent) -> List[str]:
    """Return list of problems (empty = valid)."""
    problems = []
    try:
        ev.validate()
    except ValueError as e:
        problems.append(str(e))
    return problems


def validate_frame(frame: Dict[str, float], ranges: ParamRanges) -> List[str]:
    """Check that a frame contains all required fields and stays inside ranges."""
    problems = []
    missing = [k for k in REQUIRED_18 if k not in frame]
    if missing:
        problems.append(f"missing fields: {missing}")

    for k, v in frame.items():
        if k not in ranges.fields:
            continue
        if not isinstance(v, (int, float)):
            problems.append(f"{k}: non-numeric value {v!r}")
            continue
        if v != v or v == float("inf") or v == float("-inf"):  # NaN / Inf
            problems.append(f"{k}: non-finite value {v}")
            continue
        spec = ranges.fields[k]
        if v < spec["min"] or v > spec["max"]:
            problems.append(f"{k}: {v} outside [{spec['min']}, {spec['max']}]")
    return problems
