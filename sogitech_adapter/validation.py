"""Input and frame validation helpers."""
from __future__ import annotations

import math
from typing import Dict, List
from .ranges import REQUIRED_18, ParamRanges
from .events import MusicalEvent


def validate_event(ev: MusicalEvent) -> List[str]:
    """Return list of problems (empty = valid)."""
    try:
        ev.validate()
    except (AttributeError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def validate_frame(frame: Dict[str, float], ranges: ParamRanges) -> List[str]:
    """Require the exact 18-field schema, finite values, and legal field domains."""
    problems: List[str] = []
    if not isinstance(frame, dict):
        return ["frame must be a dictionary"]
    missing = [key for key in REQUIRED_18 if key not in frame]
    extra = [key for key in frame if key not in ranges.fields]
    if missing:
        problems.append(f"missing fields: {missing}")
    if extra:
        problems.append(f"unknown fields: {extra}")

    for key in REQUIRED_18:
        if key not in frame:
            continue
        value = frame[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            problems.append(f"{key}: value must be a finite number, got {value!r}")
            continue
        spec = ranges.fields[key]
        if value < spec["min"] or value > spec["max"]:
            problems.append(f"{key}: {value} outside [{spec['min']}, {spec['max']}]")
        if spec["scale"] == "enum" and not float(value).is_integer():
            problems.append(f"{key}: enum value must be an integer, got {value}")
    return problems
