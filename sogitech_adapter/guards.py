"""Musical safety guards for the 18-field parameter stream."""
from __future__ import annotations

from typing import Dict, Any
from .ranges import ParamRanges


def apply_all_guards(frame: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Run the full guard stack. Order matters."""
    frame = dict(frame)
    frame = ranges.apply_musical_ceilings(frame)
    frame = guard_hp_lp(frame, ranges)
    frame = guard_resonance_coupling(frame, ranges)
    frame = guard_minimum_levels(frame, ranges)
    return frame


def guard_hp_lp(frame: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Enforce filterHP < ratio * filterLP (normalised space)."""
    ratio = ranges.guards.get("hp_lt_lp_ratio", 0.5)
    lp = frame.get("filterLP", 0.7)
    hp = frame.get("filterHP", 0.05)
    if hp > ratio * lp:
        frame["filterHP"] = float(ratio * lp)
    return frame


def guard_resonance_coupling(frame: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Soft-limit resonance when additive blend is high."""
    cfg = ranges.guards.get("res_soft_couple_additive", {})
    trigger = cfg.get("trigger_above", 0.7)
    ceiling = cfg.get("res_ceiling", 0.6)
    if frame.get("additiveBlend", 0.0) > trigger:
        if frame.get("filterRes", 0.0) > ceiling:
            frame["filterRes"] = float(ceiling)
    return frame


def guard_minimum_levels(frame: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Prevent silent voices."""
    total = (
        frame.get("osc1Level", 0.0)
        + frame.get("osc2Level", 0.0)
        + frame.get("subLevel", 0.0)
        + frame.get("noiseLevel", 0.0)
    )
    min_total = ranges.guards.get("min_total_level", 0.01)
    if total < min_total:
        frame["osc1Level"] = float(ranges.guards.get("fallback_osc1_level", 0.05))
    return frame
