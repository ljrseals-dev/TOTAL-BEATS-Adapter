# Parameter Contract: sogitech.params.v1

**Version:** v1  
**Field count:** exactly 18  
**Source of truth:** ranges.json, pinned to a verified Sogitech source commit.

## Canonical order

```
midiNote, velocity, pressure,
osc1Wave, osc2Wave, osc1Level, osc2Level,
subLevel, noiseLevel, additiveBlend, powerSagDepth,
filterLP, filterHP, filterRes, keyTrack, velMod,
glideTime, residualGain
```

## Ownership

| Field | Owner | Notes |
|---|---|---|
| midiNote | passthrough | From MusicalEvent |
| velocity | passthrough | From MusicalEvent |
| pressure | passthrough | From MusicalEvent / channel pressure |
| osc1Wave | controller | Enum 0–3 |
| osc2Wave | controller | Enum 0–3 |
| osc1Level | controller | |
| osc2Level | controller | |
| subLevel | controller | |
| noiseLevel | controller | |
| additiveBlend | controller | |
| powerSagDepth | controller | Musical ceiling 0.6 |
| filterLP | controller | Normalised 0–1 |
| filterHP | controller | Normalised 0–1 |
| filterRes | controller | |
| keyTrack | controller | |
| velMod | controller | |
| glideTime | rule | Derived from interval + density |
| residualGain | controller | Musical ceiling 0.3 |

## Musical ceilings (guard layer)

- powerSagDepth ≤ 0.6
- residualGain ≤ 0.3
- filterHP ≤ 0.5 × filterLP
- Minimum total oscillator level ≥ 0.01

Waveform selections, note identity, velocity, and pressure are event/discrete values; the timing layer must not interpolate them. Only continuous synthesis controls are rate-limited and smoothed.
