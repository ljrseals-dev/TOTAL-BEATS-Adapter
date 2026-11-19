# Paired Controller Dataset Format

The trainer accepts one NumPy NPZ archive. This is a schema, not a generated dataset. Do not populate it with random labels or outputs from the legacy Sogitech pseudo-label script.

## Required arrays

| Array | Required shape/type | Meaning |
|---|---|---|
| `features` | float32 `(N, 64)` | Feature vectors built by this repository's v1-64 constructor |
| `parameters` | float64/float32 `(N, 18)` | Physical target values in canonical order below |
| `feature_contract` | scalar string | `v1-64` |
| `parameter_contract` | scalar string | `sogitech.params.v1` |
| `parameter_names` | string array `(18,)` | Exact canonical field order below |
| `range_source_commit` | scalar string | Exact source commit from `ranges.json` |
| `provenance` | scalar non-empty string | Dataset origin, license/permission, capture or render process |

## Canonical parameter order

```
midiNote, velocity, pressure, osc1Wave, osc2Wave,
osc1Level, osc2Level, subLevel, noiseLevel, additiveBlend,
powerSagDepth, filterLP, filterHP, filterRes, keyTrack, velMod,
glideTime, residualGain
```

## Sample pairing

Every row is one aligned observation:

1. Capture a real compositional MusicalEvent and the corresponding GlobalContext.
2. Build the feature vector using the exact v1-64 implementation.
3. Record the corresponding physical Sogitech parameter frame that produced or is intended to produce the paired observation.
4. Validate the frame against ranges.json and every musical guard.
5. Store the feature and target in the same row.

The feature row and target row must refer to the same event/time slice. Do not pair arbitrary audio descriptors with unrelated parameter values. Keep train and validation splits separated by source recording, composition, or session to avoid leakage.

## Validation performed by the loader

- Contract IDs, parameter order, and range-source commit must match exactly.
- At least two paired rows, with identical row counts.
- Feature shape must be `(N, 64)`; parameter shape must be `(N, 18)`.
- All samples must be finite; feature values must be within [-1, 1].
- Parameter targets must fit their physical ranges.
- Waveform targets must be integer enum values.
- Power-sag, residual, HP/LP, resonance-coupling, and minimum-level guards must hold.

The training command additionally requires at least ten rows so it can create a non-trivial train/validation split.

## Example archive creation

The following is illustrative packaging code. It does not generate training labels; `features` and `parameters` must come from the aligned, reviewed data collection process described above.

```python
import numpy as np
from sogitech_adapter.ranges import ParamRanges, REQUIRED_18

ranges = ParamRanges("ranges.json")
np.savez_compressed(
    "paired-v1-64.npz",
    features=features.astype(np.float32),       # shape (N, 64)
    parameters=parameters.astype(np.float64),   # shape (N, 18), physical units
    feature_contract=np.asarray("v1-64"),
    parameter_contract=np.asarray("sogitech.params.v1"),
    parameter_names=np.asarray(REQUIRED_18),
    range_source_commit=np.asarray(ranges.source_commit),
    provenance=np.asarray("Describe recording/render provenance and rights here"),
)
```

Never commit private or licensed source audio into this repository without the required rights. Store the dataset outside Git and retain its SHA-256 with the resulting checkpoint.
