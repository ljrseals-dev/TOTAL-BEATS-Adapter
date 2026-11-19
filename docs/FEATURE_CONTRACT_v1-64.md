# Feature Contract: sogitech.features.v1-64

**Version:** v1-64  
**Dimension:** exactly 64  
**Frozen:** yes — changes require a new version tag and projection adapter

---

## Layout

| Block | Indices | Count | Description |
|-------|---------|-------|-------------|
| A     | 0–15    | 16    | Event-level (note, velocity, brightness, instrument one-hot, pan…) |
| B     | 16–27   | 12    | Local context (density, interval, tempo, novelty…) |
| C     | 28–35   | 8     | Global / section (progress, energy, key, mode, atmosphere) |
| D     | 36–43   | 8     | Temporal dynamics (brightness history taps) |
| E     | 44–59   | 16    | Audio embedding (zeros when no reference audio) |
| F     | 60–63   | 4     | Pool statistics (mean, var, min, max of A–D) |

See `FEATURE_NAMES` in `sogitech_adapter/features.py` for the exact ordered list.

---

## Rules

1. All values are expected in roughly `[-1, 1]` or `[0, 1]`.
2. Instrument one-hot is exclusive (or all-zero for unknown).
3. Block E must be exactly zeros when `ref_audio` is absent.
4. Pitch-class features must be octave-invariant.
5. Any change to ordering or semantics requires bumping `CONTRACT_VERSION` and updating the checksum test.
