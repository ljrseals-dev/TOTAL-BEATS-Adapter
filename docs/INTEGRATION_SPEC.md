# TOTAL BEATS ↔ SOGITECH — Integration Specification

**Status:** draft spec, v0.1 (Phase 0 verified)  
**Scope:** adapter, feature contract, controller usage, training, parameter ranges, build order

---

## 0. What is verified vs. assumed

**Verified from Sogitech repo (2026-11-19):**

- 18-field control stream names and ranges (see `ranges.json`)
- 64-dim feature vector input to MLP controller
- Cmajor decoder: additive + VA + PowerSag + dual-pole filter, currently **monophonic**
- Offline Python reference renderer exists
- Codec quantizes to JSON sidecar; controller refuses random weights
- MIT license
- Wave parameters are continuous 0–3 (sine / saw / square / triangle)
- `filterLP` / `filterHP` are normalised 0–1 inside the parameter vector

---

## 1. Architecture

```text
TOTAL BEATS / Neural Tide
   │  MusicalEvent stream + global context
   ▼
[1] Adapter (this package)
   │  • pass-through fields (midiNote, velocity, pressure)
   │  • Feature Constructor → 64-dim f
   │  • controller.predict(f) → timbre params
   │  • rule/guard layer (clamp, smooth, musical ceilings)
   ▼
[2] 18-field parameter frames @ control rate
   ▼
[3] Sogitech Cmajor decoder (or Python reference renderer for offline)
   ▼
   Audio (+ pan applied post-decoder)
```

**Three invariants:**

1. Composition never emits synthesis parameters. It emits events + context.
2. Synthesis never decides musical structure. It consumes frames.
3. The 64-dim vector and the 18-field frame are versioned, frozen contracts.

---

## 2. Contract A — MusicalEvent

```
midiNote      int   0..127
velocity      float 0..1
duration      float seconds
brightness    float 0..1
instrument    enum  pad | bell | bass | rhodes | ocean-bass | glass-rhodes | unknown
pan           float -1..1
startTime     float seconds
pressure      float 0..1 (optional)
```

Plus a separate global context object.

---

## 3. Contract B — 18-field control stream

See `ranges.json` for authoritative verified ranges.

**Ownership split:**

| Field          | Owner        | Notes |
|----------------|--------------|-------|
| midiNote       | passthrough  | From event |
| velocity       | passthrough  | From event |
| pressure       | passthrough  | From event / CC |
| osc1Wave..residualGain | controller / rule | See ranges.json |

---

## 4. Feature Constructor — 64 dimensions

Frozen as `sogitech.features.v1-64`. See `sogitech_adapter/features.py` and `FEATURE_NAMES`.

Blocks:

- **A (16)** Event-level
- **B (12)** Local context
- **C (8)** Global / section
- **D (8)** Temporal dynamics (brightness history)
- **E (16)** Audio embedding (zeros when absent)
- **F (4)** Pool stats

---

## 5. Build Order

```
Phase 0  Verify contracts          ← DONE (ranges.json verified)
Phase 1  Adapter skeleton          ← in progress
Phase 2  Feature Constructor       ← implemented
Phase 3  Controller training
Phase 4  Guard layer + timing
Phase 5  Polyphony (Cmajor)
Phase 6  Offline pipeline
Phase 7  Optional audio embedding
```

---

## 6. Open questions still useful to confirm

1. Exact control-frame rate supported by the Cmajor processors.
2. Whether pressure is sourced from channel pressure, CC, or synthesised.
3. Bit-exactness expectations between Python reference renderer and Cmajor path.
