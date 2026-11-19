# Cmajor Polyphony Design Notes

**Status:** design only — not yet implemented in the upstream Sogitech Cmajor processors  
**Priority:** Phase 5 (but prerequisite for usable pads / sustained layers)

---

## Why this matters

The current Sogitech Cmajor processors are monophonic. Pads, bells, and layered textures are the primary target use-case for TOTAL BEATS integration. A monophonic decoder will make the integration demo sound broken.

---

## Voice state (must be per-voice)

```cmajor
struct Voice {
    int     active;
    int     note;
    float   velocity;
    float   pressure;
    float   age;            // seconds since note-on
    float   release_t;      // seconds since note-off, -1 if held
    int     held;           // 1 while key down

    // Oscillator state — MUST be per-voice
    float   phase_osc1;
    float   phase_osc2;
    float   phase_sub;
    float   additive_phases[32];

    // PowerSag and filter state — MUST be per-voice
    float   sag_env;
    float   lp_z1, lp_z2;
    float   hp_z1, hp_z2;

    // Glide
    float   glide_from;
    float   glide_t;

    // Per-voice parameter frame snapshot
    float   level_osc1, level_osc2, level_sub, level_noise;
    // ... remaining fields from the 18-param frame
}
```

**Critical:** PowerSag state, dual-pole filter state, and all oscillator phases currently live in globals in the monophonic design. Every one of them must move into `Voice` or polyphony will produce intermodulation and filter-state crosstalk.

---

## Allocator sketch

- Recommended voice count: **16** (CPU headroom for 32-harmonic additive)
- Steal policy:
  1. Prefer releasing voice with lowest level
  2. Else oldest held voice (FIFO)
  3. Never steal the most recently triggered voice
- **5 ms fade-before-reuse** is mandatory to avoid clicks

---

## Controller call placement

**Recommended for v1: per note-on (Option A)**

```
noteOn → adapter.predict(f) → 18-field frame → store in voice
```

- One small MLP call per note — negligible cost
- Frame is static for the voice lifetime unless the adapter pushes updates
- Simple and predictable

**Option B (later):** global control-rate refresh every 5–10 ms for evolving pads. Requires good per-voice smoothing offsets so voices do not move in perfect unison.

---

## Release ownership

Release is owned by the voice, not the adapter. The adapter does not emit a “note-off frame”; it simply stops refreshing. This keeps the adapter timing-agnostic.

---

## Headroom

16 voices × 32-harmonic additive will clip. Scale each voice by `1/sqrt(active_count)` or apply soft-clip with generous headroom (≈ 0.4 for 16 voices).

---

## Prerequisite before polyphony

Check the additive oscillator implementation. If it calls `sin()` per partial per sample, convert to wavetable lookup or a band-limited recursive oscillator **first**. That is a prerequisite, not an optimisation.
