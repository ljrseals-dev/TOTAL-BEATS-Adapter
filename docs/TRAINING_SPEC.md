# Training Specification — Neural Parameter Controller

**Status:** Not started  
**Goal:** Learn a stable mapping from 64-dim musical features → 18-dim Sogitech parameters.

## Principles

1. Do not train on arbitrary parameter guesses.
2. Prefer synthetic round-trip data generated from the existing Python reference renderer.
3. Secondary source: analysis–resynthesis pairs from real audio.
4. Instrument conditioning via FiLM or embedding, not hard presets.
5. Refuse to ship random weights.

## Primary data path (synthetic round-trip)

```
sample 18-param vector p   (Gaussian around musical preset clusters)
        │
Python reference renderer
        │
Feature Constructor → 64-dim f
        │
train:  f → p_hat
```

Loss: weighted L1 / Huber on continuous dims + cross-entropy on wave enums.

## Sampling discipline

- Cluster centres = musical presets (pad, bell, bass, rhodes, ocean, glass)
- Reject samples that render below energy floor or above harshness proxy
- Cover register, velocity, duration, and harmonic context variation

## Validation gates

- Round-trip feature distance
- Round-trip spectral distance
- Instrument region separation
- Zero range violations after output layer + guards
- Extreme clamp test (HP/LP, resonance + additive, residual)

## Human loop

Only after synthetic validation passes. Small A/B preference budget.
