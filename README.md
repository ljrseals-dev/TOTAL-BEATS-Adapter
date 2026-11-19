# Sogitech ↔ TOTAL BEATS Adapter

**Bridge between Lewis James Seals Jr.’s TOTAL BEATS / Neural Tide generative engine and the Sogitech hybrid DSP synthesizer.**

This repository contains the integration layer, verified parameter contracts, feature constructor, adapter skeleton, tests, and design notes required to turn TOTAL BEATS compositional events into Sogitech 18-field parameter streams.

---

## Status

| Phase | Description                        | State          |
|-------|------------------------------------|----------------|
| 0     | Verify contracts & ranges          | **Complete**   |
| 1     | Adapter skeleton                   | In progress    |
| 2     | Feature Constructor (v1-64)        | Implemented    |
| 3     | Controller training                | Not started    |
| 4     | Guard layer + timing policy        | Partial        |
| 5     | Cmajor polyphony                   | Design only    |
| 6     | Offline render pipeline            | Not started    |
| 7     | Audio embedding (block E)          | Stub only      |

---

## Quick start

```bash
# from the root of this package
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest

# ranges must be verified (they are)
python -c "from sogitech_adapter.ranges import ParamRanges; r = ParamRanges('ranges.json'); print('OK', r.contract)"

# run the contract tests
pytest tests/test_contract_v1_64.py -v
```

---

## Repository layout

```
ranges.json                          # verified 18-field ranges (Phase 0)
sogitech_adapter/
    ranges.py                        # loader that refuses unverified ranges
    features.py                      # 64-dim Feature Constructor (v1-64)
    adapter.py                       # MusicalEvent → 18-field frame
tests/
    test_contract_v1_64.py           # contract lock-down tests
docs/
    INTEGRATION_SPEC.md              # full architecture & contracts
    FEATURE_CONTRACT_v1-64.md        # frozen 64-dim layout
    POLYPHONY_DESIGN.md              # Cmajor voice allocator design
```

---

## Core design rules

1. **Composition never emits synthesis parameters.** TOTAL BEATS emits `MusicalEvent`s + context.
2. **Synthesis never decides musical structure.** Sogitech only consumes 18-field frames.
3. **Contracts are frozen.** The 64-dim feature vector and the 18-field parameter list are versioned. Changes go through projection adapters, not in-place edits.
4. **Fail loudly.** Unverified ranges or missing controller weights raise hard errors (same discipline as Sogitech refusing random weights).

---

## Relationship to upstream projects

- **Sogitech** — https://github.com/ljrseals-dev/Sogitech  
  Hybrid VA + 32-harmonic additive + PowerSag + dual-pole filter. MIT.

- **TOTAL BEATS** — generative music engine (RULES / NEURAL / HYBRID) by Lewis James Seals Jr.  
  Produces the high-level `MusicalEvent` stream this adapter consumes.

---

## License

MIT (aligned with Sogitech).
