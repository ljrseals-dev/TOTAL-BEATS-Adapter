# TOTAL BEATS Adapter

**Bridge between TOTAL BEATS / Neural Tide and the Sogitech hybrid DSP synthesizer.**

Converts compositional intelligence (musical events + context) into expressive, safe, renderable synthesis parameter streams — and provides the scaffolding for a complete album-production pipeline.

---

## Current status (2026-11-19)

| Phase | Component                        | State              |
|-------|----------------------------------|--------------------|
| 0     | Contract integrity                 | **Complete**       |
| 1     | Production-grade adapter         | **Substantial**    |
| 2     | Neural controller + training     | Not started        |
| 3     | Guard + timing system            | Partial (core done)|
| 4     | Polyphonic Sogitech integration  | Design only        |
| 5     | Offline audio rendering          | Not started        |
| 6     | Learned audio embedding          | Stub only          |
| 7     | Mixing and mastering             | Not started        |
| 8     | Complete album production        | Spec only          |

**Contract defect fixed:** `brightness_hist_7` is correctly at index 43; `audio_emb_0` at index 44.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pip install -e .

# Verify contracts
python -c "from sogitech_adapter import ParamRanges; r = ParamRanges('ranges.json'); print(r.contract, len(r.fields))"

# Run tests
pytest tests/ -v

# Smoke the adapter
python scripts/smoke_adapter.py
```

---

## Repository layout

```
TOTAL-BEATS-Adapter/
├── ranges.json                     # verified 18-field ranges
├── sogitech_adapter/
│   ├── adapter.py                  # MusicalEvent → 18-field frame
│   ├── features.py                 # 64-dim Feature Constructor (v1-64)
│   ├── ranges.py                   # loader that refuses unverified ranges
│   ├── events.py                   # MusicalEvent + GlobalContext
│   ├── guards.py                   # musical safety guards
│   ├── timing.py                   # FramePolicy + smoother
│   └── validation.py               # event and frame validators
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FEATURE_CONTRACT_v1-64.md
│   ├── PARAMETER_CONTRACT_v1.md
│   ├── INTEGRATION_SPEC.md
│   ├── POLYPHONY_DESIGN.md
│   ├── TRAINING_SPEC.md
│   ├── RENDERING_SPEC.md
│   └── ALBUM_PRODUCTION.md
├── tests/
│   └── test_contract_v1_64.py
├── scripts/
│   └── smoke_adapter.py
├── training/                       # (scaffolding)
├── audio/                          # (scaffolding)
├── embeddings/                     # (scaffolding)
├── album/                          # (scaffolding)
└── .github/workflows/tests.yml
```

---

## Core design rules

1. **Composition never emits synthesis parameters.** TOTAL BEATS emits `MusicalEvent`s + context.
2. **Synthesis never decides musical structure.** Sogitech only consumes 18-field frames.
3. **Contracts are frozen.** 64-dim features and 18-field parameters are versioned.
4. **Fail loudly.** Unverified ranges or missing controller weights raise hard errors.
5. **Do not invent trained weights.** The controller refuses random initialisation.

---

## Upstream projects

- **Sogitech** — https://github.com/ljrseals-dev/Sogitech (MIT)
- **TOTAL BEATS** — generative music engine by Lewis James Seals Jr.

---

## License

MIT
