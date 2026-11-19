# TOTAL BEATS Adapter

**Bridge between TOTAL BEATS / Neural Tide and the Sogitech hybrid DSP synthesizer.**

Converts compositional intelligence (musical events + context) into expressive, safe, renderable synthesis parameter streams — and provides the scaffolding for a complete album-production pipeline.

---

## Current status (2026-11-19)

| Phase | Component | State |
|---|---|---|
| 0 | Contract integrity | **Complete** |
| 1 | Production-grade adapter | **Repair in progress** |
| 2 | v1-64 controller + trainer | **Implemented; dataset and weights absent** |
| 3 | Guard + timing system | **Core repairs implemented; further integration pending** |
| 4 | Polyphonic Sogitech integration | Design only |
| 5 | Offline audio rendering | Not started |
| 6 | Learned audio embedding | Stub only |
| 7 | Mixing and mastering | Not started |
| 8 | Complete album production | Spec only |

**Feature contract:** `brightness_hist_7` is index 43; `audio_emb_0` is index 44. The audio block currently contains only simple reference-audio statistics and zeros, not a learned embedding. Normalization statistics remain identity placeholders until a versioned corpus is fitted.

**Parameter source pin:** `ranges.json` is checked against Sogitech commit `1a40ed625743d16e41341a0a3c08d0a2b6f45ff5`, not a moving branch name.

---

## Installation and quick start

Base adapter and test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Verify contracts
python -c "from sogitech_adapter import ParamRanges; r = ParamRanges('ranges.json'); print(r.contract, len(r.fields))"

# Run tests
pytest tests/ -v

# Smoke the adapter with explicitly enabled demo defaults (not neural inference)
python scripts/smoke_adapter.py
```

PyTorch is deliberately optional and is never silently substituted:

- Controller inference: `pip install -e ".[neural]"`
- Controller training: `pip install -e ".[train]"`

Production use must pass a compatible trained controller to `SogitechAdapter`. The loader rejects legacy checkpoints trained on a different 64-value feature layout. Training requires a reviewed paired v1-64 dataset documented in `training/DATASET_FORMAT.md`. No dataset or trained weights are currently included.

Missing PyTorch raises an actionable error at the point where inference or training is requested. See [the import policy](docs/IMPORT_POLICY.md) for the required/optional dependency rules and tests.

---

## Repository layout

```
TOTAL-BEATS-Adapter/
├── ranges.json
├── sogitech_adapter/
│   ├── adapter.py
│   ├── controller.py
│   ├── features.py
│   ├── ranges.py
│   ├── events.py
│   ├── guards.py
│   ├── timing.py
│   └── validation.py
├── training/
│   ├── dataset.py
│   ├── train_controller.py
│   └── DATASET_FORMAT.md
├── docs/
├── tests/
├── scripts/smoke_adapter.py
└── .github/workflows/tests.yml
```

## Core design rules

1. **Composition never emits synthesis parameters.** TOTAL BEATS emits `MusicalEvent` objects and context.
2. **Synthesis never decides musical structure.** Sogitech only consumes 18-field frames.
3. **Contracts are versioned.** The 64-dim feature vector and 18-field parameter frame are frozen interfaces.
4. **Fail loudly.** Unverified ranges, malformed frames, and missing production controller weights raise hard errors.
5. **Demo mode is explicit.** Fixed safe defaults are available only when `allow_safe_defaults=True`; they are not neural predictions.
6. **No incompatible checkpoints.** Controller weights must declare the exact feature, parameter, controller, and range-source contracts.
7. **No fabricated training data.** The trainer accepts only paired feature/parameter datasets with provenance and guard validation, and holds out complete source groups.
8. **No fake imports.** Required imports fail; optional capabilities fail explicitly when requested but unavailable. See [docs/IMPORT_POLICY.md](docs/IMPORT_POLICY.md).

---

## Upstream projects

- **Sogitech** — https://github.com/ljrseals-dev/Sogitech (MIT)
- **TOTAL BEATS** — generative music engine by Lewis James Seals Jr.

## License

MIT
