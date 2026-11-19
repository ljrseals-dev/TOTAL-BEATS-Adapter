# TOTAL BEATS Adapter

**Bridge between TOTAL BEATS / Neural Tide and the Sogitech hybrid DSP synthesizer.**

Converts compositional intelligence (musical events + context) into expressive, safe, renderable synthesis parameter streams — and provides the scaffolding for a complete album-production pipeline.

---

## Current status (2026-11-19)

| Phase | Component                        | State              |
|-------|----------------------------------|--------------------|
| 0     | Contract integrity               | **Complete**       |
| 1     | Production-grade adapter         | **Repair in progress** |
| 2     | Neural controller + training     | Not started        |
| 3     | Guard + timing system            | **Core repairs in progress** |
| 4     | Polyphonic Sogitech integration  | Design only        |
| 5     | Offline audio rendering          | Not started        |
| 6     | Learned audio embedding          | Stub only          |
| 7     | Mixing and mastering             | Not started        |
| 8     | Complete album production        | Spec only          |

**Feature contract:** `brightness_hist_7` is index 43; `audio_emb_0` is index 44.

**Parameter source pin:** `ranges.json` is checked against Sogitech commit `1a40ed625743d16e41341a0a3c08d0a2b6f45ff5`, not a moving branch name.

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

# Smoke the adapter with explicitly enabled demo defaults (not neural inference)
python scripts/smoke_adapter.py
```

Production use must pass a trained controller to `SogitechAdapter`. The smoke script opts into fixed demo defaults explicitly; missing trained weights are never silently treated as a trained model.

---

## Repository layout

```
TOTAL-BEATS-Adapter/
├── ranges.json
├── sogitech_adapter/
│   ├── adapter.py
│   ├── features.py
│   ├── ranges.py
│   ├── events.py
│   ├── guards.py
│   ├── timing.py
│   └── validation.py
├── docs/
├── tests/
│   ├── test_contract_v1_64.py
│   └── test_adapter_safety.py
├── scripts/smoke_adapter.py
└── .github/workflows/tests.yml
```

---

## Core design rules

1. **Composition never emits synthesis parameters.** TOTAL BEATS emits `MusicalEvent` objects and context.
2. **Synthesis never decides musical structure.** Sogitech only consumes 18-field frames.
3. **Contracts are versioned.** The 64-dim feature vector and 18-field parameter frame are frozen interfaces.
4. **Fail loudly.** Unverified or malformed ranges and missing production controller weights raise hard errors.
5. **Demo mode is explicit.** Fixed safe defaults are available only when `allow_safe_defaults=True`; they are not neural predictions.
6. **Do not invent trained weights.** The adapter does not initialise or substitute a random controller.

---

## Upstream projects

- **Sogitech** — https://github.com/ljrseals-dev/Sogitech (MIT)
- **TOTAL BEATS** — generative music engine by Lewis James Seals Jr.

## License

MIT
