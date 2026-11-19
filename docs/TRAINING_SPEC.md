# Neural Controller Training Specification

**Implementation state:** controller architecture, strict checkpoint loader, dataset validator, and trainer are implemented. No training corpus or trained weights are present; training is not yet complete.

## Contract compatibility

The adapter controller is trained for:

- Feature input: exactly 64 values, contract `sogitech.features.v1-64`.
- Parameter targets: exactly 18 physical values in the canonical order in `ranges.json`.
- Range provenance: the exact pinned Sogitech commit recorded by `ranges.json`.
- Controller architecture: 64 → 128 → 128, with 16 bounded continuous outputs and two four-class waveform heads.

Do not load the legacy Sogitech controller checkpoint merely because it has 64 input dimensions. The upstream legacy training script constructs a different 64-value vector (eight audio-analysis values followed by zeros), not this repository's v1-64 feature layout. Its checkpoint is intentionally rejected by this loader unless it carries this adapter's exact controller, feature, parameter, and range-source metadata.

## Dataset format

Training consumes a paired NPZ file. See `training/DATASET_FORMAT.md`.

Each row must contain:
- One finite 64-dimensional v1-64 feature vector.
- One finite 18-field physical parameter target in canonical order.
- Explicit feature and parameter contract identifiers.
- The exact range-source commit.
- A non-empty provenance description.

Targets must already satisfy all hard musical guards. The dataset loader rejects mismatched contracts, invalid shapes, non-finite values, out-of-range targets, fractional waveform enums, and guard violations. It does not create pseudo-labels.

## Training objective

- Smooth L1 loss on the sixteen continuous parameters after range-aware normalization.
- Cross-entropy loss for osc1Wave and osc2Wave categorical outputs.
- Fixed seeded train/validation split.
- Validation loss, normalized continuous MAE, and per-oscillator waveform accuracy are stored in the checkpoint.
- The checkpoint records the dataset SHA-256, provenance, contracts, range-source commit, seed, and split sizes.

## Commands

```bash
pip install -e ".[train]"
python -m training.train_controller \
  --data path/to/paired-v1-64.npz \
  --ranges ranges.json \
  --out weights/controller-v1-64.pt
```

Load only a compatible trained checkpoint:

```python
from sogitech_adapter import ParamRanges, load_controller
from sogitech_adapter.adapter import SogitechAdapter

ranges = ParamRanges("ranges.json")
controller = load_controller("weights/controller-v1-64.pt", ranges)
adapter = SogitechAdapter(ranges=ranges, controller=controller)
```

The adapter refuses missing controller weights in production. Fixed defaults remain available only through explicit demo mode and are not trained inference.

## Acceptance gates before production use

1. Dataset provenance and licensing are reviewed.
2. Train/validation separation prevents leakage by source recording or session.
3. Validation metrics beat documented baselines and are reported per parameter group.
4. Held-out audio and event sequences are rendered and auditioned.
5. All predicted frames pass range and musical guard validation.
6. The checkpoint's dataset hash, feature contract, parameter contract, and range-source commit are verified.
7. No model is described as production-ready solely because the training script completed.

No controller weights are checked into this repository. The absence of a suitable v1-64 dataset is an explicit blocker, not a reason to fabricate labels or ship random initialization.
