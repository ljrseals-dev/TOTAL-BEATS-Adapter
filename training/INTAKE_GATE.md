# Dataset Intake Gate

Use the preflight command when a real paired v1-64 NPZ archive is supplied:

```bash
pip install -e ".[dev]"
python -m training.intake --data /secure/path/paired-v1-64.npz --ranges ranges.json
```

The command runs the same strict dataset loader used by the trainer and emits a JSON report containing the archive SHA-256, sample count, feature/parameter shapes, source-group count, contract IDs, and pinned range-source commit. It does not modify the archive or copy it into the repository.

A successful structural report is **not** a training approval. The report always sets:

- `rights_and_provenance_review.status = pending_human_review`
- `training_authorized = false`

Before training, a reviewer must independently establish and record the source of every recording/session, the applicable license or explicit permission, consent where applicable, and the permitted use (including model training and redistribution of derived artifacts). A free-text provenance description, a checksum, or passing schema validation is not evidence of those rights.

Keep private or licensed source audio and dataset archives outside Git unless repository publication rights are explicitly established. Preserve the SHA-256 report with the reviewed intake record. Do not edit the report to claim approval without attaching verifiable evidence and identifying the reviewer.

The preflight requires at least two distinct source groups and the strict v1-64/18-parameter schema. The trainer has the additional minimum of ten samples and performs its own group-held-out split. Neither command creates labels, repairs malformed pairs, or converts the legacy Sogitech feature layout.
