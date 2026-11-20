"""Dataset intake preflight; validates structure without authorizing training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sogitech_adapter.ranges import ParamRanges
from training.dataset import load_training_dataset


def inspect_dataset(dataset_path: str | Path, ranges_path: str | Path = "ranges.json") -> dict:
    """Return a structural intake report; rights review always remains pending."""
    dataset = Path(dataset_path)
    ranges = ParamRanges(ranges_path)
    features, parameters, groups, provenance = load_training_dataset(dataset, ranges)

    digest = hashlib.sha256()
    with dataset.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "record_type": "total_beats_dataset_intake",
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": digest.hexdigest(),
        "sample_count": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "parameter_shape": list(parameters.shape),
        "source_group_count": int(np.unique(groups).size),
        "feature_contract": "v1-64",
        "parameter_contract": "sogitech.params.v1",
        "range_source_commit": ranges.source_commit,
        "provenance_description": provenance,
        "structural_validation": "passed",
        "rights_and_provenance_review": {
            "status": "pending_human_review",
            "reviewed_by": None,
            "evidence_reference": None,
        },
        "training_authorized": False,
        "note": (
            "Structural validation does not establish ownership, license compatibility, "
            "consent, or permission to train. Complete and independently review the "
            "rights evidence before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Paired v1-64 NPZ dataset")
    parser.add_argument("--ranges", default="ranges.json")
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.data, args.ranges), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
