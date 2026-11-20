import json

import pytest

from sogitech_adapter.ranges import ParamRanges
from training.intake import inspect_dataset
from training.dataset import DatasetError
from tests.test_controller_training import _write_dataset


def test_intake_reports_hash_and_keeps_rights_review_pending(tmp_path):
    ranges = ParamRanges("ranges.json")
    dataset = tmp_path / "paired.npz"
    _write_dataset(dataset, ranges)

    report = inspect_dataset(dataset, "ranges.json")

    assert report["structural_validation"] == "passed"
    assert report["sample_count"] == 10
    assert report["source_group_count"] == 2
    assert len(report["dataset_sha256"]) == 64
    assert report["rights_and_provenance_review"]["status"] == "pending_human_review"
    assert report["training_authorized"] is False


def test_intake_rejects_invalid_dataset_without_issuing_report(tmp_path):
    ranges = ParamRanges("ranges.json")
    dataset = tmp_path / "invalid.npz"
    _write_dataset(dataset, ranges, feature_contract="legacy-layout")

    with pytest.raises(DatasetError):
        inspect_dataset(dataset, "ranges.json")
