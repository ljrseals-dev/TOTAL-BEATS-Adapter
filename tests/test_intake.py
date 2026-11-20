import numpy as np
import pytest

from sogitech_adapter.ranges import ParamRanges, REQUIRED_18
from training.dataset import DatasetError
from training.intake import inspect_dataset


def _write_dataset(path, ranges, feature_contract="v1-64"):
    rows = 10
    features = np.zeros((rows, 64), dtype=np.float32)
    target = np.asarray([
        60.0, 0.7, 0.0, 1.0, 2.0, 0.7, 0.2, 0.1, 0.01,
        0.2, 0.2, 0.7, 0.1, 0.2, 0.5, 0.5, 0.03, 0.0,
    ], dtype=np.float64)
    parameters = np.tile(target, (rows, 1))
    np.savez(
        path,
        features=features,
        parameters=parameters,
        groups=np.asarray(["recording-a"] * 5 + ["recording-b"] * 5),
        feature_contract=np.asarray(feature_contract),
        parameter_contract=np.asarray("sogitech.params.v1"),
        parameter_names=np.asarray(REQUIRED_18),
        range_source_commit=np.asarray(ranges.source_commit),
        provenance=np.asarray("test fixture only; no rights assertion"),
    )


def test_intake_reports_hash_and_keeps_rights_review_pending(tmp_path):
    ranges = ParamRanges("ranges.json")
    dataset = tmp_path / "paired.npz"
    _write_dataset(dataset, ranges)

    report = inspect_dataset(dataset, "ranges.json")

    assert report["structural_validation"] == "passed"
    assert report["sample_count"] == 10
    assert report["source_group_count"] == 2
    assert len(report["dataset_sha256"]) == 64
    assert report["rights_review"] == "pending_human_review"
    assert report["rights_and_provenance_review"]["status"] == "pending_human_review"
    assert report["training_authorized"] is False


def test_intake_rejects_invalid_dataset_without_issuing_report(tmp_path):
    ranges = ParamRanges("ranges.json")
    dataset = tmp_path / "invalid.npz"
    _write_dataset(dataset, ranges, feature_contract="legacy-layout")

    with pytest.raises(DatasetError):
        inspect_dataset(dataset, "ranges.json")
