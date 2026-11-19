import numpy as np
import pytest

from sogitech_adapter.controller import load_controller
from sogitech_adapter.ranges import ParamRanges, REQUIRED_18
from training.dataset import DatasetError, load_training_dataset
from training.train_controller import _group_split


RANGES_PATH = "ranges.json"


def _write_dataset(path, ranges, *, feature_contract="v1-64", violate_guard=False):
    rows = 10
    features = np.zeros((rows, 64), dtype=np.float32)
    parameters = np.tile(np.array([
        60.0, 0.7, 0.0, 1.0, 2.0, 0.7, 0.2, 0.1, 0.01,
        0.2, 0.2, 0.7, 0.1, 0.2, 0.5, 0.5, 0.03, 0.0,
    ], dtype=np.float64), (rows, 1))
    if violate_guard:
        parameters[:, REQUIRED_18.index("filterHP")] = 0.6
    np.savez(
        path,
        features=features,
        parameters=parameters,
        groups=np.asarray(["recording-a"] * 5 + ["recording-b"] * 5),
        feature_contract=np.asarray(feature_contract),
        parameter_contract=np.asarray("sogitech.params.v1"),
        parameter_names=np.asarray(REQUIRED_18),
        range_source_commit=np.asarray(ranges.source_commit),
        provenance=np.asarray("unit-test paired feature/parameter fixture"),
    )


def test_dataset_loader_accepts_versioned_paired_samples(tmp_path):
    ranges = ParamRanges(RANGES_PATH)
    path = tmp_path / "paired.npz"
    _write_dataset(path, ranges)
    features, parameters, groups, provenance = load_training_dataset(path, ranges)
    assert features.shape == (10, 64)
    assert parameters.shape == (10, 18)
    assert groups.shape == (10,)
    assert provenance.startswith("unit-test")


def test_dataset_loader_rejects_wrong_feature_contract(tmp_path):
    ranges = ParamRanges(RANGES_PATH)
    path = tmp_path / "wrong-contract.npz"
    _write_dataset(path, ranges, feature_contract="legacy-8-plus-zeros")
    with pytest.raises(DatasetError, match="feature contract"):
        load_training_dataset(path, ranges)


def test_dataset_loader_rejects_targets_that_bypass_guards(tmp_path):
    ranges = ParamRanges(RANGES_PATH)
    path = tmp_path / "unsafe-targets.npz"
    _write_dataset(path, ranges, violate_guard=True)
    with pytest.raises(DatasetError, match="filterHP targets violate"):
        load_training_dataset(path, ranges)


def test_training_validation_split_keeps_source_groups_disjoint():
    groups = np.asarray(["recording-a"] * 5 + ["recording-b"] * 5)
    training, validation = _group_split(groups, 0.2, 17)
    assert set(groups[training]).isdisjoint(set(groups[validation]))
    assert len(training) > 0
    assert len(validation) > 0


def test_group_split_rejects_single_source_group():
    with pytest.raises(ValueError, match="two source groups"):
        _group_split(np.asarray(["only-one"] * 10), 0.2, 17)


def test_controller_loader_does_not_fabricate_missing_weights(tmp_path):
    ranges = ParamRanges(RANGES_PATH)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_controller(tmp_path / "missing-controller.pt", ranges)
