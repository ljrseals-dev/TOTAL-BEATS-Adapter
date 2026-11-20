"""Validated paired-feature/parameter dataset reader for controller training."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from sogitech_adapter.controller import ENUM_NAMES, FEATURE_CONTRACT, PARAMETER_CONTRACT
from sogitech_adapter.ranges import ParamRanges, REQUIRED_18


class DatasetError(ValueError):
    """Raised when a training dataset does not match the frozen contracts."""


def load_training_dataset(path: str | Path, ranges: ParamRanges):
    """Load paired v1-64 features, physical targets, and source-group labels.

    Required arrays: features, parameters, groups, feature_contract,
    parameter_contract, parameter_names, range_source_commit, provenance.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"training dataset not found: {dataset_path}")
    try:
        archive = np.load(dataset_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"unable to open NPZ dataset: {exc}") from exc

    with archive:
        required = {
            "features", "parameters", "groups", "feature_contract", "parameter_contract",
            "parameter_names", "range_source_commit", "provenance",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise DatasetError(f"dataset is missing arrays: {missing}")
        try:
            feature_contract = str(np.asarray(archive["feature_contract"]).item())
            parameter_contract = str(np.asarray(archive["parameter_contract"]).item())
            source_commit = str(np.asarray(archive["range_source_commit"]).item())
            provenance = str(np.asarray(archive["provenance"]).item()).strip()
            parameter_names = [str(item) for item in np.asarray(archive["parameter_names"]).tolist()]
            features = np.asarray(archive["features"], dtype=np.float32)
            parameters = np.asarray(archive["parameters"], dtype=np.float64)
            groups = np.asarray(archive["groups"]).astype(str)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"invalid dataset metadata or array dtype: {exc}") from exc

    if feature_contract != FEATURE_CONTRACT:
        raise DatasetError(f"expected feature contract {FEATURE_CONTRACT!r}, got {feature_contract!r}")
    if parameter_contract != PARAMETER_CONTRACT:
        raise DatasetError(f"expected parameter contract {PARAMETER_CONTRACT!r}, got {parameter_contract!r}")
    if parameter_names != REQUIRED_18:
        raise DatasetError("parameter_names must match the canonical 18-field contract order")
    if source_commit != ranges.source_commit:
        raise DatasetError("dataset range_source_commit does not match the loaded ranges.json")
    if not provenance:
        raise DatasetError("dataset provenance must be a non-empty description")

    if features.ndim != 2 or features.shape[1] != 64:
        raise DatasetError(f"features must have shape (N, 64), got {features.shape}")
    if parameters.ndim != 2 or parameters.shape[1] != len(REQUIRED_18):
        raise DatasetError(f"parameters must have shape (N, 18), got {parameters.shape}")
    if features.shape[0] != parameters.shape[0] or features.shape[0] < 2:
        raise DatasetError("features and parameters must have equal row counts and at least two rows")
    if groups.shape != (features.shape[0],) or np.any(np.char.str_len(groups) == 0):
        raise DatasetError("groups must contain one non-empty source recording/session ID per row")
    if np.unique(groups).size < 2:
        raise DatasetError("at least two distinct source groups are required for leakage-safe validation")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(parameters)):
        raise DatasetError("features and parameter targets must contain only finite values")
    if np.any(features < -1.0) or np.any(features > 1.0):
        raise DatasetError("feature values must remain within the v1-64 range [-1, 1]")

    indices = {name: index for index, name in enumerate(REQUIRED_18)}
    for name in REQUIRED_18:
        values = parameters[:, indices[name]]
        spec = ranges.fields[name]
        if np.any(values < float(spec["min"])) or np.any(values > float(spec["max"])):
            raise DatasetError(f"{name} targets fall outside the verified physical range")
        if name in ENUM_NAMES and np.any(values != np.floor(values)):
            raise DatasetError(f"{name} targets must be integer enum indices")

    guards = ranges.guards
    if np.any(parameters[:, indices["powerSagDepth"]] > guards["power_sag_musical_ceiling"]):
        raise DatasetError("powerSagDepth targets exceed the musical guard ceiling")
    if np.any(parameters[:, indices["residualGain"]] > guards["residual_musical_ceiling"]):
        raise DatasetError("residualGain targets exceed the musical guard ceiling")
    hp_limit = guards["hp_lt_lp_ratio"] * parameters[:, indices["filterLP"]]
    if np.any(parameters[:, indices["filterHP"]] > hp_limit):
        raise DatasetError("filterHP targets violate the filterHP/filterLP guard")
    resonance_guard = guards["res_soft_couple_additive"]
    coupled = parameters[:, indices["additiveBlend"]] > resonance_guard["trigger_above"]
    if np.any(parameters[coupled, indices["filterRes"]] > resonance_guard["res_ceiling"]):
        raise DatasetError("filterRes targets violate additive/resonance coupling")
    levels = sum(parameters[:, indices[name]] for name in
                 ("osc1Level", "osc2Level", "subLevel", "noiseLevel"))
    if np.any(levels < guards["min_total_level"]):
        raise DatasetError("parameter targets violate the minimum oscillator level guard")

    return features, parameters, groups, provenance
