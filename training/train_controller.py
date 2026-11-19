#!/usr/bin/env python3
"""Train the adapter's v1-64 controller from a validated paired NPZ dataset.

The script never invents targets or emits weights when the dataset contract fails.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from sogitech_adapter.controller import (
    CONTROLLER_CONTRACT,
    CONTINUOUS_NAMES,
    CONTROLLER_OUTPUT_NAMES,
    ENUM_NAMES,
    FEATURE_CONTRACT,
    PARAMETER_CONTRACT,
    build_network,
)
from sogitech_adapter.ranges import ParamRanges, REQUIRED_18
from training.dataset import load_training_dataset


def _prepare_targets(parameters: np.ndarray, ranges: ParamRanges):
    index = {name: position for position, name in enumerate(REQUIRED_18)}
    continuous = np.column_stack([
        np.asarray([ranges.to_norm(name, float(value)) for value in parameters[:, index[name]]],
                   dtype=np.float32)
        for name in CONTINUOUS_NAMES
    ])
    osc1 = parameters[:, index["osc1Wave"]].astype(np.int64)
    osc2 = parameters[:, index["osc2Wave"]].astype(np.int64)
    return continuous, osc1, osc2


def _group_split(groups: np.ndarray, validation_fraction: float, seed: int):
    """Split whole source recordings/sessions so rows from one source cannot leak."""
    unique_groups = np.unique(np.asarray(groups).astype(str))
    if unique_groups.size < 2:
        raise ValueError("at least two source groups are required for a leakage-safe split")
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    validation_count = max(1, int(round(unique_groups.size * validation_fraction)))
    validation_count = min(validation_count, unique_groups.size - 1)
    validation_groups = shuffled_groups[:validation_count]
    is_validation = np.isin(groups, validation_groups)
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def _evaluate(torch, model, features, continuous, osc1, osc2):
    model.eval()
    with torch.no_grad():
        predicted, wave1_logits, wave2_logits = model(features)
        continuous_loss = torch.nn.functional.smooth_l1_loss(predicted, continuous)
        wave1_loss = torch.nn.functional.cross_entropy(wave1_logits, osc1)
        wave2_loss = torch.nn.functional.cross_entropy(wave2_logits, osc2)
        total = continuous_loss + 0.25 * (wave1_loss + wave2_loss)
        continuous_mae = torch.mean(torch.abs(predicted - continuous)).item()
        wave1_accuracy = (torch.argmax(wave1_logits, dim=-1) == osc1).float().mean().item()
        wave2_accuracy = (torch.argmax(wave2_logits, dim=-1) == osc2).float().mean().item()
    return {
        "loss": float(total.item()),
        "continuous_mae_normalized": float(continuous_mae),
        "osc1_wave_accuracy": float(wave1_accuracy),
        "osc2_wave_accuracy": float(wave2_accuracy),
    }


def train(args) -> Path:
    if (args.epochs < 1 or args.batch_size < 1 or args.seed < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("epochs, batch_size, seed, and learning_rate are invalid")
    if (not math.isfinite(args.validation_fraction)
            or not 0.05 <= args.validation_fraction <= 0.5):
        raise ValueError("validation_fraction must be between 0.05 and 0.5")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install the train extra") from exc

    ranges = ParamRanges(args.ranges)
    features_np, parameters_np, groups, provenance = load_training_dataset(args.data, ranges)
    if features_np.shape[0] < 10:
        raise ValueError("at least 10 paired samples are required for a train/validation split")

    continuous_np, osc1_np, osc2_np = _prepare_targets(parameters_np, ranges)
    training_indices, validation_indices = _group_split(
        groups, args.validation_fraction, args.seed
    )
    if training_indices.size < 2 or validation_indices.size < 1:
        raise ValueError("source-group split produced an unusable train/validation partition")

    torch.manual_seed(args.seed)
    x = torch.as_tensor(features_np, dtype=torch.float32)
    y_cont = torch.as_tensor(continuous_np, dtype=torch.float32)
    y_osc1 = torch.as_tensor(osc1_np, dtype=torch.long)
    y_osc2 = torch.as_tensor(osc2_np, dtype=torch.long)
    model = build_network(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    for _epoch in range(args.epochs):
        model.train()
        shuffled = rng.permutation(training_indices)
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start:start + args.batch_size]
            predicted, wave1_logits, wave2_logits = model(x[batch])
            continuous_loss = torch.nn.functional.smooth_l1_loss(predicted, y_cont[batch])
            wave1_loss = torch.nn.functional.cross_entropy(wave1_logits, y_osc1[batch])
            wave2_loss = torch.nn.functional.cross_entropy(wave2_logits, y_osc2[batch])
            loss = continuous_loss + 0.25 * (wave1_loss + wave2_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    metrics = _evaluate(
        torch, model,
        x[validation_indices], y_cont[validation_indices],
        y_osc1[validation_indices], y_osc2[validation_indices],
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("training produced non-finite validation metrics; refusing to save weights")

    digest = hashlib.sha256()
    with Path(args.data).open("rb") as dataset_stream:
        for chunk in iter(lambda: dataset_stream.read(1024 * 1024), b""):
            digest.update(chunk)

    checkpoint = {
        "controller_contract": CONTROLLER_CONTRACT,
        "feature_contract": FEATURE_CONTRACT,
        "parameter_contract": PARAMETER_CONTRACT,
        "parameter_names": list(REQUIRED_18),
        "output_names": list(CONTROLLER_OUTPUT_NAMES),
        "enum_names": list(ENUM_NAMES),
        "continuous_names": list(CONTINUOUS_NAMES),
        "range_source_commit": ranges.source_commit,
        "state_dict": model.state_dict(),
        "dataset_sha256": digest.hexdigest(),
        "dataset_provenance": provenance,
        "source_group_count": int(np.unique(groups).size),
        "sample_count": int(features_np.shape[0]),
        "training_sample_count": int(training_indices.size),
        "validation_sample_count": int(validation_indices.size),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "validation_metrics": metrics,
    }

    output = Path(args.out)
    if output.resolve() == Path(args.data).resolve():
        raise ValueError("checkpoint output path must not overwrite the training dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Validated paired NPZ training dataset")
    parser.add_argument("--ranges", default="ranges.json")
    parser.add_argument("--out", default="weights/controller-v1-64.pt")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = train(args)
    print(f"Saved contract-validated controller: {output}")


if __name__ == "__main__":
    main()
