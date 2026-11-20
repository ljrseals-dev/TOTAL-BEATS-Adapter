import builtins
from types import SimpleNamespace

import pytest

from sogitech_adapter.controller import ControllerError, build_network, load_controller
from sogitech_adapter.ranges import ParamRanges
from training.train_controller import train


def _without_torch(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_build_network_fails_actionably_when_torch_is_missing(monkeypatch):
    _without_torch(monkeypatch)
    with pytest.raises(ControllerError, match=r"PyTorch is required.*neural extra") as error:
        build_network()
    assert isinstance(error.value.__cause__, ImportError)


def test_checkpoint_loader_fails_actionably_when_torch_is_missing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "controller.pt"
    checkpoint.write_bytes(b"not a real checkpoint; import must fail first")
    _without_torch(monkeypatch)

    with pytest.raises(ControllerError, match=r"PyTorch is required.*neural extra") as error:
        load_controller(checkpoint, ParamRanges("ranges.json"))
    assert isinstance(error.value.__cause__, ImportError)


def test_training_fails_actionably_when_torch_is_missing(monkeypatch):
    _without_torch(monkeypatch)
    args = SimpleNamespace(
        epochs=1,
        batch_size=2,
        seed=1,
        learning_rate=0.001,
        validation_fraction=0.2,
        data="not-opened-before-torch-import.npz",
        ranges="ranges.json",
        out="unused.pt",
    )
    with pytest.raises(RuntimeError, match=r"PyTorch is required.*train extra") as error:
        train(args)
    assert isinstance(error.value.__cause__, ImportError)
