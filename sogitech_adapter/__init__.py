"""Sogitech ↔ TOTAL BEATS / Neural Tide adapter package."""

from .ranges import ParamRanges, RangeError, REQUIRED_18
from .controller import (
    ControllerError,
    TrainedParameterController,
    load_controller,
)

__all__ = [
    "ParamRanges",
    "RangeError",
    "REQUIRED_18",
    "ControllerError",
    "TrainedParameterController",
    "load_controller",
]
__version__ = "0.1.0"
__contract__ = "sogitech.features.v1-64"
