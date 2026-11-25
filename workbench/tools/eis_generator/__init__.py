"""Homogeneous through-plane catalyst-layer impedance generator."""

from .logic import (
    DEFAULT_PARAMETERS,
    generate_cl_eis,
    two_rail_impedance,
    validate_parameters,
)

__all__ = [
    "DEFAULT_PARAMETERS",
    "generate_cl_eis",
    "two_rail_impedance",
    "validate_parameters",
]
