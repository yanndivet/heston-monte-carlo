"""
Validation utilities for Heston Monte Carlo.
"""
import numpy as np
from typing import Union, List


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability [0, 1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_correlation(value: float, name: str) -> None:
    """Validate that a value is a valid correlation [-1, 1]."""
    if not -1 <= value <= 1:
        raise ValueError(f"{name} must be in [-1, 1], got {value}")


def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str) -> None:
    """Validate that an array has the expected shape."""
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")


def validate_scheme(scheme: str, valid_schemes: List[str]) -> None:
    """Validate that a scheme is in the list of valid schemes."""
    if scheme not in valid_schemes:
        raise ValueError(f"Scheme must be one of {valid_schemes}, got '{scheme}'")


def check_numerical_stability(array: np.ndarray, name: str, 
                            max_value: float = 1e10) -> None:
    """Check for numerical instability (inf, nan, very large values)."""
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinite values") 
    if np.any(np.abs(array) > max_value):
        raise ValueError(f"{name} contains very large values (>{max_value})")


def format_parameter_summary(config) -> str:
    """Format a configuration object for display."""
    lines = []
    for field, value in config.__dict__.items():
        if isinstance(value, float):
            lines.append(f"  {field:<10} = {value:8.4f}")
        else:
            lines.append(f"  {field:<10} = {value}")
    return "\n".join(lines)