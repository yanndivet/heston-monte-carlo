"""Variance reduction techniques for Monte Carlo simulation."""

from .random_generation import (
    generate_randoms,
    estimate_variance_reduction
)

__all__ = [
    "generate_randoms",
    "estimate_variance_reduction"
]