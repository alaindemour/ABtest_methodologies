"""
Bayesian utilities for A/B testing.

This package provides Bayesian methods for non-inferiority testing,
variant selection, and conversion rate analysis using Beta-Bernoulli
conjugate models.
"""

from .utils import (
    test_non_inferiority,
    test_non_inferiority_weakly_informative,
    select_best_variant
)

__all__ = [
    'test_non_inferiority',
    'test_non_inferiority_weakly_informative',
    'select_best_variant'
]
