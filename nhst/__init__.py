"""
NHST (Null Hypothesis Significance Testing) utilities for A/B testing.

This package provides frequentist statistical methods for hypothesis testing
and non-inferiority testing in conversion rate experiments.
"""

from .utils import nhst_non_inferiority_test, compute_sample_size_non_inferiority

__all__ = ['nhst_non_inferiority_test', 'compute_sample_size_non_inferiority']
