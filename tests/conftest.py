"""
Shared pytest fixtures for A/B testing framework tests.

This module provides reusable test fixtures to ensure consistency
across all test files and reduce code duplication.
"""

from typing import Dict, Any
import pytest


@pytest.fixture
def control_group_data() -> Dict[str, int]:
    """
    Provides standard control group data from notebook examples.

    Returns:
        Dictionary with 'n' (total users) and 'x' (converted users)

    Why this fixture:
        - Ensures all tests use consistent control group data
        - Matches the actual experiment data from the notebooks
        - Makes test data easily discoverable and modifiable
    """
    control_total_users: int = 4411
    control_converted_users: int = 3138

    control_data: Dict[str, int] = {
        'n': control_total_users,
        'x': control_converted_users
    }

    return control_data


@pytest.fixture
def variants_experiment_data() -> Dict[str, Dict[str, int]]:
    """
    Provides standard variants data from notebook examples.

    Returns:
        Dictionary mapping variant names to their statistics
        Each variant has 'n' (total users) and 'x' (converted users)

    Why this fixture:
        - Three variants (A, B, C) from actual experiment
        - Provides realistic sample sizes for testing
        - Ensures consistent test data across all test functions
    """
    variants: Dict[str, Dict[str, int]] = {
        'A': {'n': 561, 'x': 381},
        'B': {'n': 285, 'x': 192},
        'C': {'n': 294, 'x': 201}
    }

    return variants


@pytest.fixture
def non_inferiority_test_parameters() -> Dict[str, Any]:
    """
    Provides standard parameters for non-inferiority testing.

    Returns:
        Dictionary with epsilon (margin), alpha (significance),
        prior_strength, and threshold values

    Why these values:
        - epsilon=0.05: Accept up to 5% degradation in conversion rate
        - alpha=0.05: Standard 5% significance level (reference only in Bayesian)
        - alpha_prior_strength=20: Weak prior that lets data dominate
        - threshold=0.95: Require 95% probability for non-inferiority decision
    """
    test_params: Dict[str, Any] = {
        'epsilon': 0.05,
        'alpha': 0.05,
        'alpha_prior_strength': 20,
        'threshold': 0.95
    }

    return test_params
