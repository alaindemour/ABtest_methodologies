"""
Pytest suite for Bayesian non-inferiority testing with weakly informative priors.

This test suite validates that the weakly informative prior methodology:
1. Produces mathematically correct posterior distributions
2. Makes correct non-inferiority decisions based on probability thresholds
3. Handles edge cases appropriately

---
🧠 THE LOGIC BLUEPRINT
---
For each test function, we:
1. SETUP: Load control and variant data from fixtures
2. EXECUTE: Run the non-inferiority test function
3. VALIDATE: Assert that results match expected mathematical properties
4. VERIFY: Check that decision logic (non-inferior vs not) is correct

Key validations:
- Results contain all expected fields (prior, posterior, probability, decision)
- Posterior parameters are updated correctly from prior + data
- Probabilities are valid (between 0 and 1)
- Non-inferiority decisions match probability threshold
- Conversion rates are calculated correctly
"""

from typing import Dict, Any
import pytest
from bayesian import test_non_inferiority_weakly_informative


def test_function_returns_results_for_all_variants(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that the function returns results for every input variant.

    Why this test:
        - Ensures no variants are silently dropped
        - Validates basic function execution without errors
        - Confirms output structure matches input structure
    """
    # ARRANGE: Extract test parameters
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT: Run the non-inferiority test
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Every input variant has a result
    expected_variant_names: set = set(variants_experiment_data.keys())
    actual_variant_names: set = set(test_results.keys())

    assert actual_variant_names == expected_variant_names, (
        f"Expected results for variants {expected_variant_names}, "
        f"but got results for {actual_variant_names}"
    )


def test_result_structure_contains_all_required_fields(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that each variant result contains all required fields.

    Why this test:
        - Validates the output contract/schema
        - Ensures downstream code can safely access all fields
        - Catches breaking changes to the API
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Every result has all required fields
    required_field_names: list[str] = [
        'prior_params',
        'prior_mean',
        'posterior_params',
        'variant_rate',
        'probability',
        'is_non_inferior'
    ]

    for variant_name, variant_result in test_results.items():
        for required_field in required_field_names:
            assert required_field in variant_result, (
                f"Variant '{variant_name}' result is missing required field '{required_field}'"
            )


def test_prior_parameters_are_mathematically_valid(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that prior parameters (alpha, beta) are positive numbers.

    Why this test:
        - Beta distribution requires alpha > 0 and beta > 0
        - Invalid priors would cause mathematical errors
        - Ensures the prior setup logic is correct
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Prior parameters are positive
    for variant_name, variant_result in test_results.items():
        prior_alpha: float = variant_result['prior_params'][0]
        prior_beta: float = variant_result['prior_params'][1]

        assert prior_alpha > 0, (
            f"Variant '{variant_name}' has invalid prior alpha: {prior_alpha} (must be > 0)"
        )
        assert prior_beta > 0, (
            f"Variant '{variant_name}' has invalid prior beta: {prior_beta} (must be > 0)"
        )


def test_posterior_parameters_increase_from_prior(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that posterior parameters are larger than prior (data was added).

    Why this test:
        - Posterior = Prior + Data in Beta-Bernoulli conjugate model
        - posterior_alpha = prior_alpha + successes
        - posterior_beta = prior_beta + failures
        - If posterior <= prior, the Bayesian update failed
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Posterior parameters >= Prior parameters
    for variant_name, variant_result in test_results.items():
        prior_alpha: float = variant_result['prior_params'][0]
        prior_beta: float = variant_result['prior_params'][1]
        posterior_alpha: float = variant_result['posterior_params'][0]
        posterior_beta: float = variant_result['posterior_params'][1]

        # Why >= instead of >: If we observed data (n>0), posterior must be strictly larger
        # We use >= to handle the edge case of n=0 (no data), though unlikely in practice
        assert posterior_alpha >= prior_alpha, (
            f"Variant '{variant_name}': posterior_alpha ({posterior_alpha}) "
            f"should be >= prior_alpha ({prior_alpha})"
        )
        assert posterior_beta >= prior_beta, (
            f"Variant '{variant_name}': posterior_beta ({posterior_beta}) "
            f"should be >= prior_beta ({prior_beta})"
        )


def test_probability_values_are_in_valid_range(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that all probability values are between 0 and 1.

    Why this test:
        - Probabilities must be in [0, 1] by mathematical definition
        - Values outside this range indicate calculation errors
        - Critical for decision-making logic
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: All probabilities are in [0, 1]
    for variant_name, variant_result in test_results.items():
        probability: float = variant_result['probability']

        assert 0.0 <= probability <= 1.0, (
            f"Variant '{variant_name}' has invalid probability: {probability} "
            f"(must be between 0 and 1)"
        )


def test_non_inferiority_decision_matches_threshold(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that is_non_inferior flag matches probability vs threshold comparison.

    Why this test:
        - Validates the decision logic is consistent
        - is_non_inferior should be True iff probability >= threshold
        - Ensures no logic bugs in the decision rule
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Decision logic is consistent with probability and threshold
    for variant_name, variant_result in test_results.items():
        probability: float = variant_result['probability']
        is_non_inferior: bool = variant_result['is_non_inferior']
        expected_decision: bool = probability >= probability_threshold

        assert is_non_inferior == expected_decision, (
            f"Variant '{variant_name}': is_non_inferior={is_non_inferior}, "
            f"but probability={probability:.4f} vs threshold={probability_threshold:.4f} "
            f"suggests it should be {expected_decision}"
        )


def test_variant_conversion_rates_are_calculated_correctly(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that variant_rate (posterior mean) is close to observed rate.

    Why this test:
        - With weak prior (alpha=20), posterior should be dominated by data
        - Posterior mean = (prior_alpha + successes) / (prior_alpha + prior_beta + total)
        - Should be very close to observed rate when sample size >> prior strength
        - Large divergence indicates calculation error

    🔍 VERIFICATION NOTE:
        This test might fail if:
        - Prior is very strong relative to sample size
        - Sample size is tiny (n < 50)
        - We change to a more informative prior
        Current tolerance: 0.02 (2 percentage points)
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: Posterior mean is close to observed rate
    tolerance: float = 0.02  # Allow 2 percentage point difference

    for variant_name, variant_result in test_results.items():
        variant_data: Dict[str, int] = variants_experiment_data[variant_name]
        observed_total_users: int = variant_data['n']
        observed_converted_users: int = variant_data['x']
        observed_conversion_rate: float = observed_converted_users / observed_total_users

        posterior_mean_rate: float = variant_result['variant_rate']

        rate_difference: float = abs(posterior_mean_rate - observed_conversion_rate)

        assert rate_difference < tolerance, (
            f"Variant '{variant_name}': posterior_mean_rate={posterior_mean_rate:.4f} "
            f"differs from observed_rate={observed_conversion_rate:.4f} "
            f"by {rate_difference:.4f} (tolerance: {tolerance})"
        )


def test_all_variants_share_same_prior_parameters(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> None:
    """
    Test that all variants receive the same prior (derived from control).

    Why this test:
        - Weakly informative prior is based on control group data
        - All variants should start with the same prior belief
        - Different priors would bias comparisons between variants
    """
    # ARRANGE
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    # ACT
    test_results: Dict[str, Dict[str, Any]] = test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    # ASSERT: All variants have identical prior parameters
    variant_names: list[str] = list(test_results.keys())

    if len(variant_names) < 2:
        pytest.skip("Need at least 2 variants to compare priors")

    # Use first variant as reference
    reference_variant_name: str = variant_names[0]
    reference_prior_params: tuple = test_results[reference_variant_name]['prior_params']
    reference_alpha: float = reference_prior_params[0]
    reference_beta: float = reference_prior_params[1]

    # Compare all other variants to reference
    for variant_name in variant_names[1:]:
        variant_prior_params: tuple = test_results[variant_name]['prior_params']
        variant_alpha: float = variant_prior_params[0]
        variant_beta: float = variant_prior_params[1]

        assert variant_alpha == reference_alpha, (
            f"Variant '{variant_name}' has prior_alpha={variant_alpha}, "
            f"but '{reference_variant_name}' has prior_alpha={reference_alpha}. "
            f"All variants should share the same prior."
        )
        assert variant_beta == reference_beta, (
            f"Variant '{variant_name}' has prior_beta={variant_beta}, "
            f"but '{reference_variant_name}' has prior_beta={reference_beta}. "
            f"All variants should share the same prior."
        )
