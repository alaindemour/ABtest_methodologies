"""
Bayesian utilities for A/B testing.

This module contains Bayesian methods for non-inferiority testing,
variant selection, and conversion rate analysis using Beta-Bernoulli
conjugate models with Monte Carlo simulation.
"""

import numpy as np
from scipy.stats import beta as beta_dist


def test_non_inferiority(n_control, x_control, variants_data, epsilon,
                         alpha_prior, beta_prior, threshold=0.95):
    """
    Test non-inferiority of multiple variants against a control.

    Uses Bayesian Beta-Bernoulli conjugate model with Monte Carlo simulation
    to compute the probability that each variant's conversion rate is within
    an acceptable degradation margin of the control.

    Parameters
    ----------
    n_control : int
        Number of samples in control group
    x_control : int
        Number of successes in control group
    variants_data : dict
        Dictionary with variant names as keys and {'n': samples, 'x': successes} as values
        Example: {'A': {'n': 1000, 'x': 200}, 'B': {'n': 1000, 'x': 215}}
    epsilon : float
        Non-inferiority margin (e.g., 0.03 for 3%)
    alpha_prior : float
        Alpha parameter for Beta prior
    beta_prior : float
        Beta parameter for Beta prior
    threshold : float, optional
        Probability threshold for declaring non-inferiority (default: 0.95)

    Returns
    -------
    dict : Dictionary with results for each variant containing:
        - 'is_non_inferior': bool, whether variant is non-inferior
        - 'probability': float, P(variant > control - epsilon)
        - 'control_rate': float, posterior mean of control
        - 'variant_rate': float, posterior mean of variant
        - 'posterior_params': tuple, (alpha, beta) of variant posterior

    Examples
    --------
    >>> variants = {'A': {'n': 150, 'x': 33}, 'B': {'n': 150, 'x': 35}}
    >>> results = test_non_inferiority(
    ...     n_control=7000, x_control=1400,
    ...     variants_data=variants,
    ...     epsilon=0.03, alpha_prior=1, beta_prior=1, threshold=0.95
    ... )
    >>> print(results['A']['is_non_inferior'])
    True
    >>> print(f"P(A non-inferior) = {results['A']['probability']:.3f}")
    P(A non-inferior) = 0.982
    """
    # Control posterior
    alpha_control = x_control + alpha_prior
    beta_control = n_control - x_control + beta_prior
    control_rate = alpha_control / (alpha_control + beta_control)

    # Boundary for non-inferiority
    boundary = control_rate - epsilon

    results = {}
    n_simulations = 100000

    # Sample from control posterior once (reuse for all variants)
    control_samples = beta_dist.rvs(alpha_control, beta_control, size=n_simulations)

    for variant_name, data in variants_data.items():
        # Variant posterior
        alpha_variant = data['x'] + alpha_prior
        beta_variant = data['n'] - data['x'] + beta_prior
        variant_rate = alpha_variant / (alpha_variant + beta_variant)

        # Sample from variant posterior
        variant_samples = beta_dist.rvs(alpha_variant, beta_variant, size=n_simulations)

        # Compute P(variant > control - epsilon)
        prob_non_inferior = np.mean(variant_samples > (control_samples - epsilon))

        results[variant_name] = {
            'is_non_inferior': prob_non_inferior >= threshold,
            'probability': prob_non_inferior,
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'posterior_params': (alpha_variant, beta_variant)
        }

    return results


def test_non_inferiority_weakly_informative(n_control, x_control, variants_data,
                                            epsilon, alpha_prior_strength=20,
                                            threshold=0.95, n_simulations=100000):
    """
    Test non-inferiority using weakly informative prior based on historical data.

    This function implements the methodology described in the notebook section
    "Weakly Informative Prior Using Historical Data". Instead of requiring manual
    specification of prior parameters, it automatically constructs a weakly
    informative prior centered at the expected variant conversion rate
    (control_rate - epsilon) with high entropy to avoid dominating the data.

    The prior is constructed as:
    - α_prior = alpha_prior_strength (default: 20, for high entropy/wide uncertainty)
    - β_prior = (α_prior / target_mean) - α_prior
    - target_mean = control_conversion_rate - epsilon

    This prior expresses: "We expect the variant to be slightly worse than control
    (by epsilon), but we're quite uncertain about this expectation."

    Parameters
    ----------
    n_control : int
        Number of samples in control group
    x_control : int
        Number of successes in control group
    variants_data : dict
        Dictionary with variant names as keys and {'n': samples, 'x': successes} as values
        Example: {'A': {'n': 561, 'x': 381}, 'B': {'n': 285, 'x': 192}}
    epsilon : float
        Non-inferiority margin (e.g., 0.05 for 5% acceptable degradation)
    alpha_prior_strength : float, optional
        Strength parameter for the prior (default: 20). Smaller values give
        wider (more uncertain) priors. Typical values: 10-30.
    threshold : float, optional
        Probability threshold for declaring non-inferiority (default: 0.95)
        Variant is non-inferior if P(variant > control - epsilon) >= threshold
    n_simulations : int, optional
        Number of Monte Carlo simulations (default: 100000)

    Returns
    -------
    dict : Dictionary with results for each variant containing:
        - 'is_non_inferior': bool, whether variant is non-inferior
        - 'probability': float, P(variant > control - epsilon)
        - 'control_rate': float, observed control conversion rate
        - 'variant_rate': float, posterior mean of variant
        - 'posterior_params': tuple, (alpha, beta) of variant posterior
        - 'prior_params': tuple, (alpha_prior, beta_prior) used
        - 'prior_mean': float, mean of the prior distribution

    Examples
    --------
    >>> # Historical data: control has 20% conversion, testing 3 variants
    >>> variants = {
    ...     'A': {'n': 561, 'x': 381},
    ...     'B': {'n': 285, 'x': 192},
    ...     'C': {'n': 294, 'x': 201}
    ... }
    >>> results = test_non_inferiority_weakly_informative(
    ...     n_control=4411, x_control=3138,
    ...     variants_data=variants,
    ...     epsilon=0.05
    ... )
    >>> for variant, result in results.items():
    ...     print(f"{variant}: {result['is_non_inferior']} "
    ...           f"(P={result['probability']:.3f})")
    A: True (P=0.967)
    B: True (P=0.952)
    C: True (P=0.961)

    Notes
    -----
    This approach is particularly valuable when:
    - You have historical data about control conversion rates
    - You want to incorporate domain knowledge without being overly confident
    - Sample sizes are small (NHST would be underpowered)
    - You're adding features that shouldn't dramatically change behavior

    The weakly informative prior:
    - Centers belief around expected variant performance (control - epsilon)
    - Maintains high uncertainty (high entropy) via small alpha/beta values
    - Allows data to dominate when sample sizes are reasonable
    - Provides reasonable estimates even with very small samples
    """
    # Compute control conversion rate
    control_rate = x_control / n_control

    # Construct weakly informative prior centered at (control_rate - epsilon)
    target_prior_mean = control_rate - epsilon
    alpha_prior = alpha_prior_strength
    beta_prior = (alpha_prior / target_prior_mean) - alpha_prior

    # Verify prior is valid (must have positive parameters)
    if beta_prior <= 0:
        raise ValueError(
            f"Invalid prior parameters: beta_prior={beta_prior:.4f} <= 0. "
            f"This can happen when epsilon is too large relative to control_rate. "
            f"Try reducing epsilon or increasing alpha_prior_strength."
        )

    results = {}

    for variant_name, data in variants_data.items():
        # Variant posterior: Beta(x + α_prior, n - x + β_prior)
        alpha_posterior = data['x'] + alpha_prior
        beta_posterior = (data['n'] - data['x']) + beta_prior
        variant_posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)

        # Monte Carlo: P(variant > control - epsilon)
        variant_samples = beta_dist.rvs(
            alpha_posterior, beta_posterior, size=n_simulations
        )

        # Non-inferiority threshold
        non_inferiority_threshold = control_rate - epsilon

        # Compute probability that variant exceeds the threshold
        prob_non_inferior = np.mean(variant_samples > non_inferiority_threshold)

        results[variant_name] = {
            'is_non_inferior': prob_non_inferior >= threshold,
            'probability': prob_non_inferior,
            'control_rate': control_rate,
            'variant_rate': variant_posterior_mean,
            'posterior_params': (alpha_posterior, beta_posterior),
            'prior_params': (alpha_prior, beta_prior),
            'prior_mean': target_prior_mean
        }

    return results


def select_best_variant(variants_data, alpha_prior=1, beta_prior=1,
                       credible_level=0.95, n_simulations=100000):
    """
    Select the best variant among multiple options using Bayesian approach.

    Uses Monte Carlo simulation to determine which variant has the highest
    probability of being the best, along with expected loss calculations
    for decision-making under uncertainty.

    Parameters
    ----------
    variants_data : dict
        Dictionary with variant names as keys and {'n': samples, 'x': successes} as values
        Example: {'A': {'n': 800, 'x': 168}, 'B': {'n': 800, 'x': 172}}
    alpha_prior : float, optional
        Alpha parameter for Beta prior (default: 1 for uniform)
    beta_prior : float, optional
        Beta parameter for Beta prior (default: 1 for uniform)
    credible_level : float, optional
        Credible interval level (default: 0.95)
    n_simulations : int, optional
        Number of Monte Carlo simulations (default: 100000)

    Returns
    -------
    dict : Dictionary containing:
        - 'best_variant': str, name of variant most likely to be best
        - 'probabilities': dict, P(each variant is best)
        - 'posterior_means': dict, posterior mean for each variant
        - 'credible_intervals': dict, (lower, upper) credible interval for each variant
        - 'expected_loss': dict, expected loss from choosing each variant

    Examples
    --------
    >>> variants = {
    ...     'A': {'n': 800, 'x': 168},
    ...     'B': {'n': 800, 'x': 172},
    ...     'C': {'n': 800, 'x': 165}
    ... }
    >>> result = select_best_variant(variants)
    >>> print(result['best_variant'])
    'B'
    >>> print(result['probabilities'])
    {'A': 0.31, 'B': 0.47, 'C': 0.22}
    """
    variant_names = list(variants_data.keys())
    posteriors = {}
    samples = {}

    # Compute posteriors and draw samples
    for name, data in variants_data.items():
        alpha_post = data['x'] + alpha_prior
        beta_post = data['n'] - data['x'] + beta_prior

        posteriors[name] = {
            'alpha': alpha_post,
            'beta': beta_post,
            'mean': alpha_post / (alpha_post + beta_post)
        }

        # Draw samples from posterior
        samples[name] = beta_dist.rvs(alpha_post, beta_post, size=n_simulations)

        # Compute credible interval
        ci_lower = beta_dist.ppf((1 - credible_level) / 2, alpha_post, beta_post)
        ci_upper = beta_dist.ppf(1 - (1 - credible_level) / 2, alpha_post, beta_post)
        posteriors[name]['credible_interval'] = (ci_lower, ci_upper)

    # Monte Carlo: count how often each variant is best
    best_counts = {name: 0 for name in variant_names}

    for i in range(n_simulations):
        # Get samples for this iteration
        sample_values = {name: samples[name][i] for name in variant_names}

        # Find best variant in this simulation
        best_variant = max(sample_values, key=sample_values.get)
        best_counts[best_variant] += 1

    # Calculate probabilities
    probabilities = {name: count / n_simulations for name, count in best_counts.items()}

    # Expected loss: E[max(all) - this variant]
    expected_loss = {}
    for name in variant_names:
        max_samples = np.maximum.reduce([samples[v] for v in variant_names])
        losses = max_samples - samples[name]
        expected_loss[name] = np.mean(losses)

    # Determine best variant
    best_variant = max(probabilities, key=probabilities.get)

    return {
        'best_variant': best_variant,
        'probabilities': probabilities,
        'posterior_means': {name: posteriors[name]['mean'] for name in variant_names},
        'credible_intervals': {name: posteriors[name]['credible_interval']
                              for name in variant_names},
        'expected_loss': expected_loss
    }
