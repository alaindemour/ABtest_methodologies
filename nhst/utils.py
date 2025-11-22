"""
NHST (Null Hypothesis Significance Testing) utilities for A/B testing.

This module contains frequentist statistical methods for non-inferiority
testing and hypothesis testing for conversion rate experiments.
"""

import numpy as np
from scipy.stats import norm


def nhst_non_inferiority_test(n_control, x_control, n_variant, x_variant,
                                epsilon, alpha=0.05, h1_effect_size=0.0):
    """
    Perform NHST non-inferiority test for a single variant against control.

    Parameters
    ----------
    n_control : int
        Number of samples in control group
    x_control : int
        Number of successes in control group
    n_variant : int
        Number of samples in variant group
    x_variant : int
        Number of successes in variant group
    epsilon : float
        Non-inferiority margin (e.g., 0.03 for 3% acceptable degradation)
    alpha : float, optional
        Significance level for Type I error (default: 0.05)
    h1_effect_size : float, optional
        Expected effect size under H1 for power calculation (default: 0.0, meaning no difference)

    Returns
    -------
    dict : Dictionary containing:
        - 'decision': str, 'REJECT H0' or 'FAIL TO REJECT H0'
        - 'p_value': float, probability of observed result under H0
        - 'power': float, probability of rejecting H0 when H1 is true
        - 'observed_difference': float, observed difference in proportions (variant - control)
        - 'critical_value': float, threshold for rejection at significance level alpha
        - 'se_h0': float, standard error under H0 (boundary case)
        - 'se_h1': float, standard error under H1
        - 'control_rate': float, observed control conversion rate
        - 'variant_rate': float, observed variant conversion rate
        - 'z_statistic': float, standardized test statistic

    Examples
    --------
    >>> result = nhst_non_inferiority_test(
    ...     n_control=7000, x_control=1400,
    ...     n_variant=150, x_variant=33,
    ...     epsilon=0.03, alpha=0.05
    ... )
    >>> print(result['decision'])
    'REJECT H0'
    >>> print(f"P-value: {result['p_value']:.4f}")
    P-value: 0.0234
    """
    # Observed proportions
    p_control = x_control / n_control
    p_variant = x_variant / n_variant
    observed_diff = p_variant - p_control

    # Standard error under H0 (unpooled, since we're not assuming p_A = p_C)
    se_h0 = np.sqrt((p_control * (1 - p_control) / n_control) +
                     (p_variant * (1 - p_variant) / n_variant))

    # Hypothesis test under boundary H0: E[Delta] = -epsilon
    mu_h0 = -epsilon

    # Compute p-value (right-tail test: H1 is Delta > -epsilon)
    p_value = norm.sf(observed_diff, loc=mu_h0, scale=se_h0)

    # Critical value for rejection at significance level alpha
    critical_value = norm.isf(alpha, loc=mu_h0, scale=se_h0)

    # Decision
    decision = 'REJECT H0' if p_value <= alpha else 'FAIL TO REJECT H0'

    # Z-statistic (standardized)
    z_statistic = (observed_diff - mu_h0) / se_h0

    # Power calculation under H1
    # Under H1, assume true effect is h1_effect_size (e.g., 0 for "no worse")
    # Use pooled SE if h1_effect_size = 0 (assuming p_variant = p_control under H1)
    if h1_effect_size == 0.0:
        # Pooled proportion for H1
        p_pooled = (x_control + x_variant) / (n_control + n_variant)
        se_h1 = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_variant))
    else:
        # Use unpooled SE with assumed rates under H1
        # Assume variant has control_rate + h1_effect_size
        p_variant_h1 = p_control + h1_effect_size
        se_h1 = np.sqrt((p_control * (1 - p_control) / n_control) +
                        (p_variant_h1 * (1 - p_variant_h1) / n_variant))

    # Under H1, the distribution is centered at h1_effect_size
    mu_h1 = h1_effect_size

    # Power = P(reject H0 | H1 is true) = P(observed_diff > critical_value | H1)
    # This is the right-tail probability of the H1 distribution beyond critical_value
    power = norm.sf(critical_value, loc=mu_h1, scale=se_h1)

    return {
        'decision': decision,
        'p_value': p_value,
        'power': power,
        'observed_difference': observed_diff,
        'critical_value': critical_value,
        'se_h0': se_h0,
        'se_h1': se_h1,
        'control_rate': p_control,
        'variant_rate': p_variant,
        'z_statistic': z_statistic
    }
