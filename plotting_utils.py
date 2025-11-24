"""
Plotting utilities for AB test methodologies.

This module contains helper functions for creating visualizations
related to hypothesis testing and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm, beta as beta_dist


def plot_gaussian_hypothesis_test(mu_H0, sigma_H0, observed_value, alpha, epsilon,
                                   figsize=(7.5, 4.5)):
    """
    Plot a Gaussian distribution under the null hypothesis with hypothesis test annotations.

    This function visualizes the sampling distribution under H0 for a non-inferiority test,
    showing the probability density, observed value, critical value, and the right-tail
    probability (p-value).

    Parameters
    ----------
    mu_H0 : float
        Mean of the distribution under the null hypothesis (typically -epsilon)
    sigma_H0 : float
        Standard deviation of the distribution under the null hypothesis (typically SE)
    observed_value : float
        The observed test statistic (e.g., observed delta)
    alpha : float
        Significance level for the test (e.g., 0.05)
    epsilon : float
        The non-inferiority margin (positive value)
    figsize : tuple, optional
        Figure size as (width, height). Default is (7.5, 4.5)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    The function plots:
    - The Gaussian PDF curve
    - Shaded right-tail area beyond the observed value (representing p-value)
    - Vertical line at the observed value
    - Vertical line at the critical value
    - Vertical line at the mean under H0

    Examples
    --------
    >>> fig, ax = plot_gaussian_hypothesis_test(
    ...     mu_H0=-0.01, sigma_H0=0.005, observed_value=0.002,
    ...     alpha=0.05, epsilon=0.01
    ... )
    >>> plt.show()
    """
    # Domain for plotting (±6σ around the mean, clipped to reasonable bounds)
    left = mu_H0 - 6 * sigma_H0
    right = mu_H0 + 6 * sigma_H0
    xs = np.linspace(left, right, 1000)
    pdf = norm.pdf(xs, loc=mu_H0, scale=sigma_H0)

    # Right-tail probability
    p = norm.sf(observed_value, loc=mu_H0, scale=sigma_H0)

    # Critical value at significance alpha (one-sided)
    crit_x = norm.ppf(1 - alpha, loc=mu_H0, scale=sigma_H0)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.plot(xs, pdf, color="C0", lw=2, label="Density")

    # Shade right tail
    mask = xs >= observed_value
    ax.fill_between(xs[mask], pdf[mask], color="C1", alpha=0.35,
                     label=f"Right tail p = {p:.4g}")

    # Vertical line at observed value
    ax.axvline(observed_value, color="C1", ls="--", lw=1.5,
               label=f"observed delta = {observed_value:.4f}")

    # Vertical line at critical value
    ax.axvline(crit_x, color="C2", ls="-.", lw=1.5,
               label=f"critical value c = {crit_x:.4f}")

    # Vertical line at the mean for the null hypothesis H0
    ax.axvline(-epsilon, color="k", ls=":", lw=1.5,
               label=f"mean under H0 = {-epsilon:.4f}")

    # Decorations
    ax.set_title(f"Normal(μ={mu_H0:.4f}, σ={sigma_H0:.4f}) — Right-tail beyond x")
    ax.set_xlabel("Δ (difference in proportions) under the null H0")
    ax.set_ylabel("Probability density under H0")
    ax.legend(loc="best")
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()

    return fig, ax


def plot_type_ii_error_analysis(mu_H1, sigma_H1, critical_value, hatDelta_observed,
                                epsilon, beta, power, figsize=(7.5, 4.5)):
    """
    Plot Type II error (β) and power analysis for a hypothesis test.

    This function visualizes the sampling distribution under the alternative hypothesis H1,
    showing the Type II error region, power, and key test boundaries.

    Parameters
    ----------
    mu_H1 : float
        Mean of the distribution under the alternative hypothesis H1
    sigma_H1 : float
        Standard deviation of the distribution under H1
    critical_value : float
        The critical value from the null hypothesis test
    hatDelta_observed : float
        The observed difference in proportions
    epsilon : float
        The non-inferiority margin (positive value)
    beta : float
        Type II error probability
    power : float
        Statistical power (1 - beta)
    figsize : tuple, optional
        Figure size as (width, height). Default is (7.5, 4.5)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    The function plots:
    - The Gaussian PDF curve under H1
    - Shaded Type II error region (left tail beyond critical value)
    - Vertical lines at critical value, observed delta, mean under H1, and H0 boundary
    - Power annotation in a text box

    Examples
    --------
    >>> fig, ax = plot_type_ii_error_analysis(
    ...     mu_H1=0.02, sigma_H1=0.005, critical_value=0.001,
    ...     hatDelta_observed=0.015, epsilon=0.01, beta=0.1, power=0.9
    ... )
    >>> plt.show()
    """
    # Domain for plotting (±6σ around the mean, clipped to reasonable bounds)
    left = mu_H1 - 6 * sigma_H1
    right = mu_H1 + 6 * sigma_H1
    xs = np.linspace(left, right, 1000)
    pdf = norm.pdf(xs, loc=mu_H1, scale=sigma_H1)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.plot(xs, pdf, color="C0", lw=2, label="Density under H₁")

    # Shade left tail (Type II error region)
    mask = xs <= critical_value
    ax.fill_between(xs[mask], pdf[mask], color="C1", alpha=0.35,
                    label=f"Type II error β = {beta:.4g}")

    # Vertical line at critical value
    ax.axvline(critical_value, color="C2", ls="-.", lw=1.5,
               label=f"critical value c = {critical_value:.4f}")

    # Vertical line at observed delta
    ax.axvline(hatDelta_observed, color="C1", ls="--", lw=1.5,
               label=f"observed delta = {hatDelta_observed:.4f}")

    # Vertical line at the mean under H1
    ax.axvline(mu_H1, color="k", ls=":", lw=1.5,
               label=f"mean under H₁ = {mu_H1:.4f}")

    # Vertical line at the H0 boundary (for reference)
    ax.axvline(-epsilon, color="gray", ls=":", lw=1.5, alpha=0.7,
               label=f"H₀ boundary = {-epsilon:.4f}")

    # Decorations
    ax.set_title(f"Normal(μ={mu_H1:.4f}, σ={sigma_H1:.4f}) — Type II Error (β) and Power")
    ax.set_xlabel("Δ (difference in proportions) under H₁")
    ax.set_ylabel("Probability density under H₁")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    # Add text annotation for power
    power_text = f"Power = 1 - β = {power:.4f}"
    ax.text(0.98, 0.95, power_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig, ax


def plot_beta_prior_comparison(cases=None, figsize=(10, 7)):
    """
    Plot comparison of different Beta prior distributions.

    This function creates a 2x2 grid comparing four different Beta prior distributions,
    including uninformative, weakly informative, strongly informative, and bi-modal priors.

    Parameters
    ----------
    cases : list of tuples, optional
        List of (title, (kind, params)) tuples. If None, uses default cases:
        - Uninformative flat: Beta(1, 1)
        - Weakly informative: Beta(3, 12)
        - Strong conviction: Beta(200, 800)
        - Bi-modal mixture: mix of Beta(5,20) and Beta(20,5)
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 7)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects (2x2)

    Notes
    -----
    For single Beta distributions, plots:
    - PDF curve with filled area
    - Mean (dotted vertical line)
    - 95% credible interval (dashed lines)

    For mixture distributions, plots:
    - Combined PDF
    - Component means (dashed lines)
    - Mixture mean (dotted line)

    Examples
    --------
    >>> fig, axes = plot_beta_prior_comparison()
    >>> plt.show()
    """
    if cases is None:
        # Default cases
        cases = [
            ("Uninformative (flat)", ("single", (1, 1))),
            ("Weakly informative (centered, high entropy)", ("single", (3, 12))),
            ("Strong conviction (centered, low entropy)", ("single", (200, 800))),
            ("Bi-modal (mixture of Betas)", ("mixture", ((5, 20, 0.5), (20, 5, 0.5))))
        ]

    x = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    for ax, (title, (kind, params)) in zip(axes.ravel(), cases):
        if kind == "single":
            a, b = params
            y = beta_dist.pdf(x, a, b)
            mean = a / (a + b)
            ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

            ax.plot(x, y, color="C0", lw=2)
            ax.fill_between(x, y, color="C0", alpha=0.12)
            ax.axvline(mean, color="k", ls=":", lw=1.2, label=f"mean={mean:.3f}")
            ax.axvline(ci_low, color="C1", ls="--", lw=1, label="95% CI")
            ax.axvline(ci_high, color="C1", ls="--", lw=1)
            ax.set_title(f"{title}\nBeta(α={a}, β={b})", fontsize=10)
            ax.legend(fontsize=8, loc="best")
        else:
            # Mixture of two Betas: (a1, b1, w1), (a2, b2, w2)
            (a1, b1, w1), (a2, b2, w2) = params
            y = w1 * beta_dist.pdf(x, a1, b1) + w2 * beta_dist.pdf(x, a2, b2)
            m1, m2 = a1 / (a1 + b1), a2 / (a2 + b2)
            mix_mean = w1 * m1 + w2 * m2

            ax.plot(x, y, color="C0", lw=2, label="mixture pdf")
            ax.fill_between(x, y, color="C0", alpha=0.12)
            # Component means
            ax.axvline(m1, color="C2", ls="--", lw=1, label=f"mean₁={m1:.2f}")
            ax.axvline(m2, color="C3", ls="--", lw=1, label=f"mean₂={m2:.2f}")
            # Mixture mean
            ax.axvline(mix_mean, color="k", ls=":", lw=1.2, label=f"mixture mean={mix_mean:.2f}")
            ax.set_title(f"{title}\n{w1:.1f}·Beta({a1},{b1}) + {w2:.1f}·Beta({a2},{b2})",
                        fontsize=10)
            ax.legend(fontsize=8, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("p")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")

    fig.suptitle("Four Beta Priors: flat, weakly centered, strongly centered, bi-modal",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, axes


def plot_prior_vs_posterior(alpha, beta_param, control_group_conversion_rate,
                           epsilon, p_L, p_U, figsize=(10, 6)):
    """
    Plot non-informative prior vs posterior distribution for Bayesian analysis.

    This function visualizes the comparison between a non-informative Beta(1,1) prior
    and the posterior distribution after observing data, including credible intervals
    and key reference points.

    Parameters
    ----------
    alpha : float
        Alpha parameter of the posterior Beta distribution
    beta_param : float
        Beta parameter of the posterior Beta distribution
    control_group_conversion_rate : float
        The conversion rate of the control group
    epsilon : float
        The non-inferiority margin (positive value)
    p_L : float
        Lower bound of the 95% credible interval
    p_U : float
        Upper bound of the 95% credible interval
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 6)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    The function plots:
    - Prior distribution Beta(1,1) as a dashed blue line
    - Posterior distribution as a solid red line
    - Non-inferiority boundary (vertical dotted line)
    - Control group conversion rate (vertical dotted line)
    - Shaded 95% credible interval region

    Examples
    --------
    >>> fig, ax = plot_prior_vs_posterior(
    ...     alpha=50, beta_param=200, control_group_conversion_rate=0.2,
    ...     epsilon=0.01, p_L=0.17, p_U=0.23
    ... )
    >>> plt.show()
    """
    # Auto-calculate x-range based on credible interval with padding
    boundary = control_group_conversion_rate - epsilon
    x_min = max(0.0, min(p_L, boundary) - 0.05)
    x_max = min(1.0, max(p_U, control_group_conversion_rate) + 0.05)

    x_range = np.linspace(x_min, x_max, 1000)
    prior_noninformative_pdf = beta_dist.pdf(x_range, 1, 1)  # Beta(1,1) - uniform
    posterior_noninformative_pdf = beta_dist.pdf(x_range, alpha, beta_param)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_range, prior_noninformative_pdf, 'b--', lw=2,
            label='Prior: Beta(1, 1) - Non-informative')
    ax.plot(x_range, posterior_noninformative_pdf, 'r-', lw=2,
            label=f'Posterior: Beta({alpha:.1f}, {beta_param:.1f})')

    # Mark the non-inferiority boundary
    ax.axvline(control_group_conversion_rate - epsilon, color='k', ls=':', lw=1.5,
               label=f'Non-inferiority boundary = {control_group_conversion_rate - epsilon:.2f}')

    # Mark the control conversion rate
    ax.axvline(control_group_conversion_rate, color='g', ls=':', lw=1.5,
               label=f'Control conversion rate = {control_group_conversion_rate:.2f}')

    # Shade the 95% credible interval
    mask = (x_range >= p_L) & (x_range <= p_U)
    ax.fill_between(x_range[mask], posterior_noninformative_pdf[mask], alpha=0.3,
                    color='red', label='95% Credible Interval')

    ax.set_xlabel('Conversion rate p_A', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title('Non-informative Prior vs Posterior Distribution', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()

    return fig, ax


def plot_informative_prior_posterior_comparison(alpha_prior, beta_prior, alpha_posterior,
                                                beta_posterior, threshold, figsize=(10, 6)):
    """
    Plot informative prior vs posterior with non-inferiority tail areas.

    This function visualizes both a weakly-informative prior and the posterior
    distribution, highlighting the tail areas above the non-inferiority threshold.
    This is useful for demonstrating how data updates beliefs in a Bayesian framework.

    Parameters
    ----------
    alpha_prior : float
        Alpha parameter of the prior Beta distribution
    beta_prior : float
        Beta parameter of the prior Beta distribution
    alpha_posterior : float
        Alpha parameter of the posterior Beta distribution
    beta_posterior : float
        Beta parameter of the posterior Beta distribution
    threshold : float
        The non-inferiority boundary (typically control_rate - epsilon)
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 6)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    prob_non_inferior_post : float
        Posterior probability above threshold
    prob_non_inferior_prior : float
        Prior probability above threshold

    Notes
    -----
    The function plots:
    - Prior distribution as a dashed blue line
    - Posterior distribution as a solid red line
    - Non-inferiority boundary (vertical dotted line)
    - Shaded posterior tail area (red, darker)
    - Shaded prior tail area (blue, lighter)
    - Both probabilities are also returned for further use

    The plot range is automatically adjusted to focus on the relevant region
    around the threshold.

    Examples
    --------
    >>> fig, ax, post_prob, prior_prob = plot_informative_prior_posterior_comparison(
    ...     alpha_prior=20, beta_prior=80, alpha_posterior=53, beta_posterior=197,
    ...     threshold=0.17
    ... )
    >>> print(f"Posterior P(p_A > threshold) = {post_prob:.4f}")
    >>> plt.show()
    """
    # Compute probabilities (posterior and prior) for reference
    prob_non_inferior_post = beta_dist.sf(threshold, alpha_posterior, beta_posterior)
    prob_non_inferior_prior = beta_dist.sf(threshold, alpha_prior, beta_prior)

    # Plot range focused around the relevant region
    x_min = max(0.0, threshold - 0.12)
    x_max = min(1.0, threshold + 0.28)
    x_range = np.linspace(x_min, x_max, 1200)

    prior_pdf = beta_dist.pdf(x_range, alpha_prior, beta_prior)
    post_pdf = beta_dist.pdf(x_range, alpha_posterior, beta_posterior)

    fig, ax = plt.subplots(figsize=figsize)

    # Curves
    ax.plot(x_range, prior_pdf, color="#1f77b4", lw=2, ls="--",
            label=f"Prior Beta({alpha_prior:.2f}, {beta_prior:.2f})")
    ax.plot(x_range, post_pdf, color="#d62728", lw=2.5,
            label=f"Posterior Beta({alpha_posterior:.2f}, {beta_posterior:.2f})")

    # Non-inferiority boundary
    ax.axvline(threshold, color="k", ls=":", lw=1.8,
               label=f"Non-inferiority boundary = {threshold:.2f}")

    # Shade posterior tail (non-inferiority probability)
    mask_post = x_range >= threshold
    ax.fill_between(x_range[mask_post], post_pdf[mask_post], color="#d62728", alpha=0.3,
                    label=f"Posterior tail area = {prob_non_inferior_post:.3f}")

    # Optional: shade prior tail for comparison (lighter)
    mask_prior = x_range >= threshold
    ax.fill_between(x_range[mask_prior], prior_pdf[mask_prior], color="#1f77b4", alpha=0.15,
                    label=f"Prior tail area = {prob_non_inferior_prior:.3f}")

    # Decorations
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Conversion rate p_A", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Prior vs Posterior and Non-Inferiority Probability (tail area)", fontsize=14)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()

    return fig, ax, prob_non_inferior_post, prob_non_inferior_prior


def plot_weakly_informative_prior_with_variants(alpha_prior, beta_prior, variants_posteriors,
                                                 threshold, control_rate=None, epsilon=None,
                                                 colors=None, figsize=(12, 7)):
    """
    Plot weakly informative prior and all variant posteriors on the same figure.

    This function visualizes a common weakly informative prior alongside all variant
    posterior distributions, showing how data updates beliefs for each variant.
    Useful for comparing how different variants respond to the same prior.

    Parameters
    ----------
    alpha_prior : float
        Alpha parameter of the prior Beta distribution (common to all variants)
    beta_prior : float
        Beta parameter of the prior Beta distribution (common to all variants)
    variants_posteriors : dict
        Dictionary with variant names as keys, each containing:
        - 'alpha': float, alpha parameter of posterior Beta distribution
        - 'beta': float, beta parameter of posterior Beta distribution
        - 'n': int, sample size
        - 'x': int, number of successes
    threshold : float
        The non-inferiority boundary (typically control_rate - epsilon)
    control_rate : float, optional
        The control group conversion rate (for reference line)
    epsilon : float, optional
        The non-inferiority margin (for reference)
    colors : dict, optional
        Dictionary mapping variant names to color codes.
        If None, uses default colors
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 7)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    The function plots:
    - Prior distribution (thick dashed gray line)
    - Each variant's posterior (solid colored lines)
    - Non-inferiority threshold (red dotted line)
    - Control rate if provided (black dotted line)
    - Posterior tail probabilities above threshold

    Examples
    --------
    >>> variants_post = {
    ...     'A': {'alpha': 401, 'beta': 190, 'n': 561, 'x': 381},
    ...     'B': {'alpha': 212, 'beta': 103, 'n': 285, 'x': 192},
    ...     'C': {'alpha': 221, 'beta': 103, 'n': 294, 'x': 201}
    ... }
    >>> fig, ax = plot_weakly_informative_prior_with_variants(
    ...     alpha_prior=20, beta_prior=10.24,
    ...     variants_posteriors=variants_post,
    ...     threshold=0.66, control_rate=0.71, epsilon=0.05
    ... )
    >>> plt.show()
    """
    # Default colors if not provided
    if colors is None:
        colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    # Calculate prior mean and probability
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    prob_prior = 1 - beta_dist.cdf(threshold, alpha_prior, beta_prior)

    # Determine x-axis range
    x_min = max(0.0, threshold - 0.10)
    x_max = min(1.0, threshold + 0.20)
    if control_rate is not None:
        x_max = min(1.0, max(x_max, control_rate + 0.05))
    x_range = np.linspace(x_min, x_max, 1200)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot prior
    prior_pdf = beta_dist.pdf(x_range, alpha_prior, beta_prior)
    ax.plot(x_range, prior_pdf, color='gray', lw=3, ls='--', alpha=0.8,
            label=f'Prior: Beta(α={alpha_prior:.1f}, β={beta_prior:.1f}), mean={prior_mean:.3f}')

    # Shade prior tail area
    mask_prior = x_range >= threshold
    ax.fill_between(x_range[mask_prior], prior_pdf[mask_prior],
                    color='gray', alpha=0.1,
                    label=f'Prior P(>threshold) = {prob_prior:.3f}')

    # Plot each variant's posterior
    variant_stats = []
    for name in sorted(variants_posteriors.keys()):
        data = variants_posteriors[name]
        alpha_post = data['alpha']
        beta_post = data['beta']
        n = data.get('n', 0)
        x = data.get('x', 0)

        # Calculate statistics
        post_mean = alpha_post / (alpha_post + beta_post)
        obs_rate = x / n if n > 0 else 0
        prob_non_inferior = 1 - beta_dist.cdf(threshold, alpha_post, beta_post)

        # Get color
        color = colors.get(name, f'C{hash(name) % 10}')

        # Plot posterior PDF
        post_pdf = beta_dist.pdf(x_range, alpha_post, beta_post)
        ax.plot(x_range, post_pdf, color=color, lw=2.5,
                label=f'{name}: n={n}, x={x} (obs={obs_rate:.3f})')

        # Mark posterior mean
        ax.axvline(post_mean, color=color, ls=':', lw=1.2, alpha=0.6)

        # Shade tail area for this variant
        mask_post = x_range >= threshold
        ax.fill_between(x_range[mask_post], post_pdf[mask_post],
                       color=color, alpha=0.15)

        variant_stats.append({
            'name': name,
            'post_mean': post_mean,
            'prob': prob_non_inferior,
            'n': n,
            'x': x
        })

    # Add threshold line
    ax.axvline(threshold, color='red', ls=':', lw=2.5,
              label=f'Non-inferiority threshold = {threshold:.3f}')

    # Add control rate line if provided
    if control_rate is not None:
        ax.axvline(control_rate, color='black', ls='-.', lw=2, alpha=0.7,
                  label=f'Control rate = {control_rate:.3f}')

    # Title with epsilon if provided
    if epsilon is not None:
        title = f'Weakly Informative Prior and Variant Posteriors\n(ε = {epsilon:.3f}, threshold = control - ε)'
    else:
        title = 'Weakly Informative Prior and Variant Posteriors'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.grid(True, ls=':', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=1)

    # Add text box with posterior probabilities
    textstr = 'P(variant > threshold):\n'
    for stat in variant_stats:
        status = '✓' if stat['prob'] >= 0.95 else '✗'
        textstr += f"{status} {stat['name']}: {stat['prob']:.3f}\n"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

    plt.tight_layout()

    return fig, ax


def plot_multiple_posteriors_comparison(posteriors, control_group_conversion_rate, epsilon,
                                       colors=None, x_range=None, figsize=(12, 6)):
    """
    Plot multiple posterior distributions for variant comparison.

    This function visualizes posterior Beta distributions for multiple variants,
    showing their PDFs, means, credible intervals, and reference boundaries.
    Useful for comparing multiple test variants against a control.

    Parameters
    ----------
    posteriors : dict
        Dictionary with variant names as keys, each containing:
        - 'alpha': float, alpha parameter of posterior Beta distribution
        - 'beta': float, beta parameter of posterior Beta distribution
        - 'mean': float, posterior mean
        - 'ci_95': tuple of (lower, upper) bounds of 95% credible interval
    control_group_conversion_rate : float
        The conversion rate of the control group
    epsilon : float
        The non-inferiority margin (positive value)
    colors : dict, optional
        Dictionary mapping variant names to color codes.
        If None, uses default colors for 'A', 'B', 'C'
    x_range : array-like, optional
        X-axis range for plotting. If None, uses np.linspace(0.15, 0.30, 1000)
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 6)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    The function plots for each variant:
    - PDF curve (solid line)
    - Mean (dashed vertical line)
    - 95% credible interval (shaded region)

    Additionally shows:
    - Non-inferiority boundary (red dotted line)
    - Control group conversion rate (black dotted line)

    Examples
    --------
    >>> posteriors = {
    ...     'A': {'alpha': 169, 'beta': 633, 'mean': 0.2107, 'ci_95': (0.183, 0.240)},
    ...     'B': {'alpha': 173, 'beta': 629, 'mean': 0.2157, 'ci_95': (0.187, 0.245)},
    ...     'C': {'alpha': 166, 'beta': 636, 'mean': 0.2069, 'ci_95': (0.179, 0.236)}
    ... }
    >>> fig, ax = plot_multiple_posteriors_comparison(
    ...     posteriors, control_group_conversion_rate=0.2, epsilon=0.03
    ... )
    >>> plt.show()
    """
    # Default colors if not provided
    if colors is None:
        colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    # Auto-calculate x_range if not provided
    if x_range is None:
        # Find min and max across all credible intervals
        all_ci_lower = [posteriors[name]['ci_95'][0] for name in posteriors.keys()]
        all_ci_upper = [posteriors[name]['ci_95'][1] for name in posteriors.keys()]
        boundary = control_group_conversion_rate - epsilon

        # Include boundary and control rate in the range calculation
        x_min = max(0.0, min(min(all_ci_lower), boundary) - 0.05)
        x_max = min(1.0, max(max(all_ci_upper), control_group_conversion_rate) + 0.05)

        x_range = np.linspace(x_min, x_max, 1000)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each variant
    for name in sorted(posteriors.keys()):
        alpha_p = posteriors[name]['alpha']
        beta_p = posteriors[name]['beta']
        mean_p = posteriors[name]['mean']
        ci_lower, ci_upper = posteriors[name]['ci_95']

        # Get color (use default if variant not in colors dict)
        color = colors.get(name, f'C{hash(name) % 10}')

        # Plot PDF
        pdf = beta_dist.pdf(x_range, alpha_p, beta_p)
        ax.plot(x_range, pdf, color=color, lw=2.5,
                label=f'{name}: mean={mean_p:.4f}')

        # Mark the mean
        ax.axvline(mean_p, color=color, ls='--', lw=1, alpha=0.5)

        # Shade 95% credible interval
        mask = (x_range >= ci_lower) & (x_range <= ci_upper)
        ax.fill_between(x_range[mask], pdf[mask], alpha=0.2, color=color)

    # Add non-inferiority boundary
    boundary = control_group_conversion_rate - epsilon
    ax.axvline(boundary, color='red', ls=':', lw=2,
               label=f'Non-inferiority boundary ({boundary:.2f})')

    # Add control rate
    ax.axvline(control_group_conversion_rate, color='black', ls=':', lw=2,
               label=f'Control rate ({control_group_conversion_rate:.2f})')

    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title('Posterior Distributions for Variants A, B, C', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, ls=':', alpha=0.3)
    plt.tight_layout()

    return fig, ax
