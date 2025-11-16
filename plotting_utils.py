"""
Plotting utilities for AB test methodologies.

This module contains helper functions for creating visualizations
related to hypothesis testing and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm


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
