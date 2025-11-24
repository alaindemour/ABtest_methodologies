"""
Test script for the improved plotting function that shows weakly informative
prior and all variant posteriors together.
"""

import matplotlib.pyplot as plt
from plotting_utils import plot_weakly_informative_prior_with_variants

# Data from the notebook example
nC = 4411
xC_observed = 3138
control_group_conversion_rate = xC_observed / nC

# Three variants with actual experiment data
variants_data = {
    'A': {'n': 561, 'x': 381},
    'B': {'n': 285, 'x': 192},
    'C': {'n': 294, 'x': 201}
}

# Test parameters
epsilon = 0.05  # 5% non-inferiority margin
alpha_prior_strength = 20

# Compute prior parameters (same as in the utility function)
target_prior_mean = control_group_conversion_rate - epsilon
alpha_prior = alpha_prior_strength
beta_prior = (alpha_prior / target_prior_mean) - alpha_prior

print(f"Control conversion rate: {control_group_conversion_rate:.4f}")
print(f"Non-inferiority threshold: {control_group_conversion_rate - epsilon:.4f}")
print(f"\nPrior parameters:")
print(f"  α = {alpha_prior:.2f}")
print(f"  β = {beta_prior:.2f}")
print(f"  mean = {target_prior_mean:.4f}")

# Compute posteriors for all variants
variants_posteriors = {}
for name, data in variants_data.items():
    alpha_post = data['x'] + alpha_prior
    beta_post = (data['n'] - data['x']) + beta_prior

    variants_posteriors[name] = {
        'alpha': alpha_post,
        'beta': beta_post,
        'n': data['n'],
        'x': data['x']
    }

    post_mean = alpha_post / (alpha_post + beta_post)
    print(f"\nVariant {name}:")
    print(f"  Observed: {data['x']}/{data['n']} = {data['x']/data['n']:.4f}")
    print(f"  Posterior: Beta({alpha_post:.1f}, {beta_post:.1f})")
    print(f"  Posterior mean: {post_mean:.4f}")

# Create the plot
print("\n" + "="*80)
print("Creating plot with prior and all variant posteriors...")
print("="*80)

fig, ax = plot_weakly_informative_prior_with_variants(
    alpha_prior=alpha_prior,
    beta_prior=beta_prior,
    variants_posteriors=variants_posteriors,
    threshold=control_group_conversion_rate - epsilon,
    control_rate=control_group_conversion_rate,
    epsilon=epsilon
)

# Save the figure
output_file = 'weakly_informative_prior_all_variants.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Show the plot
plt.show()

print("\nPlot features:")
print("  • Gray dashed line: Common weakly informative prior for all variants")
print("  • Colored solid lines: Posterior distributions for each variant")
print("  • Red dotted line: Non-inferiority threshold (control - ε)")
print("  • Black dash-dot line: Control group conversion rate")
print("  • Text box: Probability each variant exceeds threshold (✓ if ≥95%)")
