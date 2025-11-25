"""
Test script for the improved plotting function that shows weakly informative
prior and all variant posteriors together.

This demonstrates the simplified usage where you can pass the output of
test_non_inferiority_weakly_informative() directly to the plotting function.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from bayesian import test_non_inferiority_weakly_informative
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

print("="*80)
print("BAYESIAN NON-INFERIORITY TEST WITH WEAKLY INFORMATIVE PRIOR")
print("="*80)

print(f"\nControl group:")
print(f"  n = {nC:,}")
print(f"  x = {xC_observed:,}")
print(f"  Conversion rate: {control_group_conversion_rate:.4f} ({control_group_conversion_rate*100:.2f}%)")

print(f"\nTest parameters:")
print(f"  Non-inferiority margin (ε): {epsilon:.2%}")
print(f"  Non-inferiority threshold: {control_group_conversion_rate - epsilon:.4f}")
print(f"  Required probability: 95%")

print(f"\nVariants:")
for name, data in variants_data.items():
    rate = data['x'] / data['n']
    print(f"  {name}: n={data['n']:3d}, x={data['x']:3d}, rate={rate:.4f} ({rate*100:.2f}%)")

# Run the test
print("\n" + "="*80)
print("Running test_non_inferiority_weakly_informative...")
print("="*80)

results = test_non_inferiority_weakly_informative(
    n_control=nC,
    x_control=xC_observed,
    variants_data=variants_data,
    epsilon=epsilon,
    alpha_prior_strength=alpha_prior_strength
)

print("\nResults:")
for variant_name, result in results.items():
    status = "✓ NON-INFERIOR" if result['is_non_inferior'] else "✗ NOT NON-INFERIOR"
    print(f"\n  Variant {variant_name}: {status}")
    print(f"    Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    print(f"    Posterior mean: {result['variant_rate']:.4f}")

# Create the plot - NOW SIMPLIFIED!
print("\n" + "="*80)
print("Creating plot with prior and all variant posteriors...")
print("="*80)
print("\nSimplified usage: Just pass the results directly!")
print("  fig, ax = plot_weakly_informative_prior_with_variants(results)")

fig, ax = plot_weakly_informative_prior_with_variants(results)

# Save the figure
output_file = 'weakly_informative_prior_all_variants.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

print("\n" + "="*80)
print("KEY BENEFITS")
print("="*80)
print("""
1. SIMPLIFIED USAGE:
   - Just pass test results directly to plotting function
   - No need to manually extract alpha_prior, beta_prior, threshold, etc.
   - All parameters auto-detected from results

2. COMPREHENSIVE VISUALIZATION:
   - Gray dashed line: Common weakly informative prior
   - Colored solid lines: Posterior for each variant
   - Red dotted line: Non-inferiority threshold
   - Black dash-dot line: Control rate
   - Text box: P(variant > threshold) with ✓/✗ indicators

3. BACKWARDS COMPATIBLE:
   - Still supports legacy manual format if needed
   - Automatically detects which format you're using
""")
