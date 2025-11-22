"""
Test script for the weakly informative prior non-inferiority function.

This script validates that the new function produces results consistent
with the methodology described in the notebook.
"""

from bayesian import test_non_inferiority_weakly_informative

# Data from the notebook example
nC = 4411
xC_observed = 3138
control_group_conversion_rate = xC_observed / nC

# Three variants with actual experiment data
variants = {
    'A': {'n': 561, 'x': 381},
    'B': {'n': 285, 'x': 192},
    'C': {'n': 294, 'x': 201}
}

# Test parameters
epsilon = 0.05  # 5% non-inferiority margin
alpha = 0.05    # 5% significance level (not used in Bayesian, but for reference)

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
for name, data in variants.items():
    rate = data['x'] / data['n']
    print(f"  {name}: n={data['n']:3d}, x={data['x']:3d}, rate={rate:.4f} ({rate*100:.2f}%)")

# Run the test
results = test_non_inferiority_weakly_informative(
    n_control=nC,
    x_control=xC_observed,
    variants_data=variants,
    epsilon=epsilon,
    alpha_prior_strength=20,  # Same as notebook
    threshold=0.95
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

for variant_name, result in results.items():
    print(f"\n{'─'*80}")
    print(f"Variant {variant_name}:")
    print(f"{'─'*80}")
    print(f"  Prior parameters: α={result['prior_params'][0]:.2f}, β={result['prior_params'][1]:.2f}")
    print(f"  Prior mean: {result['prior_mean']:.4f}")
    print(f"  Posterior parameters: α={result['posterior_params'][0]:.2f}, β={result['posterior_params'][1]:.2f}")
    print(f"  Posterior mean (variant rate): {result['variant_rate']:.4f}")
    print(f"  P(variant > control - ε): {result['probability']:.4f} ({result['probability']*100:.2f}%)")

    if result['is_non_inferior']:
        print(f"  ✓ NON-INFERIOR (probability {result['probability']:.3f} >= 0.95)")
    else:
        print(f"  ✗ NOT NON-INFERIOR (probability {result['probability']:.3f} < 0.95)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

non_inferior_count = sum(1 for r in results.values() if r['is_non_inferior'])
print(f"\n{non_inferior_count}/{len(variants)} variants are non-inferior")

if non_inferior_count == len(variants):
    print("\n✓ All variants pass non-inferiority test!")
    print("  You can proceed to select the best variant.")
elif non_inferior_count > 0:
    print(f"\n⚠ Only some variants are non-inferior.")
    print(f"  Consider removing variants that failed:")
    for name, result in results.items():
        if not result['is_non_inferior']:
            print(f"    - Variant {name}")
else:
    print("\n✗ No variants are non-inferior.")
    print("  All variants may degrade performance too much.")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
The weakly informative prior approach:
1. Uses historical control data (71.13% conversion) to set expectations
2. Centers prior at control - ε = 66.13% (what we'd minimally accept)
3. Maintains high uncertainty (α=20) to let data dominate
4. Provides actionable probabilities even with small samples (n≈300-600)

Compare to NHST: With these sample sizes, traditional tests would be
severely underpowered and likely fail to reach significance.
""")
