"""
Pytest suite for plotting weakly informative prior with variant posteriors.

This test suite validates that the plotting function:
1. Executes without errors when given valid test results
2. Returns proper matplotlib Figure and Axes objects
3. Can save plots to files successfully
4. Handles the simplified API (passing test results directly)

---
🧠 THE LOGIC BLUEPRINT
---
For plotting tests, we:
1. SETUP: Generate test results using the Bayesian test function
2. EXECUTE: Call the plotting function with those results
3. VALIDATE: Assert that matplotlib objects are created correctly
4. VERIFY: Check that saved files exist and are non-empty
5. CLEANUP: Use pytest's tmp_path fixture for automatic cleanup

Key validations:
- Function returns Figure and Axes objects (not None)
- Axes contains plot elements (lines, labels, etc.)
- Saved PNG files are created and have reasonable file sizes
- No exceptions are raised during plotting
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Why non-interactive backend:
# - Tests run in environments without display (CI/CD, containers)
# - Agg backend renders to memory/files without GUI
# - Must be set before importing pyplot
matplotlib.use('Agg')

# Why we import modules instead of functions with "test_" prefix:
# - A/B testing functions are correctly named test_* (they test hypotheses)
# - But pytest collects any "test_*" in a test file's namespace
# - Qualified imports keep these functions out of the test namespace
# - This prevents pytest from trying to run domain functions as pytest tests
import bayesian.utils as bayesian_utils
import plotting_utils


@pytest.fixture
def bayesian_test_results(
    control_group_data: Dict[str, int],
    variants_experiment_data: Dict[str, Dict[str, int]],
    non_inferiority_test_parameters: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Generate real Bayesian test results for plotting tests.

    Why this fixture:
        - Plotting tests need actual test results as input
        - Reusing the real function ensures integration testing
        - Returns the same data structure that users will actually pass
    """
    control_total_users: int = control_group_data['n']
    control_converted_users: int = control_group_data['x']
    epsilon_margin: float = non_inferiority_test_parameters['epsilon']
    prior_strength: int = non_inferiority_test_parameters['alpha_prior_strength']
    probability_threshold: float = non_inferiority_test_parameters['threshold']

    results: Dict[str, Dict[str, Any]] = bayesian_utils.test_non_inferiority_weakly_informative(
        n_control=control_total_users,
        x_control=control_converted_users,
        variants_data=variants_experiment_data,
        epsilon=epsilon_margin,
        alpha_prior_strength=prior_strength,
        threshold=probability_threshold
    )

    return results


def test_plotting_function_returns_matplotlib_objects(
    bayesian_test_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test that plotting function returns Figure and Axes objects.

    Why this test:
        - Users may want to customize plots after creation
        - Returning Figure and Axes is the matplotlib convention
        - None return values would indicate function failure
    """
    # ARRANGE: Test results are already prepared by fixture

    # ACT: Call the plotting function
    returned_figure, returned_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
        bayesian_test_results
    )

    # ASSERT: Returns are the correct matplotlib types
    assert isinstance(returned_figure, Figure), (
        f"Expected Figure object, but got {type(returned_figure)}"
    )
    assert isinstance(returned_axes, Axes), (
        f"Expected Axes object, but got {type(returned_axes)}"
    )

    # Cleanup: Close the figure to free memory
    plt.close(returned_figure)


def test_plotting_function_creates_non_empty_axes(
    bayesian_test_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test that the plotting function actually draws content on the axes.

    Why this test:
        - An empty axes would indicate the plot logic failed silently
        - We expect lines (prior, posteriors, thresholds)
        - We expect text (legend, labels, annotations)
    """
    # ARRANGE
    # (Test results prepared by fixture)

    # ACT
    plot_figure, plot_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
        bayesian_test_results
    )

    # ASSERT: Axes contains plot elements

    # Check that lines were drawn (prior, posteriors, thresholds)
    drawn_lines: list = plot_axes.get_lines()
    number_of_lines: int = len(drawn_lines)

    # Why at least 5 lines:
    # - 1 prior line (gray dashed)
    # - 3 variant posteriors (one per variant A, B, C)
    # - 1 threshold line (red dotted)
    # - 1 control rate line (black dash-dot)
    # Total minimum: 6 lines, but we check for >= 5 to be slightly lenient
    minimum_expected_lines: int = 5

    assert number_of_lines >= minimum_expected_lines, (
        f"Expected at least {minimum_expected_lines} lines on plot, "
        f"but found only {number_of_lines}"
    )

    # Check that axes has a title
    plot_title: str = plot_axes.get_title()
    assert plot_title != "", "Plot should have a non-empty title"

    # Check that axes has x-label
    x_label: str = plot_axes.get_xlabel()
    assert x_label != "", "Plot should have a non-empty x-axis label"

    # Check that axes has y-label
    y_label: str = plot_axes.get_ylabel()
    assert y_label != "", "Plot should have a non-empty y-axis label"

    # Cleanup
    plt.close(plot_figure)


def test_plotting_function_saves_file_successfully(
    bayesian_test_results: Dict[str, Dict[str, Any]],
    tmp_path: Path
) -> None:
    """
    Test that the plot can be saved to a PNG file.

    Why this test:
        - Users need to save plots for reports and presentations
        - File I/O can fail for various reasons (permissions, disk space)
        - We validate the full workflow: create plot → save → verify file exists

    🔍 VERIFICATION NOTE:
        This test uses pytest's tmp_path fixture which:
        - Creates a unique temporary directory for each test
        - Automatically cleans up after test completion
        - Prevents test pollution and disk space leaks
    """
    # ARRANGE: Create output file path in temporary directory
    output_file_path: Path = tmp_path / "test_weakly_informative_plot.png"

    # ACT: Create plot and save to file
    plot_figure, plot_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
        bayesian_test_results
    )

    # Save the figure to the temporary file
    plot_figure.savefig(
        output_file_path,
        dpi=150,
        bbox_inches='tight'
    )

    # ASSERT: File was created
    assert output_file_path.exists(), (
        f"Plot file should exist at {output_file_path}, but it does not"
    )

    # ASSERT: File has reasonable size (not empty, not suspiciously small)
    file_size_bytes: int = output_file_path.stat().st_size
    minimum_reasonable_size_bytes: int = 1000  # 1 KB minimum for a real PNG

    assert file_size_bytes > minimum_reasonable_size_bytes, (
        f"Plot file is only {file_size_bytes} bytes, "
        f"which is suspiciously small (expected > {minimum_reasonable_size_bytes} bytes). "
        f"The file may be empty or corrupted."
    )

    # Cleanup
    plt.close(plot_figure)


def test_plotting_function_handles_all_variants(
    bayesian_test_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test that the plot includes a line for each variant in the results.

    Why this test:
        - Ensures no variants are silently dropped from visualization
        - Users expect one posterior line per variant
        - Missing variants would lead to incomplete analysis
    """
    # ARRANGE
    number_of_variants: int = len(bayesian_test_results)

    # ACT
    plot_figure, plot_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
        bayesian_test_results
    )

    # ASSERT: We should have lines for all variants plus auxiliary lines

    drawn_lines: list = plot_axes.get_lines()
    total_lines_drawn: int = len(drawn_lines)

    # Why this calculation:
    # - number_of_variants posterior lines (one per variant)
    # - 1 prior line (shared by all variants)
    # - 1 threshold line (non-inferiority boundary)
    # - 1 control rate line (reference)
    expected_minimum_lines: int = number_of_variants + 3

    assert total_lines_drawn >= expected_minimum_lines, (
        f"For {number_of_variants} variants, expected at least {expected_minimum_lines} lines "
        f"({number_of_variants} posteriors + 1 prior + 1 threshold + 1 control), "
        f"but found {total_lines_drawn}"
    )

    # Cleanup
    plt.close(plot_figure)


def test_plotting_function_does_not_raise_exceptions(
    bayesian_test_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test that plotting completes without raising any exceptions.

    Why this test:
        - Explicit test that the function doesn't crash
        - Complements other tests by focusing on execution stability
        - Any exception should fail this test clearly
    """
    # ARRANGE
    # (Test results prepared by fixture)

    # ACT & ASSERT: Function should not raise any exceptions
    try:
        plot_figure, plot_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
            bayesian_test_results
        )

        # If we got here, no exception was raised
        execution_successful: bool = True

    except Exception as error:
        # If any exception occurs, fail the test with details
        pytest.fail(
            f"Plotting function raised an unexpected exception: {type(error).__name__}: {error}"
        )

    finally:
        # Cleanup: Close any figures that may have been created
        plt.close('all')

    assert execution_successful, "Plotting function should execute without exceptions"


def test_plotting_with_custom_figure_size(
    bayesian_test_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Test that the figsize parameter controls the figure dimensions.

    Why this test:
        - Users may need different sizes for papers, slides, or posters
        - Validates the figsize parameter is respected
        - Ensures customization options work correctly
    """
    # ARRANGE: Define a custom figure size
    custom_width_inches: float = 10.0
    custom_height_inches: float = 6.0
    custom_figsize: Tuple[float, float] = (custom_width_inches, custom_height_inches)

    # ACT: Create plot with custom size
    plot_figure, plot_axes = plotting_utils.plot_weakly_informative_prior_with_variants(
        bayesian_test_results,
        figsize=custom_figsize
    )

    # ASSERT: Figure has the requested dimensions
    actual_width_inches: float = plot_figure.get_figwidth()
    actual_height_inches: float = plot_figure.get_figheight()

    # Why use pytest.approx:
    # - Floating point comparisons can have tiny rounding errors
    # - pytest.approx handles this with a small tolerance (default 1e-6)
    assert actual_width_inches == pytest.approx(custom_width_inches), (
        f"Expected figure width {custom_width_inches} inches, "
        f"but got {actual_width_inches} inches"
    )
    assert actual_height_inches == pytest.approx(custom_height_inches), (
        f"Expected figure height {custom_height_inches} inches, "
        f"but got {actual_height_inches} inches"
    )

    # Cleanup
    plt.close(plot_figure)
