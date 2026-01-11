"""
Root pytest configuration for ABtest_methodologies.

This file prevents pytest from collecting implementation functions
that are incorrectly named with "test_" prefix.

Why this is needed:
- bayesian/utils.py has functions named "test_non_inferiority_*"
- These are IMPLEMENTATION functions, not test functions
- When test files import these functions, pytest tries to collect them
- This configuration explicitly excludes them from collection
"""

import pytest


def pytest_collection_modifyitems(session, config, items):
    """
    Modify collected test items to remove false positives.

    This hook runs after pytest collects all test items but before
    running them. We use it to filter out implementation functions
    that were incorrectly collected due to "test_" naming.

    Why we need this:
    - bayesian/utils.py::test_non_inferiority_weakly_informative
    - bayesian/utils.py::test_non_inferiority (if it exists)
    - These are imported by our real tests but should not be run as tests
    """
    # Build list of items to keep
    items_to_keep = []

    for item in items:
        # Get the actual source file where the function is DEFINED
        # (not where it's imported)
        try:
            # Why we check fspath:
            # - item.fspath is the actual file where the function is defined
            # - item.nodeid is where pytest found it (could be an import)
            # - We want to exclude functions defined outside tests/ directory
            actual_source_file = str(item.fspath)

            # Only keep items actually defined in tests/ directory
            # This excludes imported functions from bayesian/utils.py
            if '/tests/' in actual_source_file or '\\tests\\' in actual_source_file:
                # Additional check: exclude specific problematic function names
                # that are imports from implementation modules
                function_name = item.name

                # Why this exclusion list:
                # - test_non_inferiority_weakly_informative is from bayesian/utils.py
                # - test_non_inferiority is also from bayesian/utils.py
                # - These are imported, not defined in test files
                excluded_function_names = {
                    'test_non_inferiority_weakly_informative',
                    'test_non_inferiority',
                }

                if function_name not in excluded_function_names:
                    items_to_keep.append(item)
        except AttributeError:
            # If we can't get fspath, keep the item to be safe
            items_to_keep.append(item)

    # Replace items list with filtered list
    # Why we modify in-place:
    # - pytest expects us to modify the items list, not replace it
    # - We clear and re-add rather than reassigning
    items[:] = items_to_keep


# Alternative approach: Explicitly ignore directories during collection
collect_ignore = [
    'bayesian',
    'nhst',
    'rendered',
    'plotting_utils.py'
]
