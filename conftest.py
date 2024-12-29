# ----- IMPORTANT -----
# See GL#80 & GL#81
# This file can be deleted once ``_indices.py`` is fully removed.

import pytest


def warning_depr(x):
    return pytest.warns(
        FutureWarning,
        match=f"{x} is deprecated and will be removed on 2025-01-01."
    )


def warning_invalid(x):
    return pytest.warns(RuntimeWarning,
                        match=f"invalid value encountered in {x}")


warning_div_zero = pytest.warns(RuntimeWarning,
                                match="divide by zero encountered")


def pytest_configure():
    pytest.warning_depr = warning_depr
    pytest.warning_invalid = warning_invalid
    pytest.warning_div_zero = warning_div_zero


def pytest_ignore_collect(collection_path):
    return bool(str(collection_path).endswith("_indices.py"))
