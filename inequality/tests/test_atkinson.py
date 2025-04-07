import numpy as np
import pytest
from inequality.atkinson import Atkinson, atkinson


def testatkinson_function():
    # Test case for epsilon = 0.5
    incomes = np.array([10, 20, 30, 40, 50])
    result = atkinson(incomes, 0.5)
    expected = 0.06315
    assert np.isclose(result, expected, atol=1e-5), (
        f"Expected {expected}, but got {result}"
    )

    # Test case for epsilon = 1
    result = atkinson(incomes, 1)
    expected = 0.1316096
    assert np.isclose(result, expected, atol=1e-5), (
        f"Expected {expected}, but got {result}"
    )

    # Test case for epsilon = 0
    result = atkinson(incomes, 0)
    expected = 0
    assert np.isclose(result, expected, atol=1e-5), (
        f"Expected {expected}, but got {result}"
    )


def testatkinson_class():
    # Test case for epsilon = 0.5
    incomes = np.array([10, 20, 30, 40, 50])
    atkinson = Atkinson(incomes, 0.5)
    expected_A = 0.06315
    expected_EDE = 28.105398233
    assert np.isclose(atkinson.A, expected_A, atol=1e-5), (
        f"Expected Atkinson index {expected_A}, but got {atkinson.A}"
    )
    assert np.isclose(atkinson.EDE, expected_EDE, atol=1e-5), (
        f"Expected EDE {expected_EDE}, but got {atkinson.EDE}"
    )

    # Test case for epsilon = 1
    atkinson = Atkinson(incomes, 1)
    expected_A = 0.1316096
    expected_EDE = 26.0517108
    assert np.isclose(atkinson.A, expected_A, atol=1e-5), (
        f"Expected Atkinson index {expected_A}, but got {atkinson.A}"
    )
    assert np.isclose(atkinson.EDE, expected_EDE, atol=1e-5), (
        f"Expected EDE {expected_EDE}, but got {atkinson.EDE}"
    )

    # Test case for epsilon = 0
    atkinson = Atkinson(incomes, 0)
    expected_A = 0
    expected_EDE = incomes.mean()
    assert np.isclose(atkinson.A, expected_A, atol=1e-5), (
        f"Expected Atkinson index {expected_A}, but got {atkinson.A}"
    )
    assert np.isclose(atkinson.EDE, expected_EDE, atol=1e-5), (
        f"Expected EDE {expected_EDE}, but got {atkinson.EDE}"
    )


if __name__ == "__main__":
    pytest.main()
