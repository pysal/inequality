import numpy as np
from inequality.wolfson import lorenz_curve, gini_coefficient, wolfson


def test_lorenz_curve():
    income = [1, 2, 3, 4, 5]
    population, cumulative_income = lorenz_curve(income)

    # Expected cumulative income values (calculated manually)
    expected_cumulative_income = np.array(
        [0, 0.06666667, 0.2, 0.4, 0.66666667, 1])

    # Test if the Lorenz curve is calculated correctly
    np.testing.assert_almost_equal(
        cumulative_income, expected_cumulative_income, decimal=6)
    # Should include start and end points (0 and 1)
    assert len(population) == 6


def test_gini_coefficient():
    # Lorenz curve values (population, cumulative income)
    population = np.array([0, 0.25, 0.5, 0.75, 1])
    cumulative_income = np.array([0, 0.1, 0.3, 0.6, 1])
    g = gini_coefficient((population, cumulative_income))
    expected_gini = 0.25
    assert np.isclose(g, expected_gini, atol=0.01)


def test_wolfson():
    # Test with a known income distribution
    income = [6, 6, 8, 8, 10, 10, 12, 12]
    wolfson_idx = wolfson(income)
    expected_wolfson_idx = 1/12
    assert np.isclose(wolfson_idx, expected_wolfson_idx, atol=0.01)
    income = [2, 4, 6, 8, 10, 12, 14, 16]
    wolfson_idx = wolfson(income)
    expected_wolfson_idx = 11/72
    assert np.isclose(wolfson_idx, expected_wolfson_idx, atol=0.01)
