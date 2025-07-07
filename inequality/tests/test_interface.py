import numpy as np
import pandas as pd

from inequality.wolfson import lorenz_curve, wolfson


def test_lorenz_curve_with_array():
    income = np.array(
        [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    )
    population, cumulative_income = lorenz_curve(income)

    # Check that both returned arrays have the correct length (n+1)
    assert len(population) == len(income) + 1
    assert len(cumulative_income) == len(income) + 1

    # Ensure that the Lorenz curve starts at zero
    assert cumulative_income[0] == 0.0
    assert population[0] == 0.0


def test_lorenz_curve_with_list():
    income = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    population, cumulative_income = lorenz_curve(income)

    # Check that both returned arrays have the correct length (n+1)
    assert len(population) == len(income) + 1
    assert len(cumulative_income) == len(income) + 1

    # Ensure that the Lorenz curve starts at zero
    assert cumulative_income[0] == 0.0
    assert population[0] == 0.0


def test_lorenz_curve_with_dataframe():
    df = pd.DataFrame(
        {
            "income": [
                20000,
                25000,
                27000,
                30000,
                35000,
                45000,
                60000,
                75000,
                80000,
                120000,
            ]
        }
    )
    population, cumulative_income = lorenz_curve(df, column="income")

    # Check that both returned arrays have the correct length (n+1)
    assert len(population) == len(df["income"]) + 1
    assert len(cumulative_income) == len(df["income"]) + 1

    # Ensure that the Lorenz curve starts at zero
    assert cumulative_income[0] == 0.0
    assert population[0] == 0.0


def test_wolfson_with_array():
    income = np.array(
        [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    )
    wolfson_index = wolfson(income)

    # Compare the result to an expected value (based on the example)
    assert np.isclose(wolfson_index, 0.2013, atol=1e-4)


def test_wolfson_with_list():
    income = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    wolfson_index = wolfson(income)

    # Compare the result to an expected value (based on the example)
    assert np.isclose(wolfson_index, 0.2013, atol=1e-4)


def test_wolfson_with_dataframe():
    df = pd.DataFrame(
        {
            "income": [
                20000,
                25000,
                27000,
                30000,
                35000,
                45000,
                60000,
                75000,
                80000,
                120000,
            ]
        }
    )
    wolfson_index = wolfson(df, column="income")

    # Compare the result to an expected value (based on the example)
    assert np.isclose(wolfson_index, 0.2013, atol=1e-4)


def test_wolfson_with_small_dataset():
    income = [6, 6, 8, 8, 10, 10, 12, 12]
    wolfson_index = wolfson(income)

    # Compare the result to an expected value (based on the example)
    assert np.isclose(wolfson_index, 0.0833, atol=1e-4)


def test_wolfson_with_even_distribution():
    income = [2, 4, 6, 8, 10, 12, 14, 16]
    wolfson_index = wolfson(income)

    # Compare the result to an expected value (based on the example)
    assert np.isclose(wolfson_index, 0.1528, atol=1e-4)
