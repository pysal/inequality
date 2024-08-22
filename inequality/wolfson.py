"""
Wolfson Bipolarization Index Module

This module provides functions to calculate the Lorenz curve, Gini coefficient,
and Wolfson Bipolarization Index for a given distribution of income or wealth.

Author:
Serge Rey <srey@sdsu.edu>
"""

import numpy as np

__all__ = ["wolfson", "lorenz_curve"]


def lorenz_curve(y):
    """
    Calculate the Lorenz curve for a given distribution.

    Parameters:
    y (array-like): A list or array of income or wealth values.

    Returns:
    tuple: Two numpy arrays representing the cumulative share of the population
           and the cumulative share of the income/wealth.
    """
    sorted_y = np.sort(y)
    cumulative_y = np.cumsum(sorted_y)
    # Add zero for the starting point
    cumulative_y = np.insert(cumulative_y, 0, 0)
    cumulative_y = cumulative_y / cumulative_y[-1]
    cumulative_population = np.linspace(0, 1, len(y) + 1)
    return cumulative_population, cumulative_y


def _gini_coefficient(lorenz_curve):
    """
    Calculate the Gini coefficient from the Lorenz curve.

    Parameters:
    lorenz_curve (tuple): A tuple containing two numpy arrays representing the
                          cumulative share of the population and the cumulative
                          share of the income/wealth.

    Returns:
    float: The Gini coefficient, a measure of inequality ranging from
           0 (perfect equality) to 1 (perfect inequality).

    """
    lorenz_area = np.trapz(lorenz_curve[1], lorenz_curve[0])
    return 1 - 2 * lorenz_area


def wolfson(income_distribution):
    """
    Calculate the Wolfson Bipolarization Index for a given income distribution.

    Parameters
    ----------
    income_distribution : list of int or float
        A list representing the income distribution.

    Returns
    -------
    w: float
        The Wolfson Bipolarization Index value.

    Example
    -------
    >>> income_distribution = [20000, 25000, 27000, 30000, 35000, 45000, 60000,
    ...                        75000, 80000, 120000]
    >>> wolfson_index = wolfson(income_distribution)
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.2013
    """
    income_distribution = np.sort(income_distribution)
    n = len(income_distribution)
    total_income = np.sum(income_distribution)
    mean_income = total_income / n
    cumulative_income = np.cumsum(income_distribution)

    # Calculate the Gini coefficient
    g = 1 - 2 * np.sum(cumulative_income) / (n * total_income) + (n + 1) / n

    # Calculate the Wolfson Bipolarization Index
    w = 2 * mean_income * g / total_income - 1

    return w
