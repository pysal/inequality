"""
Wolfson Bipolarization Index Module

This module provides functions to calculate the Lorenz curve, Gini coefficient,
and Wolfson Bipolarization Index for a given distribution of income or wealth.

Author:
Serge Rey <srey@sdsu.edu>
"""

import numpy as np


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


def gini_coefficient(lorenz_curve):
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
    lorenz_area = np.trapezoid(lorenz_curve[1], lorenz_curve[0])
    return 1 - 2 * lorenz_area


def wolfson(y):
    """
    Calculate the Wolfson Bipolarization Index for a given distribution.
    The Wolfson Index is a measure of income polarization, which
    considers both the spread and the skewness of the income
    distribution, focusing on the middle class.


    Parameters:
    y (array-like): A list or array of income or wealth values.

    Returns:
    float: The Wolfson Bipolarization Index. A higher value indicates
       more significant polarization in the income distribution.

    Example:
    --------
    >>> income_distribution = [20000, 25000, 27000, 30000,
                                35000, 45000, 60000, 75000, 80000, 120000]
    >>> wolfson_index = wolfson(income_distribution)
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.1213
    """
    y = np.array(y)
    cumulative_population, cumulative_y = lorenz_curve(y)
    g = gini_coefficient((cumulative_population, cumulative_y))

    median_y = np.median(y)
    mean_y = np.mean(y)

    median_lorenz = np.interp(0.5, cumulative_population, cumulative_y)

    d50 = 0.5 - median_lorenz
    return (2 * d50 - g) * (mean_y / median_y)
