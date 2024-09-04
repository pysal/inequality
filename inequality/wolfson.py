"""
Wolfson Bipolarization Index Module

This module provides functions to calculate the Lorenz curve, Gini coefficient,
and Wolfson Bipolarization Index for a given distribution of income or wealth.

Author:
Serge Rey <srey@sdsu.edu>
"""

import numpy as np

from .gini import Gini

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


def wolfson(income_distribution):
    """
    Calculate the Wolfson Bipolarization Index for a given income distribution.


    The Wolfson index is constructed from the polarization curve,
    which is a rotation and rescaling of the Lorenz curve by the
    median income:

    .. math::

       W = (2D_{50} - G)\\frac{\\mu}{m}

    Where :math:`D_{50} =0.5 - L(0.5)`, :math:`L(0.5)` is the
    value of the Lorenz curve at the median, :math:`G` is the Gini
    index, :math:`\mu` is the mean, and :math:`m` is the median.

    
    See :cite:t:`wolfson1994WhenInequalities,hoffmann2024MeasuringMismeasuring`.

    

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
    >>> income_distribution = [6, 6, 8, 8, 10, 10, 12, 12]
    >>> wolfson_index = wolfson(income_distribution)
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.0833
    >>> income_distribution = [2, 4, 6, 8, 10, 12, 14, 16]
    >>> wolfson_index = wolfson(income_distribution)
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.1528

    """
    y = np.array(income_distribution)
    y_med = np.median(y)
    ordinate, lc = lorenz_curve(y)
    l50 = np.interp(.5, ordinate, lc)
    d50 = .5 - l50
    rat = y.mean() / y_med
    g = Gini(y).g
    w = (2 * d50 - g) * rat

    return w
