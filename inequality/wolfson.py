"""
Wolfson Bipolarization Index Module

This module provides functions to calculate the Lorenz curve, Gini coefficient,
and Wolfson Bipolarization Index for a given distribution of income or wealth.

Author:
Serge Rey <srey@sdsu.edu>
"""

import numpy as np

from .gini import Gini
from .utils import consistent_input

__all__ = ["wolfson", "lorenz_curve"]


@consistent_input
def lorenz_curve(data):
    """
    Calculate the Lorenz curve for a given distribution.

    This function takes an income or wealth distribution as input. The input
    can be a sequence, a NumPy array, or a Pandas DataFrame. If a DataFrame
    is provided, the `column` parameter must be used to specify which column
    contains the income or wealth values.

    Parameters
    ----------
    data : array-like or array
        A sequence or NumPy array representing the income or
        wealth distribution.

    Returns
    -------
    tuple
        Two numpy arrays: the first represents the cumulative share of the
        population, and the second represents the cumulative share of
        the income/wealth.

    Example
    -------
    >>> income = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    >>> population, income_share = lorenz_curve(income)
    >>> print(population[:2], income_share[:2])
    [0.  0.1] [0.         0.03868472]
    """
    sorted_y = np.sort(data)
    cumulative_y = np.cumsum(sorted_y)
    cumulative_y = np.insert(cumulative_y, 0, 0)
    cumulative_y = cumulative_y / cumulative_y[-1]
    cumulative_population = np.linspace(0, 1, len(data) + 1)
    return cumulative_population, cumulative_y


@consistent_input
def wolfson(data):
    """
    Calculate the Wolfson Bipolarization Index for a given income distribution.

    This function takes an income distribution and calculates the Wolfson
    Bipolarization Index. The input can be a sequence or a NumPy array.
    The Wolfson index is constructed from the polarization curve, which is
    a rotation and rescaling of the Lorenz curve by the median income:

    .. math::

       W = (2D_{50} - G)\\frac{\\mu}{m}

    Where :math:`D_{50} =0.5 - L(0.5)`, :math:`L(0.5)` is the value of the
    Lorenz curve at the median, :math:`G` is the Gini index, :math:`\\mu`
    is the mean, and :math:`m` is the median.

    See: :cite:`wolfson1994WhenInequalities`.

    Parameters
    ----------
    data : array-like or array
        A sequence or NumPy array representing the income or
        wealth distribution.

    Returns
    -------
    float
        The Wolfson Bipolarization Index value.

    Example
    -------
    >>> import pandas as pd
    >>> income_distribution = [20000, 25000, 27000, 30000, 35000, 45000, 60000,
    ...                        75000, 80000, 120000]
    >>> wolfson_index = wolfson(income_distribution)
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.2013

    >>> df = pd.DataFrame({'income': [6, 6, 8, 8, 10, 10, 12, 12]})
    >>> wolfson_index = wolfson(df, column='income')
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.0833
    """
    y = np.array(data)
    y_med = np.median(y)
    ordinate, lc = lorenz_curve(y)
    l50 = np.interp(0.5, ordinate, lc)
    d50 = 0.5 - l50
    rat = y.mean() / y_med
    g = Gini(y).g
    w = (2 * d50 - g) * rat

    return w
