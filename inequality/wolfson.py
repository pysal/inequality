"""
Wolfson Bipolarization Index Module

This module provides functions to calculate the Lorenz curve, Gini coefficient,
and Wolfson Bipolarization Index for a given distribution of income or wealth.

Author:
Serge Rey <srey@sdsu.edu>
"""

import warnings

import numpy as np

from .gini import Gini
from .utils import _resolve_array

__all__ = ["wolfson", "lorenz_curve"]


def _resolve_deprecated_column(data, column, func_name):
    if column is not None:
        warnings.warn(
            f"The 'column' argument is deprecated and will be removed in a "
            f"future release; pass the column directly instead, e.g. "
            f"{func_name}(df['{column}']).",
            FutureWarning,
            stacklevel=3,
        )
        data = data[column]
    return _resolve_array(data)


def lorenz_curve(data, *, column=None):
    """
    Calculate the Lorenz curve for a given distribution.

    This function takes an income or wealth distribution as input. The input
    can be any array-like (a sequence, a NumPy array, a pandas Series, ...).
    For a single column of a :class:`pandas.DataFrame`, pass it as a Series,
    e.g. ``lorenz_curve(df["col"])``.

    Parameters
    ----------
    data : array-like
        A sequence, NumPy array, or pandas Series representing the income or
        wealth distribution.
    column : str, optional
        Deprecated. Selecting a column of a :class:`pandas.DataFrame` by name
        instead of passing ``data`` as a Series/array directly. Will be
        removed in a future release.

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
    data = _resolve_deprecated_column(data, column, "lorenz_curve")
    sorted_y = np.sort(data)
    cumulative_y = np.cumsum(sorted_y)
    cumulative_y = np.insert(cumulative_y, 0, 0)
    cumulative_y = cumulative_y / cumulative_y[-1]
    cumulative_population = np.linspace(0, 1, len(data) + 1)
    return cumulative_population, cumulative_y


def wolfson(data, *, column=None):
    """
    Calculate the Wolfson Bipolarization Index for a given income distribution.

    This function takes an income distribution and calculates the Wolfson
    Bipolarization Index. The input can be any array-like (a sequence, a
    NumPy array, a pandas Series, ...). The Wolfson index is constructed
    from the polarization curve, which is a rotation and rescaling of the
    Lorenz curve by the median income:

    .. math::

       W = (2D_{50} - G)\\frac{\\mu}{m}

    Where :math:`D_{50} =0.5 - L(0.5)`, :math:`L(0.5)` is the value of the
    Lorenz curve at the median, :math:`G` is the Gini index, :math:`\\mu`
    is the mean, and :math:`m` is the median.

    See: :cite:`wolfson1994WhenInequalities`.

    Parameters
    ----------
    data : array-like
        A sequence, NumPy array, or pandas Series representing the income or
        wealth distribution.
    column : str, optional
        Deprecated. Selecting a column of a :class:`pandas.DataFrame` by name
        instead of passing ``data`` as a Series/array directly. Will be
        removed in a future release.

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
    >>> wolfson_index = wolfson(df["income"])
    >>> print(f"Wolfson Bipolarization Index: {wolfson_index:.4f}")
    Wolfson Bipolarization Index: 0.0833
    """
    y = _resolve_deprecated_column(data, column, "wolfson")
    y_med = np.median(y)
    ordinate, lc = lorenz_curve(y)
    l50 = np.interp(0.5, ordinate, lc)
    d50 = 0.5 - l50
    rat = y.mean() / y_med
    g = Gini(y).g
    w = (2 * d50 - g) * rat

    return w
