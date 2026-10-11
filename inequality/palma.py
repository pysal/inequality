"""Palma ratio.

The Palma ratio is the share of total income (or wealth) held by the richest
10% of the population divided by the share held by the poorest 40%.
"""

import numpy as np

from .utils import consistent_input
from .wolfson import lorenz_curve

__all__ = ["palma"]


@consistent_input
def palma(data):
    """Compute the Palma ratio of an income or wealth distribution.

    The Palma ratio is the income share of the top 10% divided by the income
    share of the bottom 40%. It is motivated by the observation that the middle
    50% of the population tends to hold a stable share of income, so differences
    in inequality are driven by the tails. The shares are read from the
    piecewise-linear Lorenz curve, so the ratio is also defined when the sample
    size is not a multiple of 10 (the population is treated as continuous).

    Parameters
    ----------
    data : array-like, pandas.Series, or pandas.DataFrame
        Income or wealth values. For a :class:`pandas.DataFrame`, pass
        ``column`` to select the values.
    column : str, optional
        Name of the column holding the values when ``data`` is a
        :class:`pandas.DataFrame`.

    Returns
    -------
    float
        The Palma ratio. It equals ``0.25`` for a perfectly equal distribution
        (10% / 40%) and grows with inequality. It is ``inf`` if the bottom 40%
        holds nothing.

    Raises
    ------
    ValueError
        If ``data`` has fewer than two observations, contains negative values,
        or sums to zero.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(palma(np.ones(10))), 6)
    0.25
    >>> incomes = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000,
    ...            80000, 120000]
    >>> round(float(palma(incomes)), 6)
    1.176471

    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 1:
        raise ValueError("'data' must be one-dimensional.")
    if y.size < 2:
        raise ValueError("At least two observations are required.")
    if np.any(y < 0):
        raise ValueError("All values in 'data' must be non-negative.")
    if y.sum() == 0:
        raise ValueError("The sum of 'data' must be positive.")

    population, cumulative = lorenz_curve(y)
    bottom_40 = np.interp(0.4, population, cumulative)
    top_10 = 1.0 - np.interp(0.9, population, cumulative)
    if bottom_40 == 0:
        return np.inf
    return top_10 / bottom_40
