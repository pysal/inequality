"""Helpers for accepting heterogeneous user input.

The measures in :mod:`inequality` historically operated on plain
:class:`numpy.ndarray` objects and only supported :mod:`pandas` structures by
coincidence.  :func:`_resolve_array` and the :func:`consistent_input` decorator
centralize the conversion so every public entry point can officially accept
any array-like (a list, tuple, set, range, generator, :class:`numpy.ndarray`,
:class:`pandas.Series`, ...) without the caller having to reach for
``.values``.

A :class:`pandas.DataFrame` is deliberately **not** accepted directly: per
the PySAL federation convention (see the discussion on
`PR #116 <https://github.com/pysal/inequality/pull/116>`__), select the column
yourself and pass the resulting Series/array, e.g. ``Gini(df["col"])`` rather
than ``Gini(df, column="col")``.
"""

from functools import wraps

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

__all__ = ["consistent_input"]


def _resolve_array(data):
    """Normalize supported input to a plain :class:`numpy.ndarray`.

    Parameters
    ----------
    data : array-like
        Any list-like object (``list``, ``tuple``, ``set``, ``range``,
        generator, :class:`numpy.ndarray`, :class:`pandas.Series`,
        :class:`pandas.Index`, ...). Strings, scalars, and
        :class:`pandas.DataFrame` are rejected.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    TypeError
        If ``data`` is a :class:`pandas.DataFrame`, or is not list-like
        (e.g. a string or a scalar).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> _resolve_array([1, 2, 3])
    array([1, 2, 3])
    >>> _resolve_array(x * x for x in range(4))
    array([0, 1, 4, 9])
    >>> _resolve_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])
    >>> _resolve_array(pd.DataFrame({"a": [1, 2]}))
    Traceback (most recent call last):
        ...
    TypeError: DataFrame input is not supported directly \
— pass a column as a Series, e.g. Gini(df['col']).
    """
    if isinstance(data, pd.DataFrame):
        # A DataFrame is technically list-like too (pandas.api.types.is_list_like
        # is True for it), but iterating one yields its *column names*, not the
        # data - silently falling through to the generic branch below would
        # compute a nonsense result instead of erroring.
        raise TypeError(
            "DataFrame input is not supported directly — pass a column as a "
            "Series, e.g. Gini(df['col'])."
        )
    if isinstance(data, np.ndarray | pd.Series):
        return np.asarray(data)
    if is_list_like(data):
        # materialize generators / other one-shot iterables and sidestep
        # numpy building a 0-d object array from e.g. a set
        return np.asarray(list(data))
    raise TypeError(
        "Input should be array-like (list, tuple, ndarray, generator, "
        "pandas Series, ...)."
    )


def consistent_input(func):
    """Decorate ``func`` so its first positional argument is normalized.

    The wrapped function is always called with a plain
    :class:`numpy.ndarray`. Callers may pass any array-like (list, tuple,
    generator, ``ndarray``, :class:`pandas.Series`, ...); a
    :class:`pandas.DataFrame` is rejected — select the column first, e.g.
    ``total(df["a"])``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> @consistent_input
    ... def total(data):
    ...     return int(data.sum())
    >>> total([1, 2, 3])
    6
    >>> total(pd.DataFrame({"a": [1, 2, 3]})["a"])
    6
    """

    @wraps(func)
    def wrapper(data, *args, **kwargs):
        return func(_resolve_array(data), *args, **kwargs)

    return wrapper
