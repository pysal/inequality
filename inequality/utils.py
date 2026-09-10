"""Helpers for accepting heterogeneous user input.

The measures in :mod:`inequality` historically operated on plain
:class:`numpy.ndarray` objects and only supported :mod:`pandas` structures by
coincidence.  :func:`_resolve_array` and the :func:`consistent_input` decorator
centralize the conversion so every public entry point can officially accept any
array-like (a list, tuple, set, range, generator, :class:`numpy.ndarray`,
:class:`pandas.Series`, ...) or a :class:`pandas.DataFrame` (via a ``column``
selector) without the caller having to reach for ``.values``.
"""

from functools import wraps

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

__all__ = ["consistent_input"]


def _resolve_array(data, column=None):
    """Normalize supported input to a plain :class:`numpy.ndarray`.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Any list-like object (``list``, ``tuple``, ``set``, ``range``,
        generator, :class:`numpy.ndarray`, :class:`pandas.Series`,
        :class:`pandas.Index`, ...) or a :class:`pandas.DataFrame`.  Strings
        and scalars are rejected.
    column : str or list of str, optional
        Required when ``data`` is a :class:`pandas.DataFrame`; selects the
        column(s) holding the values.  A single string yields a 1-D array; a
        list of strings yields a 2-D ``(n, k)`` array.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``data`` is a :class:`pandas.DataFrame` and ``column`` is not given.
    TypeError
        If ``data`` is not list-like (e.g. a string or a scalar).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> _resolve_array([1, 2, 3])
    array([1, 2, 3])
    >>> _resolve_array(x * x for x in range(4))
    array([0, 1, 4, 9])
    >>> _resolve_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])
    >>> _resolve_array(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), column="a")
    array([1, 2])
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("For DataFrame input, 'column' argument must be provided.")
        return np.asarray(data[column])
    if isinstance(data, np.ndarray | pd.Series):
        return np.asarray(data)
    if is_list_like(data):
        # materialize generators / other one-shot iterables and sidestep
        # numpy building a 0-d object array from e.g. a set
        return np.asarray(list(data))
    raise TypeError(
        "Input should be array-like (list, tuple, ndarray, generator, "
        "pandas Series, ...) or a pandas DataFrame with a 'column' selector."
    )


def consistent_input(func):
    """Decorate ``func`` so its first positional argument is normalized.

    The wrapped function is always called with a plain
    :class:`numpy.ndarray`.  Callers may pass any array-like (list, tuple,
    generator, ``ndarray``, :class:`pandas.Series`, ...) or a
    :class:`pandas.DataFrame` together with a ``column=`` keyword.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> @consistent_input
    ... def total(data):
    ...     return int(data.sum())
    >>> total([1, 2, 3])
    6
    >>> total(pd.DataFrame({"a": [1, 2, 3]}), column="a")
    6
    """

    @wraps(func)
    def wrapper(data, *args, column=None, **kwargs):
        return func(_resolve_array(data, column), *args, **kwargs)

    return wrapper
