"""
Theil Inequality metrics
"""

__author__ = "Sergio J. Rey <srey@asu.edu>"

import warnings

import numpy as np
import pandas as pd

__all__ = ["Theil", "TheilD", "TheilDSim"]

SMALL = np.finfo("float").tiny


class Theil:
    """
    Classic Theil measure of inequality.

    .. math::

        T = \\sum_{i=1}^n
            \\left( \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln
                \\left[ N \\frac{y_i}{\\sum_{i=1}^n y_i}\\right]
            \\right
        )

    Parameters
    ----------
    y : array-like, DataFrame, or sequence
        Either an `nxT` array (deprecated) or `nx1` sequence or DataFrame.
        If using a DataFrame, specify the column(s) using `column` keyword(s).

    column : str, optional
        If `y` is a DataFrame, specify the column to be used for the calculation.

    Attributes
    ----------
    T : numpy.array
        Theil's *T* index.

    Notes
    -----
    The old API (nxT arrays) is deprecated and will be removed in the future.
    Use nx1 sequences or DataFrames with a single column instead.

    """

    def __init__(self, y, column=None):
        # Deprecation warning for old API
        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
            warnings.warn(
                "The nxT input format is deprecated. In future versions, "
                "please provide nx1 sequences or a DataFrame with a single column.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Old API behavior
            n = y.shape[0]
        else:
            # New API: Handle sequence or DataFrame
            if isinstance(y, pd.DataFrame):
                if column is None:
                    raise ValueError("For DataFrame input, `column` must be specified.")
                y = y[column].values
            elif isinstance(y, list | np.ndarray):
                y = np.asarray(y)
            else:
                raise TypeError("Input must be an array, list, or DataFrame.")
            n = len(y)

        # Calculation
        y = y + SMALL * (y == 0)  # can't have 0 values
        yt = y.sum(axis=0)
        s = y / (yt * 1.0)
        lns = np.log(n * s)
        slns = s * lns
        self.T = sum(slns)


class TheilD:
    """
    Decomposition of Theil's *T* based on partitioning of
    observations into exhaustive and mutually exclusive groups.

    Parameters
    ----------
    y : array-like, DataFrame, or sequence
        Either an `nxT` array (deprecated) or `nx1` sequence or DataFrame.
        If using a DataFrame, specify the column(s) using `column` keyword(s).

    partition : array-like, DataFrame, or sequence
        Partition indicating group membership.
        If using a DataFrame, specify the column using `partition_col`.

    column : str, optional
        If `y` is a DataFrame, specify the column to be used for the calculation.

    partition_col : str, optional
        If `partition` is a DataFrame, specify the column to be used.

    Attributes
    ----------
    T : numpy.array
        Global Theil's *T*.
    bg : numpy.array
        Between-group inequality.
    wg : numpy.array
        Within-group inequality.

    """

    def __init__(self, y, partition, column=None, partition_col=None):
        # Deprecation warning for old API
        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
            warnings.warn(
                "The nxT input format is deprecated. In future versions, "
                "please provide nx1 sequences or a DataFrame with a single column.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            # New API: Handle sequence or DataFrame
            if isinstance(y, pd.DataFrame):
                if column is None:
                    raise ValueError("For DataFrame input, `column` must be specified.")
                y = y[column].values
            elif isinstance(y, list | np.ndarray):
                y = np.asarray(y)
            else:
                raise TypeError("Input must be an array, list, or DataFrame.")

        # Handle partition similarly
        if isinstance(partition, pd.DataFrame):
            if partition_col is None:
                raise ValueError(
                    "For DataFrame input, `partition_col` must be specified."
                )
            partition = partition[partition_col].values
        elif isinstance(partition, list | np.ndarray):
            partition = np.asarray(partition)
        else:
            raise TypeError("Partition must be an array, list, or DataFrame.")

        groups = np.unique(partition)
        t = Theil(y).T
        ytot = y.sum(axis=0)

        # Group totals
        gtot = np.array([y[partition == gid].sum(axis=0) for gid in groups])

        if ytot.size == 1:
            sg = gtot / (ytot * 1.0)
            sg.shape = (sg.size, 1)
        else:
            sg = np.dot(gtot, np.diag(1.0 / ytot))
        ng = np.array([sum(partition == gid) for gid in groups])
        ng.shape = (ng.size,)  # ensure ng is 1-d
        # Between group inequality
        sg = sg + (sg == 0)  # handle case when a partition has 0 for sum
        bg = np.multiply(sg, np.log(np.dot(np.diag(len(y) * 1.0 / ng), sg))).sum(axis=0)

        self.T = t
        self.bg = bg
        self.wg = t - bg


class TheilDSim:
    """
    Random permutation-based inference on Theil's inequality decomposition.

    Parameters
    ----------
    y : array-like, DataFrame, or sequence
        Either an `nxT` array (deprecated) or `nx1` sequence or DataFrame.
        If using a DataFrame, specify the column(s) using `column` keyword(s).

    partition : array-like, DataFrame, or sequence
        Partition indicating group membership.
        If using a DataFrame, specify the column using `partition_col`.

    permutations : int, optional
        Number of random permutations for inference (default: 99).

    column : str, optional
        If `y` is a DataFrame, specify the column to be used for the calculation.

    partition_col : str, optional
        If `partition` is a DataFrame, specify the column to be used.

    """

    def __init__(self, y, partition, permutations=99, column=None, partition_col=None):
        observed = TheilD(y, partition, column=column, partition_col=partition_col)
        bg_ct = observed.bg == observed.bg  # already have one extreme value
        bg_ct = bg_ct * 1.0
        results = [observed]
        for _ in range(permutations):
            yp = np.random.permutation(y)
            t = TheilD(yp, partition)
            bg_ct += 1.0 * t.bg >= observed.bg
            results.append(t)
        self.results = results
        self.T = observed.T
        self.bg_pvalue = bg_ct / (permutations * 1.0 + 1)
        self.bg = np.array([r.bg for r in results])
        self.wg = np.array([r.wg for r in results])
