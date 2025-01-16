"""Theil Inequality metrics"""

__author__ = "Sergio J. Rey <srey@sdsu.edu>"

import numpy

__all__ = ["Theil", "TheilD", "TheilDSim"]

SMALL = numpy.finfo("float").tiny


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

    y : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        with :math:`n` taken as the observations across which inequality is
        calculated.  If ``y`` is :math:`(n,)` then a scalar inequality value is
        determined. If ``y`` is :math:`(n,t)` then an array of inequality values are
        determined, one value for each column in ``y``.

    Attributes
    ----------

    T : numpy.array
        An array in the shape :math:`(t,)` or :math:`(1,)`
        containing Theil's *T* for each column of ``y``.

    Notes
    -----
    This computation involves natural logs. To prevent ``ln[0]`` from occurring, a
    small value is added to each element of ``y`` before beginning the computation.

    Examples
    --------

    >>> import libpysal
    >>> import numpy
    >>> from inequality.theil import Theil

    >>> f = libpysal.io.open(libpysal.examples.get_path('mexico.csv'))
    >>> vnames = [f'pcgdp{dec}' for dec in range(1940, 2010, 10)]
    >>> y = numpy.array([f.by_col[v] for v in vnames]).T
    >>> theil_y = Theil(y)

    >>> theil_y.T
    array([0.20894344, 0.15222451, 0.10472941, 0.10194725, 0.09560113,
           0.10511256, 0.10660832])

    """

    def __init__(self, y):
        n = len(y)
        y = y + SMALL * (y == 0)  # can't have 0 values
        yt = y.sum(axis=0)
        s = y / (yt * 1.0)
        lns = numpy.log(n * s)
        slns = s * lns
        t = sum(slns)
        self.T = t


class TheilD:
    """Decomposition of Theil's *T* based on partitioning of
    observations into exhaustive and mutually exclusive groups.

    Parameters
    ----------

    y : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        with :math:`n` taken as the observations across which inequality is
        calculated.  If ``y`` is :math:`(n,)` then a scalar inequality value is
        determined. If ``y`` is :math:`(n,t)` then an array of inequality values are
        determined, one value for each column in ``y``.
    partition : numpy.array
        An array in the shape :math:`(n,)` of elements indicating which partition
        each observation belongs to. These are assumed to be exhaustive.

    Attributes
    ----------

    T : numpy.array
        An array in the shape :math:`(t,)` or :math:`(1,)`
        containing the global inequality *T*.
    bg : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        representing between group inequality.
    wg : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        representing within group inequality.

    Examples
    --------

    >>> import libpysal
    >>> import numpy
    >>> from inequality.theil import TheilD

    >>> f = libpysal.io.open(libpysal.examples.get_path('mexico.csv'))
    >>> vnames = [f'pcgdp{dec}' for dec in range(1940, 2010, 10)]
    >>> y = numpy.array([f.by_col[v] for v in vnames]).T
    >>> regimes = numpy.array(f.by_col('hanson98'))
    >>> theil_d = TheilD(y, regimes)

    >>> theil_d.bg
    array([0.0345889 , 0.02816853, 0.05260921, 0.05931219, 0.03205257,
           0.02963731, 0.03635872])

    >>> theil_d.wg
    array([0.17435454, 0.12405598, 0.0521202 , 0.04263506, 0.06354856,
           0.07547525, 0.0702496 ])

    """

    def __init__(self, y, partition):
        groups = numpy.unique(partition)
        T = Theil(y).T  # noqa N806
        ytot = y.sum(axis=0)

        # group totals
        gtot = numpy.array([y[partition == gid].sum(axis=0) for gid in groups])

        if ytot.size == 1:  # y is 1-d
            sg = gtot / (ytot * 1.0)
            sg.shape = (sg.size, 1)
        else:
            sg = numpy.dot(gtot, numpy.diag(1.0 / ytot))
        ng = numpy.array([sum(partition == gid) for gid in groups])
        ng.shape = (ng.size,)  # ensure ng is 1-d
        n = y.shape[0]
        # between group inequality
        sg = sg + SMALL * (sg == 0)  # can't have 0 values

        bg = numpy.multiply(sg, numpy.log(numpy.dot(numpy.diag(n * 1.0 / ng), sg))).sum(
            axis=0
        )

        self.T = T
        self.bg = bg
        self.wg = T - bg


class TheilDSim:
    """Random permutation based inference on Theil's inequality decomposition.
    Provides for computationally based inference regarding the inequality
    decomposition using random spatial permutations.
    See :cite:`rey_interregional_2010`.

    Parameters
    ----------

    y : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        with :math:`n` taken as the observations across which inequality is
        calculated.  If ``y`` is :math:`(n,)` then a scalar inequality value is
        determined. If ``y`` is :math:`(n,t)` then an array of inequality values are
        determined, one value for each column in ``y``.
    partition : numpy.array
        An array in the shape :math:`(n,)` of elements indicating which partition
        each observation belongs to. These are assumed to be exhaustive.
    permutations : int
        The number of random spatial permutations for computationally
        based inference on the decomposition.

    Attributes
    ----------

    observed : numpy.array
        An array in the shape :math:`(n,t)` or :math:`(n,)`
        representing a ``TheilD`` instance for the observed data.
    bg : numpy.array
        An array in the shape ``(permutations+1, t)``
        representing between group inequality.
    bg_pvalue : numpy.array
        An array in the shape :math:`(t,1)` representing the :math:`p`-value
        for the between group measure. Measures the percentage of the realized
        values that were greater than or equal to the observed ``bg`` value.
        Includes the observed value.
    wg : numpy.array
        An array in the shape ``(permutations+1)``
        representing within group inequality. Depending on the
        shape of ``y``, the array may be 1- or 2-dimensional.

    Examples
    --------

    >>> import libpysal
    >>> import numpy
    >>> from inequality.theil import TheilDSim

    >>> f = libpysal.io.open(libpysal.examples.get_path('mexico.csv'))
    >>> vnames = [f'pcgdp{dec}' for dec in range(1940, 2010, 10)]
    >>> y = numpy.array([f.by_col[v] for v in vnames]).T
    >>> regimes = numpy.array(f.by_col('hanson98'))
    >>> numpy.random.seed(10)
    >>> theil_ds = TheilDSim(y, regimes, 999)

    >>> theil_ds.bg_pvalue
    array([0.4  , 0.344, 0.001, 0.001, 0.034, 0.072, 0.032])

    """

    def __init__(self, y, partition, permutations=99):
        observed = TheilD(y, partition)
        bg_ct = observed.bg == observed.bg  # already have one extreme value
        bg_ct = bg_ct * 1.0
        results = [observed]
        for _ in range(permutations):
            yp = numpy.random.permutation(y)
            t = TheilD(yp, partition)
            bg_ct += 1.0 * t.bg >= observed.bg
            results.append(t)
        self.results = results
        self.T = observed.T
        self.bg_pvalue = bg_ct / (permutations * 1.0 + 1)
        self.bg = numpy.array([r.bg for r in results])
        self.wg = numpy.array([r.wg for r in results])
