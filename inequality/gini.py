"""
Gini based Inequality Metrics
"""

__author__ = "Sergio J. Rey <srey@asu.edu> "

import numpy
from scipy.stats import norm

__all__ = ["Gini", "Gini_Spatial"]


def _gini(x):
    """
    Memory efficient calculation of Gini coefficient
    in relative mean difference form.

    Parameters
    ----------

    x : array-like

    Attributes
    ----------

    g : float
        Gini coefficient.

    Notes
    -----
    Based on
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm.

    """

    n = len(x)
    try:
        x_sum = x.sum()
    except AttributeError:
        x = numpy.asarray(x)
        x_sum = x.sum()
    n_x_sum = n * x_sum
    x = x.ravel()  # ensure shape is (n,)
    r_x = (2.0 * numpy.arange(1, len(x) + 1) * x[numpy.argsort(x)]).sum()
    return (r_x - n_x_sum - x_sum) / n_x_sum


class Gini:
    """
    Classic Gini coefficient in absolute deviation form.

    Parameters
    ----------

    y : numpy.array
        An array in the shape :math:`(n,1)` containing the attribute values.

    Attributes
    ----------

    g : float
       Gini coefficient.

    """

    def __init__(self, x):
        self.g = _gini(x)


class Gini_Spatial:  # noqa N801
    """
    Spatial Gini coefficient.

    Provides for computationally based inference regarding the contribution of
    spatial neighbor pairs to overall inequality across a set of regions.
    See :cite:`Rey_2013_sea`.

    Parameters
    ----------

    y : numpy.array
        An array in the shape :math:`(n,1)` containing the attribute values.
    w : libpysal.weights.W
        Binary spatial weights object.
    permutations : int (default 99)
       The number of permutations for inference.

    Attributes
    ----------

    g : float
       Gini coefficient.
    wg : float
       Neighbor inequality component (geographic inequality).
    wcg : float
       Non-neighbor inequality component (geographic complement inequality).
    wcg_share : float
       Share of inequality in non-neighbor component.
    p_sim : float
       (If ``permuations > 0``) pseudo :math:`p`-value for spatial gini.
    e_wcg : float
       (If ``permuations > 0``) expected value of non-neighbor
       inequality component (level) from permutations.
    s_wcg : float
        (If ``permuations > 0``) standard deviation non-neighbor
        inequality component (level) from permutations.
    z_wcg : float
        (If ``permuations > 0``) z-value non-neighbor inequality
        component (level) from permutations.
    p_z_sim : float
        (If ``permuations > 0``) pseudo :math:`p`-value based on
        standard normal approximation of permutation based values.
    polarization: float
        Spatial polarization index with an expected value of 1.
    polarization_p_sim: float
        (If ``permutations >0``) pseudo :math:`p`-value for polarization index.
    polarization_sim: float
        (If ``permutations >0``) polarization values under the null from permutations.

    Examples
    --------

    >>> import libpysal
    >>> import numpy
    >>> from inequality.gini import Gini_Spatial

    Use data from the 32 Mexican States, decade frequency 1940-2010.

    >>> f = libpysal.io.open(libpysal.examples.get_path('mexico.csv'))
    >>> vnames = [f'pcgdp{dec}' for dec in range(1940, 2010, 10)]
    >>> y = numpy.transpose(numpy.array([f.by_col[v] for v in vnames]))

    Define regime neighbors.

    >>> regimes = numpy.array(f.by_col('hanson98'))
    >>> w = libpysal.weights.block_weights(regimes, silence_warnings=True)
    >>> numpy.random.seed(12345)
    >>> gs = Gini_Spatial(y[:,0], w)

    >>> float(gs.p_sim)
    0.04

    >>> float(gs.wcg)
    4353856.0

    >>> float(gs.e_wcg)
    4170356.7474747472

    Thus, the amount of inequality between pairs of states that are not in the
    same regime (neighbors) is significantly higher than what is expected
    under the null of random spatial inequality.

    """

    def __init__(self, x, w, permutations=99):
        x = numpy.asarray(x)
        g = _gini(x)
        self.g = g
        n = len(x)
        den = x.mean() * 2 * n**2
        d = g * den  # sum of absolute devations SAD
        wg = self._calc(x, w)  # sum of absolute deviations for neighbor pairs
        wcg = d - wg  # sum of absolution deviations for distant pairs
        n_pairs = n * (n - 1) / 2
        n_n_pairs = w.s0 / 2
        n_d_pairs = n_pairs - n_n_pairs
        polarization = (wcg / wg) * (n_n_pairs / n_d_pairs)
        self.polarization = polarization
        self.g = g
        self.wcg = wcg
        self.wg = wg
        self.dtotal = d
        self.den = den
        self.wcg_share = wcg / den

        if permutations:
            _scale = n_n_pairs / n_d_pairs
            ids = numpy.arange(n)
            wcgp = numpy.zeros((permutations,))
            polarization_sim = numpy.zeros((permutations,))
            for perm in range(permutations):
                numpy.random.shuffle(ids)
                wcgp[perm] = d - self._calc(x[ids], w)
                polar = wcgp[perm] / (d - wcgp[perm])
                polarization_sim[perm] = polar * _scale
            above = wcgp >= self.wcg
            larger = above.sum()
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.wcgp = wcgp
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.e_wcg = wcgp.mean()
            self.s_wcg = wcgp.std()
            self.z_wcg = (self.wcg - self.e_wcg) / self.s_wcg
            self.p_z_sim = 1.0 - norm.cdf(self.z_wcg)
            self.polarization_sim = polarization_sim
            # polarization is a directional concept, upper tail only
            larger = (polarization_sim >= polarization).sum()
            self.polarization_p_sim = (larger + 1) / (permutations + 1)

    def _calc(self, x, w):
        sad_sum = 0.0
        for i, js in w.neighbors.items():
            sad_sum += numpy.abs(x[i] - x[js]).sum()
        return sad_sum
