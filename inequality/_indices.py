"""
Diversity indices as suggested in Nijkamp & Poot (2015) [1]

References
----------

[1]_ Nijkamp, P. and Poot, J. "Cultural Diversity: A Matter of Measurement".
     IZA Discussion Paper Series No. 8782
     :cite:`nijkamp2015cultural`
     https://www.econstor.eu/bitstream/10419/107568/1/dp8782.pdf
"""

import functools
import itertools
import warnings

import numpy

SMALL = numpy.finfo("float").tiny


def deprecated_function(func):
    """Decorator to mark functions as deprecated."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed on 2025-01-01.",
            FutureWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


@deprecated_function
def abundance(x):
    """
    Abundance index. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    a : float
        Abundance index.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> int(abundance(x))
    2

    """

    xs = x.sum(axis=0)
    a = numpy.sum([1 for i in xs if i > 0])
    return a


@deprecated_function
def margalev_md(x):
    """
    Margalev MD index. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    mmd : float
        Margalev MD index.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(margalev_md(x))
    0.40242960438184466

    """

    a = abundance(x)
    mmd = (a - 1.0) / numpy.log(x.sum())
    return mmd


@deprecated_function
def menhinick_mi(x):
    """
    Menhinick MI index. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    mmi : float
        Menhinick MI index.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(menhinick_mi(x))
    0.2886751345948129

    """

    a = abundance(x)
    mmi = (a - 1.0) / numpy.sqrt(x.sum())
    return mmi


@deprecated_function
def simpson_so(x):
    """
    Simpson diversity index SO. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    sso : float
        Simpson diversity index SO.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(simpson_so(x))
    0.5909090909090909

    """

    xs0 = x.sum(axis=0)
    xs = x.sum()
    num = (xs0 * (xs0 - 1.0)).sum()
    den = xs * (xs - 1.0)
    sso = num / den
    return sso


@deprecated_function
def simpson_sd(x):
    """
    Simpson diversity index SD. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    ssd : float
        Simpson diversity index SD.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(simpson_sd(x))
    0.40909090909090906

    """

    ssd = 1.0 - simpson_so(x)
    return ssd


@deprecated_function
def herfindahl_hd(x):
    """
    Herfindahl index HD. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    hhd : float
        Herfindahl index HD.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(herfindahl_hd(x))
    0.625

    """

    pgs = x.sum(axis=0)
    p = pgs.sum()
    hhd = ((pgs * 1.0 / p) ** 2).sum()
    return hhd


@deprecated_function
def theil_th(x, ridz=True):
    """
    Theil index TH as expressed in equation (32) of [2]. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).
    ridz : bool (default True)
        Flag to add a small amount to zero values to avoid zero division problems.

    Returns
    -------

    tth : float
        Theil index TH.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(theil_th(x))
    0.15106563978903298

    """

    if ridz:
        x = x + SMALL * (x == 0)  # can't have 0 values
    pa = x.sum(axis=1).astype(float)  # Area totals
    pg = x.sum(axis=0).astype(float)  # Group totals
    p = pa.sum()
    num = (x / pa[:, None]) * (numpy.log(pg / p) - numpy.log(x / pa[:, None]))
    den = ((pg / p) * numpy.log(pg / p)).sum()
    th = (pa / p)[:, None] * (num / den)
    tth = th.sum().sum()
    return tth


@deprecated_function
def theil_th_brute(x, ridz=True):
    """
    Theil index TH using inefficient computation.
    NOTE: just for result comparison, it matches ``theil_th``.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).
    ridz : bool (default True)
        Flag to add a small amount to zero values to avoid zero division problems.

    Returns
    -------

    tth : float
        Theil index TH.

    """

    if ridz:
        x = x + SMALL * (x == 0)  # can't have 0 values
    pas = x.sum(axis=1).astype(float)  # Area totals
    pgs = x.sum(axis=0).astype(float)  # Group totals
    p = pas.sum()
    th = numpy.zeros(x.shape)
    for g in numpy.arange(x.shape[1]):
        pg = pgs[g]
        for a in numpy.arange(x.shape[0]):
            pa = pas[a]
            pga = x[a, g]
            num = (pga / pa) * ((numpy.log(pg / p)) - numpy.log(pga / pa))
            den = ((pgs / p) * numpy.log(pgs / p)).sum()
            th[a, g] = (pa / p) * (num / den)
    tth = th.sum().sum()
    return tth


@deprecated_function
def fractionalization_gs(x):
    """
    Fractionalization Gini-Simpson index GS. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    fgs : float
        Fractionalization Gini-Simpson index GS.

    Examples
    --------

    >>> import numpy
    >>> x = numpy.array([[0, 1, 2], [0, 2, 4], [0, 0, 3]])
    >>> float(fractionalization_gs(x))
    0.375

    """

    fgs = 1.0 - herfindahl_hd(x)
    return fgs


@deprecated_function
def polarization(x):  # noqa ARG001
    raise RuntimeError("Not currently implemented.")


@deprecated_function
def shannon_se(x):
    """
    Shannon index SE. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    sse : float
        Shannon index SE.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> float(shannon_se(y))
    1.094070862104929

    """

    pgs = x.sum(axis=0)
    p = pgs.sum()
    ratios = pgs * 1.0 / p
    sse = -(ratios * numpy.log(ratios)).sum()
    return sse


@deprecated_function
def _gini(ys):
    """Gini for a single row to be used both by ``gini_gi`` and ``gini_gig``."""

    n = ys.flatten().shape[0]
    ys.sort()
    num = 2.0 * ((numpy.arange(n) + 1) * ys).sum()
    den = n * ys.sum()
    return (num / den) - ((n + 1.0) / n)


@deprecated_function
def gini_gi(x):
    """
    Gini GI index. :cite:`nijkamp2015cultural`

    NOTE: based on 3rd eq. of "Calculation" in:

            http://en.wikipedia.org/wiki/Gini_coefficient

    Returns same value as ``gini`` method in the R package ``reldist`` (see
    http://rss.acs.unt.edu/Rdoc/library/reldist/html/gini.html) if every
    category has at least one observation.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    ggi : float
        Gini GI index.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> float(round(gini_gi(y), 10))
    0.0512820513

    """
    ys = x.sum(axis=0)
    return _gini(ys)


@deprecated_function
def gini_gig(x):
    """
    Gini GI index. :cite:`nijkamp2015cultural`

    NOTE: based on Wolfram Mathworld formula in:

            http://mathworld.wolfram.com/GiniCoefficient.html

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    ggig : numpy.array
        Gini GI index for every group :math:`k`.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> gini_gig(y)
    array([0.125     , 0.32894737, 0.18181818])

    """

    ggig = numpy.apply_along_axis(_gini, 0, x.copy())
    return ggig


@deprecated_function
def gini_gi_m(x):
    """
    Gini GI index (equivalent to ``gini_gi``, not vectorized).
    :cite:`nijkamp2015cultural`

    NOTE: based on Wolfram Mathworld formula in:

            http://mathworld.wolfram.com/GiniCoefficient.html

    Returns same value as ``gini_gi``.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    ggim : float
        Gini GI index.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> float(round(gini_gi_m(y), 10))
    0.0512820513

    """

    xs = x.sum(axis=0)
    num = numpy.sum([numpy.abs(xi - xj) for xi, xj in itertools.permutations(xs, 2)])
    den = 2.0 * xs.shape[0] ** 2 * numpy.mean(xs)
    ggim = num / den
    return ggim


@deprecated_function
def hoover_hi(x):
    """
    Hoover index HI. :cite:`nijkamp2015cultural`

    NOTE: based on

            http://en.wikipedia.org/wiki/Hoover_index

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    hhi : float
        Hoover HI index.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> f'{hoover_hi(y):.3f}'
    '0.041'

    """

    es = x.sum(axis=0)
    e_total = es.sum()
    a_total = es.shape[0]
    s = numpy.abs((es * 1.0 / e_total) - (1.0 / a_total)).sum()
    hhi = s / 2.0
    return hhi


@deprecated_function
def similarity_w_wd(x, tau):
    """
    Similarity weighted diversity. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).
    tau : numpy.array
        A :math:`(k, k)` array where :math:`tau_{ij}` represents dissimilarity
        between group :math:`i` and group :math:`j`. Diagonal elements are
        assumed to be one.

    Returns
    -------

    swwd : float
        Similarity weighted diversity index.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> numpy.random.seed(0)
    >>> tau = numpy.random.uniform(size=(3,3))
    >>> numpy.fill_diagonal(tau, 0.)
    >>> tau = (tau + tau.T)/2
    >>> tau
    array([[0.        , 0.63003627, 0.52017529],
           [0.63003627, 0.        , 0.76883356],
           [0.52017529, 0.76883356, 0.        ]])

    >>> f'{similarity_w_wd(y, tau):.3f}'
    '0.582'

    """

    pgs = x.sum(axis=0)
    pgs = pgs * 1.0 / pgs.sum()
    s = sum(
        [
            pgs[i] * pgs[j] * tau[i, j]
            for i, j in itertools.product(numpy.arange(pgs.shape[0]), repeat=2)
        ]
    )
    swwd = 1.0 - s
    return swwd


@deprecated_function
def segregation_gsg(x):
    """
    Segregation index GS.

    This is a Duncan&Duncan index of a group against the rest combined.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    sgsg : array
        An array with GSg indices for the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> segregation_gsg(y).round(6)
    array([0.182927, 0.24714 , 0.097252])

    """

    pgs = x.sum(axis=0)
    pas = x.sum(axis=1)
    p = pgs.sum()
    first = (x.T * 1.0 / pgs[:, None]).T
    pampga = pas[:, None] - x
    pmpg = p - pgs
    second = pampga * 1.0 / pmpg[None, :]
    sgsg = 0.5 * (numpy.abs(first - second)).sum(axis=0)
    return sgsg


@deprecated_function
def modified_segregation_msg(x):
    """
    Modified segregation index GS.

    This is a modified version of GSg index as used by Van Mourik et al. (1989)
    :cite:`van_Mourik_1989`.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    ms_inds : numpy.array
        An array with MSg indices for the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> modified_segregation_msg(y).round(6)
    array([0.085207, 0.102249, 0.04355 ])

    """

    pgs = x.sum(axis=0)
    p = pgs.sum()
    ms_inds = segregation_gsg(x)  # To be updated in loop below
    for gi in numpy.arange(x.shape[1]):
        pg = pgs[gi]
        pgp = pg * 1.0 / p
        ms_inds[gi] = 2.0 * pgp * (1.0 - pgp) * ms_inds[gi]
    return ms_inds


@deprecated_function
def isolation_isg(x):
    """
    Isolation index IS. :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    iisg : numpy.array
        An array with ISg indices for the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> isolation_isg(y).round(6)
    array([1.07327 , 1.219953, 1.022711])

    """

    ws = x * 1.0 / x.sum(axis=0)
    pgapa = (x.T * 1.0 / x.sum(axis=1)).T
    pgp = x.sum(axis=0) * 1.0 / x.sum()
    iisg = (ws * pgapa / pgp).sum(axis=0)
    return iisg


@deprecated_function
def isolation_ii(x):
    """
    Isolation index :math:`II_g` as in equation (23) of [2].
    :cite:`nijkamp2015cultural`

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    iso_ii : numpy.array
        An array with IIg indices for the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> isolation_ii(y).round(6)
    array([1.11616 , 1.310804, 1.03433 ])

    """

    pa = x.sum(axis=1).astype(float)  # Area totals
    pg = x.sum(axis=0).astype(float)  # Group totals
    p = pa.sum()
    ws = x / pg

    block = (ws * (x / pa[:, None])).sum(axis=0)
    num = (block / (pg / p)) - (pg / p)
    den = 1.0 - (pg / p)
    iso_ii = num / den
    return iso_ii


@deprecated_function
def ellison_glaeser_egg(x, hs=None):
    """
    Ellison and Glaeser (1997) :cite:`ellison_1997` index of concentration.
    Implemented as in equation (5) of original reference.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        area) and :math:`k` columns (one per industry). Each cell indicates
        employment figures for area :math:`n` and industry :math:`k`.
    hs : numpy.array (default None)
        An array of dimension :math:`(k,)` containing the Herfindahl
        indices of each industry's plant sizes. If not passed, it is
        assumed every plant contains one and only one worker and thus
        :math:`H_k = 1 / P_k`, where :math:`P_k` is the total
        employment in :math:`k`.

    Returns
    -------

    eg_inds : numpy.array
        EG index for each of the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> z = numpy.random.randint(10, 50, size=(3,4))

    >>> ellison_glaeser_egg(z).round(6)
    array([0.054499, 0.016242, 0.010141, 0.028803])

    >>> numpy.random.seed(0)
    >>> v = numpy.random.uniform(0, 1, size=(4,)).round(3)
    >>> v
    array([0.549, 0.715, 0.603, 0.545])

    >>> ellison_glaeser_egg(z, hs=v).round(6)
    array([-1.06264 , -2.39227 , -1.461383, -1.117953])

    References
    ----------

    - :cite:`ellison_1997` Ellison, G. and Glaeser, E. L. "Geographic Concentration in U.S. Manufacturing Industries: A Dartboard Approach". Journal of Political Economy. 105: 889-927

    """  # noqa E501

    industry_totals = x.sum(axis=0)
    if hs is None:
        hs = 1.0 / industry_totals
    xs = x.sum(axis=1) * 1.0 / x.sum()
    part = 1.0 - (xs**2).sum()
    eg_inds = numpy.zeros(x.shape[1])
    for gi in numpy.arange(x.shape[1]):
        ss = x[:, gi] * 1.0 / industry_totals[gi]
        g = ((ss - xs) ** 2).sum()
        h = hs[gi]
        eg_inds[gi] = (g - part * h) / (part * (1.0 - h))
    return eg_inds


@deprecated_function
def ellison_glaeser_egg_pop(x):
    """
    Ellison and Glaeser (1997) :cite:`ellison_1997` index of concentration.
    Implemented to be computed with data about people (segregation/diversity)
    rather than as industry concentration, following Mare et al (2012)
    :cite:`care_2012`.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    eg_inds : numpy.array
        EG index for each of the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))
    >>> ellison_glaeser_egg_pop(y).round(6)
    array([-0.021508,  0.013299, -0.038946])

    References
    ----------

    - :cite:`ellison_1997` – Ellison, G. and Glaeser, E. L. "Geographic Concentration in U.S. Manufacturing Industries: A Dartboard Approach". Journal of Political Economy. 105: 889-927

    - :cite:`care_2012` – Care, D., Pinkerton, R., Poot, J. and Coleman, A. (2012) "Residential sorting across Auckland neighbourhoods." New Zealand Population Review, 38, 23-54.

    """  # noqa E501

    pas = x.sum(axis=1)
    pgs = x.sum(axis=0)
    p = pas.sum()
    pap = pas * 1.0 / p
    opg = 1.0 / pgs
    oopg = 1.0 - opg
    eg_inds = numpy.zeros(x.shape[1])
    for g in numpy.arange(x.shape[1]):
        pgas = x[:, g]
        pg = pgs[g]
        num1n = (((pgas * 1.0 / pg) - pap) ** 2).sum()
        num1d = 1.0 - (pap**2).sum()
        num2 = opg[g]
        den = oopg[g]
        eg_inds[g] = ((num1n / num1d) - num2) / den
    return eg_inds


@deprecated_function
def maurel_sedillot_msg(x, hs=None):
    """
    Maurel and Sedillot (1999) :cite:`maurel_1999` index of concentration.
    Implemented as in equation (7) of original reference.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        area) and :math:`k` columns (one per industry). Each cell indicates
        employment figures for area :math:`n` and industry :math:`k`.
    hs : numpy.array (default None)
        An array of dimension :math:`(k,)` containing the Herfindahl
        indices of each industry's plant sizes. If not passed, it is
        assumed every plant contains one and only one worker and thus
        :math:`H_k = 1 / P_k`, where :math:`P_k` is the total
        employment in :math:`k`.

    Returns
    -------

    ms_inds : numpy.array
        MS index for each of the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> z = numpy.random.randint(10, 50, size=(3,4))

    >>> maurel_sedillot_msg(z).round(6)
    array([ 0.078583,  0.035977,  0.039374, -0.009049])

    >>> numpy.random.seed(0)
    >>> v = numpy.random.uniform(0, 1, size=(4,)).round(3)
    >>> v
    array([0.549, 0.715, 0.603, 0.545])

    >>> maurel_sedillot_msg(z, hs=v).round(6)
    array([-1.010102, -2.324216, -1.38869 , -1.200499])

    References
    ----------

    - :cite:`maurel_1999` – Maurel, F. and Sédillot, B. (1999). "A Measure of the Geographic Concentration in French Manufacturing Industries." Regional Science and Urban Economics 29: 575-604.

    """  # noqa E501

    industry_totals = x.sum(axis=0)
    if hs is None:
        hs = 1.0 / industry_totals
    x2s = numpy.sum((x.sum(axis=1) * 1.0 / x.sum()) ** 2)
    ms_inds = numpy.zeros(x.shape[1])
    for gi in numpy.arange(x.shape[1]):
        s2s = numpy.sum((x[:, gi] * 1.0 / industry_totals[gi]) ** 2)
        h = hs[gi]
        num = ((s2s - x2s) / (1.0 - x2s)) - h
        den = 1.0 - h
        ms_inds[gi] = num / den
    return ms_inds


@deprecated_function
def maurel_sedillot_msg_pop(x):
    """
    Maurel and Sedillot (1999) :cite:`maurel_1999` index of concentration.
    Implemented to be computed with data about people (segregation/diversity)
    rather than as industry concentration, following Mare et al (2012)
    :cite:`care_2012`.

    Parameters
    ----------

    x : numpy.array
        An :math:`(N, k)` shaped array containing :math:`N` rows (one per
        neighborhood) and :math:`k` columns (one per cultural group).

    Returns
    -------

    eg_inds : numpy.array
        MS index for each of the :math:`k` groups.

    Examples
    --------

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> y = numpy.random.randint(1, 10, size=(4,3))

    >>> maurel_sedillot_msg_pop(y).round(6)
    array([-0.055036,  0.044147, -0.028666])

    References
    ----------

    - :cite:`maurel_1999` – Maurel, F. and Sédillot, B. (1999). "A Measure of the Geographic Concentration in French Manufacturing Industries." Regional Science and Urban Economics 29: 575-604.

    - :cite:`care_2012` – Care, D., Pinkerton, R., Poot, J. and Coleman, A. (2012) "Residential sorting across Auckland neighbourhoods." New Zealand Population Review, 38, 23-54.

    """  # noqa E501

    pas = x.sum(axis=1)
    pgs = x.sum(axis=0)
    p = pas.sum()
    pap = pas * 1.0 / p
    eg_inds = numpy.zeros(x.shape[1])
    for g in numpy.arange(x.shape[1]):
        pgas = x[:, g]
        pg = pgs[g]
        num1n = ((pgas * 1.0 / pg) ** 2 - pap**2).sum()
        num1d = 1.0 - (pap**2).sum()
        num2 = 1.0 / pg
        den = 1.0 - (1.0 / pg)
        eg_inds[g] = ((num1n / num1d) - num2) / den
    return eg_inds
