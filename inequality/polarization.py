"""
spatial_polarization.py

This module provides tools for assessing spatial polarization in
geographic data using graph-based connectivity structures. The core
function, `S`, implements a spatial polarization index that quantifies
the degree to which categorical groupings of a variable align with the
connectivity structure of a spatial graph.

The index captures the extent to which similar attribute values form
spatially contiguous regions. It does so by computing the number of
connected components in a subgraph where edges connect observations
within the same group, and compares this to a randomized baseline
generated via Monte Carlo permutation.

Dependencies:
    - pandas
    - numpy
    - tqdm
    - joblib

Typical use cases include:
    - Measuring spatial fragmentation or clustering of categorical
      attributes (e.g., income groups, political affiliation).
    - Comparing spatial structure across different variables or time
      periods.
    - Generating empirical p-values to assess the significance of
      observed spatial polarization patterns.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


class S:
    """
    Spatial Polarization Index

    This class computes a spatial polarization index that quantifies the
    degree to which categorical groupings of a variable align with the
    structure of a spatial graph. A higher value indicates stronger
    spatial clustering of similar values.

    The index is calculated by comparing the observed number of connected
    components in the graph induced by group membership against a null
    distribution generated through Monte Carlo permutation. See
    :cite:`rey_mind_the_gap_2026` for the underlying spatial polarization
    framework.

    Attributes
    ----------
    column : str
        The name of the variable analyzed for spatial polarization.

    n : int
        Number of observations.

    statistic_ : float
        The observed spatial polarization index.

    p_value : float
        Monte Carlo p-value based on the permutation distribution.
        Present only if `permutations > 0`.

    permutations : int
        The number of permutations used in the significance test.

    n_a_components : int
        Number of attribute-based groups (bins).

    n_g_components : int
        Number of spatial components in the original graph.

    n_i_components : int
        Number of connected components in the intersection subgraph.

    labels : pandas.DataFrame
        A dataframe containing:
            - i_labels: labels for intersection components
            - a_labels: attribute bin labels
            - g_labels: original spatial component labels

    sim : numpy.ndarray, optional
        Array of simulated index values from the permutation test.
        Present only if `keep_sim` is True.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from libpysal.weights import lat2W
    >>> from libpysal.graph import Graph
    >>> from spatial_polarization import S

    # Create a synthetic 40x40 spatial grid
    >>> y = np.arange(1600)
    >>> df = pd.DataFrame({'y': y}, index=y)
    >>> w = lat2W(40, 40)
    >>> g = Graph.from_W(w)

    # Compute the spatial polarization index
    >>> s = S(df, g, 'y', permutations=99, seed=123)
    >>> print(s)

    Notes
    -----
    The spatial polarization index measures the degree of spatial
    fragmentation by assessing how many connected subregions are
    formed by similar attribute values. This helps to identify whether
    like values are spatially clustered or dispersed.
    """

    def __init__(
        self,
        df,
        g,
        column,
        k=2,
        bins=None,
        permutations=999,
        seed=None,
        keep_sim=False,
        n_jobs=1,
        verbose=True,
    ):
        """
        Initialize the Spatial Polarization Index computation.

        Parameters
        ----------
        df : pandas.DataFrame
        Dataframe containing spatial observations. Index must align with
        nodes in the graph `g`.

        g : libpysal.graph.Graph
        A PySAL graph representing spatial adjacency.

        column : str
        Name of the column in `df` to evaluate for spatial polarization.

        k : int, default 2
        Number of quantile bins to divide the variable into, if `bins` is not specified.

        bins : list of float, optional
        Explicit cutpoints to bin the variable, passed directly to
        `pandas.cut`. Overrides `k` if provided.

        The first and last values in `bins` define the lower and upper
        bounds of the binning range. Any data values below the lowest
        bin edge or above the highest bin edge will be excluded (i.e.,
        assigned `NaN`). This can lead to errors in the computation of
        the S index if observations are dropped because of out-of-range
        values. To ensure all data are included, make sure the first
        and last bin edges bound the full range of the data.

        permutations : int, default 999
        Number of random permutations to generate the null distribution.

        seed : int or None, optional
        Seed for random number generator (reproducibility).

        keep_sim : bool, default False
        If True, store the full array of simulated statistics.

        n_jobs : int, default 1
        Number of parallel jobs for permutations. Use -1 for all CPUs.

        verbose : bool, default True
        If True, display a progress bar during permutation computation.
        """
        progress_iter = tqdm if verbose else (lambda x: x)

        n = df.shape[0]
        self.n = n
        if bins is not None:
            Ca = len(bins) - 1
            labels = range(Ca)
            clique = pd.cut(df[column], bins=bins, labels=labels)
            # Guard against out-of-range data excluded by pandas.cut
            if clique.isna().any():
                min_val, max_val = df[column].min(), df[column].max()
                bin_min, bin_max = bins[0], bins[-1]
                if min_val < bin_min or max_val > bin_max:
                    raise ValueError(
                        f"Data outside bin range: data min={min_val:.3f}, max={max_val:.3f}, "
                        f"but bins bound [{bin_min:.3f}, {bin_max:.3f}]. "
                        "Adjust 'bins' to fully include the data range."
                    )

        else:
            if not isinstance(k, int) or k < 1 or k > n:
                raise ValueError("'k' must be a positive integer less than n.")
            clique = pd.qcut(df[column], q=k, labels=False, duplicates="drop")
            clique = pd.Series(clique, index=df.index)
            Ca = k
            labels = range(Ca)

        Cg = g.n_components
        k = max(Cg, Ca)

        node_index = pd.Index(df.index)
        pos = pd.Series(np.arange(self.n), index=node_index)
        focal_nodes = g.adjacency.index.get_level_values(0)
        neighbor_nodes = g.adjacency.index.get_level_values(1)
        fi = pos.loc[focal_nodes].values
        nj = pos.loc[neighbor_nodes].values

        def _calc(clique_labels, n, k, ilabels=False):
            vals = clique_labels.values
            same = (vals[fi] == vals[nj]) & (~pd.isna(vals[fi])) & (~pd.isna(vals[nj]))

            # Union-Find over all nodes
            parent = np.arange(n)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for a, b, keep in zip(fi, nj, same):
                if keep:
                    union(a, b)

            valid = ~pd.isna(vals)
            roots = np.array([find(x) for x in range(n)])
            comp_ids = roots[valid]
            c = np.unique(comp_ids).size

            denom = max(int(valid.sum()) - k, 1)
            s = 1 - (c - k) / denom

            if ilabels:
                _, inv = np.unique(comp_ids, return_inverse=True)
                out = np.full(n, -1, dtype=int)
                out[valid] = inv
                return s, out
            return s

        s, ilabels = _calc(clique, n, k, ilabels=True)

        def permute_and_calc(series, n, k, child_seed):
            rng_i = np.random.default_rng(child_seed)
            shuffled = pd.Series(rng_i.permutation(series.values), index=series.index)
            return _calc(shuffled, n, k)

        child_seeds = np.random.SeedSequence(seed).spawn(permutations)
        sim = Parallel(n_jobs=n_jobs)(
            delayed(permute_and_calc)(clique, n, k, child_seed)
            for child_seed in progress_iter(child_seeds)
        )
        sim = np.array(sim)
        self.column = column
        if permutations > 0:
            self.p_value = ((sim >= s).sum() + 1) / (permutations + 1)

        self.labels = pd.DataFrame(
            {"i_labels": ilabels, "a_labels": clique, "g_labels": g.component_labels}
        )
        self.statistic_ = s
        self.permutations = permutations
        self.n_a_components = Ca
        self.n_g_components = Cg
        valid_i = self.labels["i_labels"] >= 0
        self.n_i_components = int(self.labels.loc[valid_i, "i_labels"].nunique())
        if keep_sim:
            self.sim = sim

    def __repr__(self):
        summary = f"""S Spatial Polarization Summary
{"=" * 55}
{"Variable:":<35}{self.column:>20}
{"n:":<35}{self.n:>20}
{"-" * 55}
{"S:":<40}{self.statistic_:>15.4f}
"""

        if self.permutations > 0:
            summary += f"""{"p-value:":<40}{self.p_value:>15.4f}
{"permutations:":<35}{self.permutations:>20d}
"""

        summary += f"""{"-" * 55}
{"Number of attribute components:":<40}{self.n_a_components:>15d}
{"Number of spatial components:":<40}{self.n_g_components:>15d}
{"Number of intersection components:":<40}{self.n_i_components:>15d}
{"=" * 55}
"""

        return summary
