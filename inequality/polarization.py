import pandas as pd
import networkx as nx
import numpy as np
from tqdm import trange


def S(df, g, column, k=2, bins=None, permutations=999,
      seed=None, keep_sim=False):
    """Compute a spatial polarization index for a variable.

    This function measures the degree of spatial polarization by
    comparing the alignment between categorical groupings of a
    variable and the connectivity structure of a spatial graph.

    A higher index value indicates stronger spatial polarization of the values.

    The observed index is compared against a null distribution
    generated via Monte Carlo permutation, producing an empirical
    p-value to assess statistical significance.

    Parameters
    ----------
    df : pandas.DataFrame
         The dataframe containing spatial observations. Its index must
         align with the nodes in the PySAL spatial graph object `g`.

    g : libpysal.graph.Graph
        A PySAL spatial Graph object representing spatial
        connectivity. Internally converted to a NetworkX graph to
        evaluate component structure.

    column : str
        The name of the column in `df` to analyze for spatial polarization.

    bins : list of float, optional
        Cut points for binning the variable into discrete categories.
        If `None` (default), the variable is split at the median into
        two groups.

    permutations : int, default 999
        Number of permutations used to generate the null distribution
        for inference.

    seed : int or None, optional
        Random seed for reproducibility of the permutation test.

    keep_sim : bool, default False
        Whether to return the full array of simulated polarization scores.

    Returns
    -------
    s : float
        The observed spatial polarization index, bounded in [0, 1].
        Higher values indicate stronger spatial separation of the
        attribute groups.

    p_value : float
        Monte Carlo p-value indicating how extreme the observed index is
        under random label assignment.

    sim : numpy.ndarray, optional
        Array of simulated polarization indices from the permutation
        distribution. Returned only if `keep_sim` is True.

    Notes
    -----
    - The polarization index is based on connected components in a subgraph
      formed from edges linking observations in the same category.
    - The observed index reflects the relative reduction in fragmentation
      compared to a randomized assignment.
    - Useful for detecting spatial clustering, ghettoization, or regional
      segregation based on socioeconomic or demographic variables.

    Example
    -------
    >>> import libpysal
    >>> import pandas as pd
    >>> import numpy as np
    >>> from libpysal.weights import lat2W
    >>> from libpysal.graph import Graph

    >>> y = np.arange(1600)
    >>> df = pd.DataFrame({'y':y}, index=y)
    >>> g = Graph.from_W(lat2W(40, 40))
    >>> S(df, g, 'y', permutations=99, seed=1)
    (1.0, 0.01)

    """
    n = df.shape[0]
    if bins is not None:
        Ca = len(bins) - 1
        labels = range(Ca)
        clique = pd.cut(df[column], bins=bins, labels=labels)
    else:
        if not isinstance(k, int) or k < 1 or k > n:
            raise ValueError("'k' must be a positive integer less than n.")
        clique = pd.qcut(df[column], q=k, labels=False, duplicates='drop')
        Ca = k
        labels = range(Ca)

    Gg = g.to_networkx()
    Cg = nx.number_connected_components(Gg)
    k = max(Cg, Ca)
    focal = g.adjacency.index.get_level_values(0)
    neighbor = g.adjacency.index.get_level_values(1)

    def _calc(clique_labels, n, k):
        left = clique_labels.loc[focal].values
        right = clique_labels.loc[neighbor].values
        edges = g.adjacency[left == right]

        Gi = nx.Graph()
        Gi.add_nodes_from(Gg.nodes)
        Gi.add_edges_from({edge for edge, _ in edges.items()})
        Ci = nx.number_connected_components(Gi)
        return 1 - (Ci - k) / (n - k)

    s = _calc(clique, n, k)

    sim = np.zeros(permutations)
    rng = np.random.default_rng(seed)

    v = np.array(clique)
    for i in trange(permutations):
        shuffled = pd.Series(rng.permutation(v))
        sim[i] = _calc(shuffled, n, k)
    print(f'{sim.mean()=}')
    print(f'{sim.std()=}')
    p_value = ((sim >= s).sum()+1) / (permutations+1)
    if keep_sim:
        return s, p_value, sim
    else:
        return s, p_value
