# test_spatial_polarization.py
import numpy as np
import pandas as pd
import pytest

# Import the implementation under test
from inequality.polarization import S


# ---------- Graph helpers (no PySAL required) ----------


class MiniGraph:
    """
    Minimal graph stub with the attributes S expects:
      - adjacency: pd.Series with MultiIndex (focal, neighbor)
      - n_components: int
      - component_labels: pd.Series aligned to node index 0..n-1
    """

    def __init__(self, n, undirected_edges):
        self.n = n
        # make symmetric adjacency (both directions)
        pairs = []
        for u, v in undirected_edges:
            pairs.append((u, v))
            pairs.append((v, u))
        mi = pd.MultiIndex.from_tuples(pairs, names=["focal", "neighbor"])
        # Values are dummies; only the index matters for S
        self.adjacency = pd.Series(np.ones(len(mi), dtype=int), index=mi)

        # compute connected components for the full graph
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

        for u, v in undirected_edges:
            union(u, v)
        roots = np.array([find(i) for i in range(n)])
        uniq, inv = np.unique(roots, return_inverse=True)
        self.n_components = int(uniq.size)
        self.component_labels = pd.Series(inv, index=pd.RangeIndex(n))


def edges_grid_rook(nrows, ncols):
    """Rook (4-neighbor) grid on an nrows x ncols lattice."""

    def nid(r, c):
        return r * ncols + c

    E = []
    for r in range(nrows):
        for c in range(ncols):
            if r + 1 < nrows:
                E.append((nid(r, c), nid(r + 1, c)))
            if c + 1 < ncols:
                E.append((nid(r, c), nid(r, c + 1)))
    return E


def edges_block_quadrants(nrows, ncols):
    """
    Four dense cliques on an nrows x ncols grid split into quadrants,
    no edges between quadrants. Assumes even nrows, ncols.
    """
    assert nrows % 2 == 0 and ncols % 2 == 0

    def nid(r, c):
        return r * ncols + c

    E = []
    rmid, cmid = nrows // 2, ncols // 2
    quads = [
        (0, rmid, 0, cmid),  # NW
        (0, rmid, cmid, ncols),  # NE
        (rmid, nrows, 0, cmid),  # SW
        (rmid, nrows, cmid, ncols),  # SE
    ]
    for r0, r1, c0, c1 in quads:
        nodes = [nid(r, c) for r in range(r0, r1) for c in range(c0, c1)]
        # make a clique among nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                E.append((nodes[i], nodes[j]))
    return E


# ---------- Fixtures ----------


@pytest.fixture
def grid8x8_rook():
    nrows = ncols = 8
    return MiniGraph(nrows * ncols, edges_grid_rook(nrows, ncols))


@pytest.fixture
def grid8x8_block():
    nrows = ncols = 8
    return MiniGraph(nrows * ncols, edges_block_quadrants(nrows, ncols))


# ---------- Tests ----------


def test_segregated_vs_swapped_block_k2(grid8x8_block):
    """
    On a 4-quadrant clique graph (dense), with k=2 and a 50/50 split:
      - Perfectly segregated rows → each quadrant monochrome → c=4 → S=1.0
      - Swapping first/last rows mixes each quadrant → c=8 → S≈0.9333
      - Permutations also give S≈0.9333 (degenerate null)
    """
    n = 64
    # values 0..63 laid row-wise
    y = np.arange(n)
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))

    # Segregated (original order): top half low, bottom half high
    s_segr = S(df, grid8x8_block, "y", k=2, permutations=0, verbose=False)
    assert np.isclose(s_segr.statistic_, 1.0)

    # Swap first and last rows: this mixes each quadrant
    # Swap nodes [0..7] with [56..63]
    y_swapped = y.copy()
    y_swapped[:8], y_swapped[-8:] = y[-8:].copy(), y[:8].copy()
    df_sw = pd.DataFrame({"y": y_swapped}, index=pd.RangeIndex(n))

    s_swap = S(
        df_sw,
        grid8x8_block,
        "y",
        k=2,
        permutations=64,
        seed=42,
        keep_sim=True,
        verbose=False,
    )
    # observed equals the null point mass
    assert np.isclose(s_swap.statistic_, 0.9333333333333333)
    assert np.allclose(s_swap.sim, 0.9333333333333333)


def test_variability_with_more_bins_or_sparser_graph(grid8x8_rook):
    """
    Using k=4 on a sparser rook grid should yield variability across permutations.
    """
    n = 64
    y = np.arange(n)  # any monotone vector; binning into quartiles
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))

    s = S(
        df,
        grid8x8_rook,
        "y",
        k=4,
        permutations=199,
        seed=123,
        keep_sim=True,
        verbose=False,
    )
    # Expect non-degenerate permutation distribution
    assert s.sim.max() - s.sim.min() > 0.0


def test_labels_and_component_count_with_nan(grid8x8_rook):
    """
    NaN in the attribute should produce i_labels == -1 at that node,
    and n_i_components should count only non-negative labels.
    """
    n = 64
    y = np.zeros(n, dtype=float)
    y[0] = np.nan  # one missing
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))

    s = S(df, grid8x8_rook, "y", k=2, permutations=0, verbose=False)

    # label at node 0 should be -1 (invalid)
    assert s.labels.loc[0, "i_labels"] == -1
    valid = s.labels["i_labels"] >= 0
    assert s.n_i_components == int(s.labels.loc[valid, "i_labels"].nunique())


def test_seed_determinism_and_pvalue_ties(grid8x8_block):
    """
    Same seed → identical sim and p-value.
    With degenerate null at 0.9333 and observed equal to it,
    (>=) tail yields a large p due to ties; strict (>) would reduce it.
    """
    n = 64
    y = np.arange(n)
    # swapped layout to match the null (degenerate at 0.9333)
    y_swapped = y.copy()
    y_swapped[:8], y_swapped[-8:] = y[-8:].copy(), y[:8].copy()
    df = pd.DataFrame({"y": y_swapped}, index=pd.RangeIndex(n))

    s1 = S(
        df,
        grid8x8_block,
        "y",
        k=2,
        permutations=99,
        seed=7,
        keep_sim=True,
        verbose=False,
    )
    s2 = S(
        df,
        grid8x8_block,
        "y",
        k=2,
        permutations=99,
        seed=7,
        keep_sim=True,
        verbose=False,
    )

    assert np.allclose(s1.sim, s2.sim)
    assert np.isclose(s1.p_value, s2.p_value)
    # observed equals sim values; (sim >= s) counts all → p close to 1
    assert np.isclose(s1.statistic_, 0.9333333333333333)
    assert np.allclose(s1.sim, s1.statistic_)


def test_repr_includes_key_fields(grid8x8_rook):
    """
    Smoke test: __repr__ includes the main summary fields without error.
    """
    n = 64
    y = np.arange(n)
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))
    s = S(df, grid8x8_rook, "y", k=2, permutations=9, seed=1, verbose=False)
    text = repr(s)
    for token in [
        "S Spatial Polarization Summary",
        "Variable:",
        "S:",
        "Number of spatial components:",
    ]:
        assert token in text


def test_bins_guard_raises_when_below_or_above_range(grid8x8_rook):
    """
    If explicit bins do not fully bound the data, S should raise ValueError.
    """
    n = 64
    y = np.arange(n)  # min=0, max=63
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))

    # Data below lowest bin edge -> error
    with pytest.raises(ValueError, match="Data outside bin range"):
        S(df, grid8x8_rook, "y", bins=[10, 32, 50], permutations=0, verbose=False)

    # Data above highest bin edge -> error
    with pytest.raises(ValueError, match="Data outside bin range"):
        S(df, grid8x8_rook, "y", bins=[-10, 10, 20], permutations=0, verbose=False)


def test_bins_guard_ok_when_bins_bound_data_and_no_nan_labels(grid8x8_rook):
    """
    When bins fully cover the data range, no error should be raised and
    no NaNs should be introduced by pandas.cut.
    """
    n = 64
    y = np.arange(n)  # min=0, max=63
    df = pd.DataFrame({"y": y}, index=pd.RangeIndex(n))

    # Pick edges that strictly bound the data range on both sides.
    # Note: pandas.cut is right-closed by default, so using [-1, 32, 64]
    # safely includes 0..63 with no NaNs from open left bound.
    s = S(df, grid8x8_rook, "y", bins=[-1, 32, 64], permutations=0, verbose=False)

    # Ensure attribute labels (bins) have no missing values
    assert not s.labels["a_labels"].isna().any()
