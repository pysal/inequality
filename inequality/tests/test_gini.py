import libpysal
import numpy
import pandas
import pytest
from libpysal.graph import Graph

from inequality.gini import Gini, Gini_Spatial


class TestGini:
    def setup_method(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = [f"pcgdp{dec}" for dec in range(1940, 2010, 10)]
        y = numpy.transpose(numpy.array([f.by_col[v] for v in vnames]))
        self.y = y[:, 0]
        regimes = numpy.array(f.by_col("hanson98"))

        self.w = libpysal.weights.block_weights(regimes, silence_warnings=True)

    def test_gini(self):
        g = Gini(self.y)
        numpy.testing.assert_almost_equal(g.g, 0.35372371173452849)

    def test_spatial(self):
        numpy.random.seed(12345)
        g = Gini_Spatial(self.y, self.w)
        numpy.testing.assert_almost_equal(g.g, 0.35372371173452849)
        numpy.testing.assert_almost_equal(g.wg, 884130.0)
        numpy.testing.assert_almost_equal(g.wcg, 4353856.0)
        numpy.testing.assert_almost_equal(g.p_sim, 0.040)
        numpy.testing.assert_almost_equal(g.e_wcg, 4170356.7474747472)

    def test_spatial_graph_matches_w(self):
        graph = Graph.from_W(self.w)

        numpy.random.seed(12345)
        from_w = Gini_Spatial(self.y, self.w)
        numpy.random.seed(12345)
        from_graph = Gini_Spatial(self.y, graph)

        for attr in ["g", "wg", "wcg", "polarization", "p_sim", "e_wcg"]:
            numpy.testing.assert_almost_equal(
                getattr(from_graph, attr), getattr(from_w, attr)
            )

    def test_spatial_graph_string_ids(self):
        # Ids that are labels, not positions, must still line up with x by order.
        labels = {i: f"state_{i}" for i in range(len(self.y))}
        adjacency = Graph.from_W(self.w).adjacency.rename(index=labels)
        graph = Graph(adjacency)

        numpy.random.seed(12345)
        g = Gini_Spatial(self.y, graph)
        numpy.testing.assert_almost_equal(g.wg, 884130.0)
        numpy.testing.assert_almost_equal(g.wcg, 4353856.0)

    def test_spatial_graph_isolate(self):
        # An isolate is stored as a zero-weight self-loop and must not count as a pair.
        x = numpy.array([1.0, 3.0, 10.0])
        adjacency = pandas.Series(
            [1.0, 1.0, 0.0],
            index=pandas.MultiIndex.from_tuples(
                [(0, 1), (1, 0), (2, 2)], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        g = Gini_Spatial(x, Graph(adjacency), permutations=0)

        # one neighbor pair counted in both directions: |1 - 3| * 2
        assert g.wg == 4.0
        numpy.testing.assert_almost_equal(g.polarization, (g.wcg / g.wg) * (1 / 2))

    def test_spatial_weighted_graph_matches_binary(self):
        # Weight values must not change the pair count or polarization.
        binary = Graph.from_W(self.w)
        row_standardized = binary.transform("r")

        numpy.random.seed(12345)
        from_binary = Gini_Spatial(self.y, binary)
        numpy.random.seed(12345)
        from_weighted = Gini_Spatial(self.y, row_standardized)

        for attr in ["wg", "wcg", "polarization", "p_sim", "polarization_p_sim"]:
            numpy.testing.assert_almost_equal(
                getattr(from_weighted, attr), getattr(from_binary, attr)
            )

    def test_spatial_self_loops_ignored(self):
        graph = Graph.from_W(self.w)
        with_loops = graph.assign_self_weight(1)

        numpy.random.seed(12345)
        expected = Gini_Spatial(self.y, graph)
        numpy.random.seed(12345)
        result = Gini_Spatial(self.y, with_loops)

        for attr in ["wg", "wcg", "polarization", "polarization_p_sim"]:
            numpy.testing.assert_almost_equal(
                getattr(result, attr), getattr(expected, attr)
            )

    def test_spatial_w_not_modified(self):
        before = self.w.sparse.toarray().copy()
        Gini_Spatial(self.y, self.w, permutations=0)
        numpy.testing.assert_array_equal(self.w.sparse.toarray(), before)

    def test_spatial_length_mismatch(self):
        with pytest.raises(ValueError, match="same number of observations"):
            Gini_Spatial(self.y[:-1], self.w, permutations=0)

    def test_spatial_isolates_only(self):
        x = numpy.array([1.0, 3.0, 10.0])
        adjacency = pandas.Series(
            [0.0, 0.0, 0.0],
            index=pandas.MultiIndex.from_tuples(
                [(0, 0), (1, 1), (2, 2)], names=["focal", "neighbor"]
            ),
            name="weight",
        )
        with pytest.warns(RuntimeWarning, match="wg == 0"):
            g = Gini_Spatial(x, Graph(adjacency), permutations=9)
        assert g.wg == 0
        assert numpy.isnan(g.polarization)
        assert numpy.isnan(g.polarization_sim).all()
