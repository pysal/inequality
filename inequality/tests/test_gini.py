import libpysal
import numpy
import pandas
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
