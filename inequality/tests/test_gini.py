import libpysal
import numpy

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
