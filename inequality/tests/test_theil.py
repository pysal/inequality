import libpysal
import numpy

from inequality.theil import Theil, TheilD, TheilDSim


class TestTheil:
    def test___init__(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = [f"pcgdp{dec}" for dec in range(1940, 2010, 10)]
        y = numpy.transpose(numpy.array([f.by_col[v] for v in vnames]))
        theil_y = Theil(y)
        numpy.testing.assert_almost_equal(
            theil_y.T,
            numpy.array(
                [
                    0.20894344,
                    0.15222451,
                    0.10472941,
                    0.10194725,
                    0.09560113,
                    0.10511256,
                    0.10660832,
                ]
            ),
        )


class TestTheilD:
    def test___init__(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = [f"pcgdp{dec}" for dec in range(1940, 2010, 10)]
        y = numpy.transpose(numpy.array([f.by_col[v] for v in vnames]))
        regimes = numpy.array(f.by_col("hanson98"))
        theil_d = TheilD(y, regimes)
        numpy.testing.assert_almost_equal(
            theil_d.bg,
            numpy.array(
                [
                    0.0345889,
                    0.02816853,
                    0.05260921,
                    0.05931219,
                    0.03205257,
                    0.02963731,
                    0.03635872,
                ]
            ),
        )

        y = numpy.array([0, 0, 0, 10, 10, 10])
        regions = numpy.array([0, 0, 0, 1, 1, 1])
        theil_d = TheilD(y, regions)
        numpy.testing.assert_almost_equal(theil_d.T, 0.6931471805599453)
        numpy.testing.assert_almost_equal(theil_d.bg, 0.6931471805599453)
        numpy.testing.assert_almost_equal(theil_d.wg, 0)


class TestTheilDSim:
    def test___init__(self):
        f = libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
        vnames = [f"pcgdp{dec}" for dec in range(1940, 2010, 10)]
        y = numpy.transpose(numpy.array([f.by_col[v] for v in vnames]))
        regimes = numpy.array(f.by_col("hanson98"))
        numpy.random.seed(10)
        theil_ds = TheilDSim(y, regimes, 999)
        numpy.testing.assert_almost_equal(
            theil_ds.bg_pvalue,
            numpy.array([0.4, 0.344, 0.001, 0.001, 0.034, 0.072, 0.032]),
        )
