"""Officially supported array-like / pandas input across the package (#104)."""

import numpy as np
import pandas as pd
import pytest

from inequality.atkinson import Atkinson, atkinson
from inequality.gini import Gini
from inequality.theil import Theil, TheilD, TheilDSim
from inequality.utils import _resolve_array, consistent_input
from inequality.wolfson import lorenz_curve, wolfson

INCOMES = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]


class TestResolveArray:
    def test_list(self):
        np.testing.assert_array_equal(_resolve_array([1, 2, 3]), np.array([1, 2, 3]))

    def test_tuple(self):
        np.testing.assert_array_equal(_resolve_array((1, 2, 3)), np.array([1, 2, 3]))

    def test_range(self):
        np.testing.assert_array_equal(_resolve_array(range(4)), np.array([0, 1, 2, 3]))

    def test_generator(self):
        np.testing.assert_array_equal(
            _resolve_array(x * x for x in range(4)), np.array([0, 1, 4, 9])
        )

    def test_iterator(self):
        np.testing.assert_array_equal(
            _resolve_array(iter([1, 2, 3])), np.array([1, 2, 3])
        )

    def test_set(self):
        # order is undefined but the contents must round-trip
        assert sorted(_resolve_array({3, 1, 2}).tolist()) == [1, 2, 3]

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_resolve_array(arr), arr)

    def test_series(self):
        np.testing.assert_array_equal(
            _resolve_array(pd.Series([1, 2, 3])), np.array([1, 2, 3])
        )

    def test_dataframe_rejected(self):
        # A DataFrame is list-like too (iterating it yields column *names*),
        # so it must be rejected explicitly rather than silently misread.
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError, match=r"DataFrame input is not supported"):
            _resolve_array(df)

    @pytest.mark.parametrize("bad", ["abc", 42, 3.5, None])
    def test_unsupported_type(self, bad):
        with pytest.raises(TypeError):
            _resolve_array(bad)

    def test_decorator(self):
        @consistent_input
        def total(data):
            return data.sum()

        assert total([1, 2, 3]) == 6
        assert total(x for x in [1, 2, 3]) == 6
        assert total(pd.DataFrame({"a": [1, 2, 3]})["a"]) == 6


class TestGiniInput:
    def test_equivalence(self):
        expected = Gini(np.array(INCOMES)).g
        assert Gini(INCOMES).g == expected
        assert Gini(tuple(INCOMES)).g == expected
        assert Gini(iter(INCOMES)).g == expected
        assert Gini(x for x in INCOMES).g == expected
        assert Gini(pd.Series(INCOMES)).g == expected
        assert Gini(pd.DataFrame({"y": INCOMES})["y"]).g == expected

    def test_dataframe_rejected(self):
        with pytest.raises(TypeError, match=r"DataFrame input is not supported"):
            Gini(pd.DataFrame({"y": INCOMES}))


class TestAtkinsonInput:
    def test_function_equivalence(self):
        expected = atkinson(np.array(INCOMES), 0.5)
        assert atkinson(INCOMES, 0.5) == expected
        assert atkinson(pd.Series(INCOMES), 0.5) == expected
        assert atkinson(pd.DataFrame({"y": INCOMES})["y"], 0.5) == expected

    def test_class_equivalence(self):
        expected = Atkinson(np.array(INCOMES), 0.5).A
        assert expected == Atkinson(pd.Series(INCOMES), 0.5).A
        assert expected == Atkinson(pd.DataFrame({"y": INCOMES})["y"], 0.5).A


class TestTheilInput:
    def test_series(self):
        expected = Theil(np.array(INCOMES)).T
        np.testing.assert_array_equal(Theil(pd.Series(INCOMES)).T, expected)

    def test_dataframe_multi_column(self):
        # Multi-column (n, t) input: select the columns yourself and convert -
        # Theil (like the rest of the package) does not accept a DataFrame.
        y = np.column_stack([INCOMES, list(reversed(INCOMES))])
        df = pd.DataFrame(y, columns=["t0", "t1"])
        np.testing.assert_allclose(Theil(df[["t0", "t1"]].to_numpy()).T, Theil(y).T)

    def test_theild_series_partition(self):
        y = np.array([0, 0, 0, 10, 10, 10])
        part = [0, 0, 0, 1, 1, 1]
        np.testing.assert_allclose(
            TheilD(pd.Series(y), pd.Series(part)).bg, TheilD(y, np.array(part)).bg
        )

    def test_theildsim_series_partition(self):
        y = np.array([1.0, 2, 3, 40, 50, 60])
        part = [0, 0, 0, 1, 1, 1]
        np.random.seed(10)
        a = TheilDSim(pd.Series(y), pd.Series(part), 99).bg_pvalue
        np.random.seed(10)
        b = TheilDSim(y, np.array(part), 99).bg_pvalue
        np.testing.assert_array_equal(a, b)


class TestWolfsonInput:
    def test_equivalence(self):
        expected = wolfson(np.array(INCOMES))
        assert wolfson(INCOMES) == expected
        assert wolfson(pd.Series(INCOMES)) == expected
        assert wolfson(pd.DataFrame({"y": INCOMES})["y"]) == expected

    def test_column_kwarg_deprecated(self):
        df = pd.DataFrame({"y": INCOMES})
        with pytest.deprecated_call(match="column"):
            deprecated = wolfson(df, column="y")
        assert deprecated == wolfson(df["y"])

    def test_lorenz_curve_column_kwarg_deprecated(self):
        df = pd.DataFrame({"y": INCOMES})
        with pytest.deprecated_call(match="column"):
            deprecated = lorenz_curve(df, column="y")
        expected = lorenz_curve(df["y"])
        np.testing.assert_array_equal(deprecated[0], expected[0])
        np.testing.assert_array_equal(deprecated[1], expected[1])
