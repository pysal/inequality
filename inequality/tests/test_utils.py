"""Officially supported array-like / pandas input across the package (GL#104)."""

import numpy as np
import pandas as pd
import pytest

from inequality.atkinson import Atkinson, atkinson
from inequality.gini import Gini
from inequality.theil import Theil, TheilD, TheilDSim
from inequality.utils import _resolve_array, consistent_input
from inequality.wolfson import wolfson

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

    def test_dataframe_single_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        np.testing.assert_array_equal(_resolve_array(df, column="a"), np.array([1, 2]))

    def test_dataframe_multi_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        np.testing.assert_array_equal(
            _resolve_array(df, column=["a", "b"]), np.array([[1, 3], [2, 4]])
        )

    def test_dataframe_requires_column(self):
        with pytest.raises(ValueError, match="'column' argument must be provided"):
            _resolve_array(pd.DataFrame({"a": [1, 2]}))

    @pytest.mark.parametrize("bad", ["abc", 42, 3.5, None])
    def test_unsupported_type(self, bad):
        with pytest.raises(TypeError, match="Input should be"):
            _resolve_array(bad)

    def test_decorator(self):
        @consistent_input
        def total(data):
            return data.sum()

        assert total([1, 2, 3]) == 6
        assert total(x for x in [1, 2, 3]) == 6
        assert total(pd.DataFrame({"a": [1, 2, 3]}), column="a") == 6


class TestGiniInput:
    def test_equivalence(self):
        expected = Gini(np.array(INCOMES)).g
        assert Gini(INCOMES).g == expected
        assert Gini(tuple(INCOMES)).g == expected
        assert Gini(iter(INCOMES)).g == expected
        assert Gini(x for x in INCOMES).g == expected
        assert Gini(pd.Series(INCOMES)).g == expected
        assert Gini(pd.DataFrame({"y": INCOMES}), column="y").g == expected


class TestAtkinsonInput:
    def test_function_equivalence(self):
        expected = atkinson(np.array(INCOMES), 0.5)
        assert atkinson(INCOMES, 0.5) == expected
        assert atkinson(pd.Series(INCOMES), 0.5) == expected
        assert atkinson(pd.DataFrame({"y": INCOMES}), 0.5, column="y") == expected

    def test_class_equivalence(self):
        expected = Atkinson(np.array(INCOMES), 0.5).A
        assert expected == Atkinson(pd.Series(INCOMES), 0.5).A
        assert expected == Atkinson(pd.DataFrame({"y": INCOMES}), 0.5, column="y").A


class TestTheilInput:
    def test_series(self):
        expected = Theil(np.array(INCOMES)).T
        np.testing.assert_array_equal(Theil(pd.Series(INCOMES)).T, expected)

    def test_dataframe_multi_column(self):
        y = np.column_stack([INCOMES, list(reversed(INCOMES))])
        df = pd.DataFrame(y, columns=["t0", "t1"])
        np.testing.assert_allclose(Theil(df, column=["t0", "t1"]).T, Theil(y).T)

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
        assert wolfson(pd.DataFrame({"y": INCOMES}), column="y") == expected
