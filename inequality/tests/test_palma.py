import numpy as np
import pandas as pd
import pytest

from inequality.palma import palma

INCOMES = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]


def test_equal_distribution_is_quarter():
    assert palma(np.ones(10)) == pytest.approx(0.25)
    # not a multiple of 10 observations
    assert palma(np.ones(7)) == pytest.approx(0.25)


def test_known_value():
    # top 10% = largest value, bottom 40% = four smallest values
    y = np.sort(INCOMES)
    assert palma(INCOMES) == pytest.approx(y[-1] / y[:4].sum())


def test_input_types_agree():
    expected = palma(INCOMES)
    assert palma(np.array(INCOMES)) == pytest.approx(expected)
    assert palma(tuple(INCOMES)) == pytest.approx(expected)
    assert palma(pd.Series(INCOMES)) == pytest.approx(expected)
    df = pd.DataFrame({"income": INCOMES})
    assert palma(df, column="income") == pytest.approx(expected)


def test_order_and_scale_invariant():
    rng = np.random.default_rng(1)
    y = rng.lognormal(size=57)
    assert palma(rng.permutation(y)) == pytest.approx(palma(y))
    assert palma(y * 1000) == pytest.approx(palma(y))


def test_more_unequal_is_larger():
    assert palma([1, 1, 1, 1, 1, 1, 1, 1, 1, 50]) > palma([1] * 10)


def test_zero_bottom_share_is_inf():
    assert palma([0, 0, 0, 0, 1, 2, 3, 4, 5, 6]) == np.inf


@pytest.mark.parametrize(
    "bad, match",
    [
        ([1.0], "two observations"),
        ([1, -2, 3], "non-negative"),
        ([0, 0, 0], "sum"),
    ],
)
def test_invalid_input(bad, match):
    with pytest.raises(ValueError, match=match):
        palma(bad)


def test_dataframe_requires_column():
    with pytest.raises(ValueError, match="column"):
        palma(pd.DataFrame({"income": INCOMES}))
