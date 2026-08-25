import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from shapely.geometry import Polygon

from inequality.pen import _check_deps, pen, pengram

# Set the backend to 'Agg' to prevent GUI windows from opening
matplotlib.use("Agg")


# Test Data Setup


@pytest.fixture
def sample_df():
    """Sample dataframe for testing the pen function."""
    data = {
        "region": ["A", "B", "C", "D"],
        "income": [50000, 60000, 70000, 80000],
        "population": [100, 150, 200, 250],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_gdf():
    """Sample GeoDataFrame for testing the pengram function."""
    data = {"region": ["A", "B", "C", "D"], "income": [50000, 60000, 70000, 80000]}
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    gdf = gpd.GeoDataFrame(data, geometry=polygons)
    return gdf


# Test _check_deps function


def test_check_deps():
    """Test that _check_deps function imports all necessary dependencies."""
    sns, mc, pd = _check_deps()
    assert sns is not None
    assert mc is not None
    assert pd is not None


# Test pen function


def test_pen_basic(sample_df):
    """Test basic functionality of the pen function."""
    ax = pen(sample_df, col="income", x="region")
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "income"
    assert ax.get_xlabel() == "region"
    assert len(ax.patches) == len(sample_df), "All regions should be plotted."
    plt.close(ax.figure)  # Close the figure to free up resources


def test_pen_weighted(sample_df):
    """Test pen function with weighting."""
    ax = pen(sample_df, col="income", x="region", weight="population")
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "income"
    assert ax.get_xlabel() == "region"
    plt.close(ax.figure)  # Close the figure to free up resources


@pytest.mark.parametrize("weight_col", ["population", None])
def test_pen_parametrized(sample_df, weight_col):
    """Test pen function with and without weighting using parameterization."""
    ax = pen(sample_df, col="income", x="region", weight=weight_col)
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)  # Close the figure to free up resources


# Test pengram function


def test_pengram_basic(sample_gdf):
    """Test basic functionality of the pengram function."""
    ax, inset_ax = pengram(sample_gdf, col="income", name="region")
    assert ax is not None
    assert inset_ax is not None
    assert isinstance(ax, plt.Axes)
    assert isinstance(inset_ax, plt.Axes)
    plt.close(ax.figure)  # Close the main figure to free up resources
    plt.close(inset_ax.figure)  # Close the inset figure to free up resources


def test_pengram_custom_inset_size(sample_gdf):
    """Test pengram function with custom inset size."""
    ax, inset_ax = pengram(sample_gdf, col="income", name="region", inset_size="50%")
    assert ax is not None
    assert inset_ax is not None
    assert isinstance(ax, plt.Axes)
    assert isinstance(inset_ax, plt.Axes)
    plt.close(ax.figure)  # Close the main figure to free up resources
    plt.close(inset_ax.figure)  # Close the inset figure to free up resources


# Test invalid cases


def test_invalid_weight_column(sample_df):
    """Test pen function with an invalid weight column."""
    with pytest.raises(KeyError, match="invalid_column"):
        pen(sample_df, col="income", x="region", weight="invalid_column")


def test_invalid_query_column(sample_gdf):
    """Test pengram function with an invalid query column."""
    with pytest.raises(KeyError, match="invalid_column"):
        pengram(sample_gdf, col="income", name="invalid_column", query=["A", "C"])
