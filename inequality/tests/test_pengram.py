import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from inequality.pen import _check_deps, pen, pengram

# Test Data Setup


@pytest.fixture
def sample_df():
    """Sample dataframe for testing the pen function."""
    data = {
        'region': ['A', 'B', 'C', 'D'],
        'income': [50000, 60000, 70000, 80000],
        'population': [100, 150, 200, 250]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_gdf():
    """Sample GeoDataFrame for testing the pengram function."""
    data = {
        'region': ['A', 'B', 'C', 'D'],
        'income': [50000, 60000, 70000, 80000]
    }
    # Random polygons for simplicity
    from shapely.geometry import Polygon
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    ]
    gdf = gpd.GeoDataFrame(data, geometry=polygons)
    return gdf

# Test _check_deps function


def test_check_deps():
    """Test that _check_deps function imports all necessary dependencies."""
    sns, mc, plt, patches, inset_axes = _check_deps()
    assert sns is not None
    assert mc is not None
    assert plt is not None
    assert patches is not None
    assert inset_axes is not None

# Test pen function


def test_pen_basic(sample_df):
    """Test basic functionality of the pen function."""
    ax = pen(sample_df, col='income', x='region')
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == 'income'
    assert ax.get_xlabel() == 'region'


def test_pen_weighted(sample_df):
    """Test pen function with weighting."""
    ax = pen(sample_df, col='income', x='region', weight='population')
    assert ax is not None
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == 'income'
    assert ax.get_xlabel() == 'region'

# Test pengram function


def test_pengram_basic(sample_gdf):
    """Test basic functionality of the pengram function."""
    ax, inset_ax = pengram(sample_gdf, col='income', name='region')
    assert ax is not None
    assert inset_ax is not None
    assert isinstance(ax, plt.Axes)
    assert isinstance(inset_ax, plt.Axes)


def test_pengram_custom_inset_size(sample_gdf):
    """Test pengram function with custom inset size."""
    ax, inset_ax = pengram(sample_gdf, col='income',
                           name='region', inset_size="50%")
    assert ax is not None
    assert inset_ax is not None
    assert isinstance(ax, plt.Axes)
    assert isinstance(inset_ax, plt.Axes)

# Test invalid cases


def test_invalid_weight_column(sample_df):
    """Test pen function with an invalid weight column."""
    with pytest.raises(KeyError):
        pen(sample_df, col='income', x='region', weight='invalid_column')


def test_invalid_query_column(sample_gdf):
    """Test pengram function with an invalid query column."""
    with pytest.raises(KeyError):
        pengram(sample_gdf, col='income',
                name='invalid_column', query=['A', 'C'])
