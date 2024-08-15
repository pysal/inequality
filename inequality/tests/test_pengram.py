import geopandas as gpd
import pandas as pd
import pytest
from inequality.pen import pen, pengram
from matplotlib.figure import Figure

# Sample DataFrames for testing


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'region': ['A', 'B', 'C', 'D'],
        'value': [100, 200, 300, 400],
        'weight': [1, 2, 3, 4]
    })


@pytest.fixture
def sample_gdf():
    from shapely.geometry import Point
    return gpd.GeoDataFrame({
        'region': ['A', 'B', 'C', 'D'],
        'value': [100, 200, 300, 400],
        'geometry': [Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)]
    })

# Test for the pen function without weights


def test_pen_no_weights(sample_df):
    fig = pen(df=sample_df, col='value', x='region')
    assert isinstance(fig, Figure), (
        "The pen function should return a Matplotlib Figure object."
    )
    assert fig.axes[0].get_xlabel() == 'region', (
        "The x-axis label should be set correctly."
    )
    assert fig.axes[0].get_ylabel() == 'value', (
        "The y-axis label should be set correctly."
    )

# Test for the pen function with weights


def test_pen_with_weights(sample_df):
    fig = pen(df=sample_df, col='value', x='region', weight='weight')
    assert isinstance(fig, Figure), (
        "The pen function should return a Matplotlib Figure object "
        "when weights are applied."
    )
    assert fig.axes[0].get_xlabel() == 'region', (
        "The x-axis label should be set correctly with weights."
    )
    assert fig.axes[0].get_ylabel() == 'value', (
        "The y-axis label should be set correctly with weights."
    )

# Test for the pengram function without highlighting


def test_pengram_no_query(sample_gdf):
    fig = pengram(gdf=sample_gdf, col='value', name='region')
    assert isinstance(fig, Figure), (
        "The pengram function should return a Matplotlib Figure object."
    )
    assert len(fig.axes) == 2, (
        "The pengram function should create two subplots."
    )

# Test for the pengram function with query


def test_pengram_with_query(sample_gdf):
    fig = pengram(gdf=sample_gdf, col='value', name='region', query=['A', 'C'])
    assert isinstance(fig, Figure), (
        "The pengram function should return a Matplotlib Figure object."
    )
    assert len(fig.axes) == 2, (
        "The pengram function should create two subplots."
    )
    assert 'A' in fig.axes[1].get_xticklabels()[0].get_text(), (
        "The x-tick labels should include the queried region 'A'."
    )
    assert 'C' in fig.axes[1].get_xticklabels()[2].get_text(), (
        "The x-tick labels should include the queried region 'C'."
    )

# Test to check if ImportError is raised for missing libraries in pen


def test_pen_importerror(sample_df, monkeypatch):
    monkeypatch.setattr(
        'builtins.__import__', lambda name, *args: exec('raise ImportError()')
    )
    with pytest.raises(ImportError):
        pen(df=sample_df, col='value', x='region')

# Test to check if ImportError is raised for missing libraries in pengram


def test_pengram_importerror(sample_gdf, monkeypatch):
    monkeypatch.setattr(
        'builtins.__import__', lambda name, *args: exec('raise ImportError()')
    )
    with pytest.raises(ImportError):
        pengram(gdf=sample_gdf, col='value', name='region')
