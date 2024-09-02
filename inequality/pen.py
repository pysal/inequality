"""
Pen's Parade and Pengram Visualizations

This module provides functions to create Pen's Parade visualizations and
extend them with choropleth maps to display the spatial distribution of
values. The `pen` function generates a traditional Pen's Parade, which is
a visual representation of income distribution or similar data, typically
used to show inequality. The `pengram` function enhances this by combining
the Pen's Parade with a choropleth map, allowing for a richer analysis of
spatial data distributions.

Author
------
Serge Rey <srey@sdsu.edu>
"""

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _check_deps(caller="pen"):
    """
    Check for required dependencies.

    Returns
    -------
    tuple
        A tuple containing the imported modules (Seaborn, mapclassify, pandas).
    """
    try:
        import seaborn as sns
    except ImportError as e:
        msg = f"{caller} requires Seaborn."
        msg = f"{msg} Install it using `conda install -c conda-forge seaborn`"
        raise ImportError(msg) from e

    try:
        import mapclassify as mc
    except ImportError as e:
        msg = f"{caller} requires mapclassify."
        msg = f"{msg} Install it using `conda install -c conda-forge mapclassify`"
        raise ImportError(msg) from e

    try:
        import pandas as pd
    except ImportError as e:
        msg = f"{caller} requires pandas. "
        msg = f"{msg} Install it using `conda install -c conda-forge pandas`"
        raise ImportError(msg) from e

    return sns, mc, pd


def pen(
    df,
    col,
    x,
    weight=None,
    ascending=True,
    xticks=True,
    total_bars=100,
    figsize=(8, 6),
    ax=None,
):
    """
    Creates the Pen's Parade visualization.

    This function generates a bar plot sorted by a specified column, with
    options to customize the x-axis ticks and figure size. The Pen's Parade
    is a visual representation of income distribution (or similar data),
    typically used to show inequality.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    col : str
        The column to plot on the y-axis.
    x : str
        The column to plot on the x-axis.
    weight : str, optional
        A column used to weight the bars in the Pen’s Parade. Default is None.
    ascending : bool, optional
        Whether to sort the DataFrame in ascending order by the `col`.
        Default is True.
    xticks : bool, optional
        Whether to show x-axis ticks. Default is True.
    total_bars : int, optional
        Total number of bars to create for the weighted Pen’s Parade. Default
        is 100.
    figsize : list, optional
        The size of the figure as a list [width, height]. Default is [8, 6].
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes instance to plot on. If None, a new figure and axes
        will be created. Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        A Matplotlib Axes object with the Pen's Parade plot.

    """

    sns, mc, pd = _check_deps()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if weight is None:
        dbfs = df.sort_values(col, ascending=ascending).reset_index(drop=True)
        sns.barplot(x=x, y=col, data=dbfs, ax=ax)
        ax.set_ylabel(col)
        ax.set_xlabel(x)
        plt.xticks(rotation=90)
        ax.set_xticks(dbfs.index)
        ax.set_xticklabels(dbfs[x], rotation=90)

        if not xticks:
            ax.set(xticks=[])
            ax.set(xlabel="")
    else:
        df["NumBars"] = (
            (df[weight] / df[weight].sum() * total_bars).apply(math.ceil).astype(int)
        )

        repeated_rows = []
        name = x
        for _, row in df.iterrows():
            repeated_rows.extend([row] * row["NumBars"])

        df_repeated = pd.DataFrame(repeated_rows)

        df_sorted = df_repeated.sort_values(by=col).reset_index(drop=True)

        unique_obs = df[name].unique()
        colors = plt.get_cmap("tab20", len(unique_obs))
        color_map = {state: colors(i) for i, state in enumerate(unique_obs)}
        bar_colors = df_sorted[name].map(color_map)

        bar_positions = np.arange(len(df_sorted))
        bar_heights = df_sorted[col]
        bar_widths = 1  # Equal width for all bars

        _ = ax.bar(
            bar_positions,
            bar_heights,
            width=bar_widths,
            color=bar_colors,
            edgecolor="black",
        )
        tick_width = plt.rcParams["xtick.major.width"]

        first_positions = []
        first_labels = []
        current_state = None
        state_index = 0
        last_name = df_sorted[name].iloc[-1]
        for i in range(len(bar_positions)):
            label = df_sorted[name].iloc[i]
            if label != current_state:
                if state_index % 2 == 0 or label == last_name:
                    first_positions.append(bar_positions[i])
                    first_labels.append(df_sorted[name].iloc[i])
                else:
                    text_y_position = bar_heights[i] + 0.05 * max(bar_heights)
                    ax.plot(
                        [bar_positions[i], bar_positions[i]],
                        [bar_heights[i], text_y_position - 550],
                        color="black",
                        linewidth=tick_width,
                    )
                    ax.text(
                        bar_positions[i],
                        text_y_position,
                        df_sorted[name].iloc[i],
                        ha="center",
                        rotation=90,
                        fontsize=8,
                    )
                current_state = df_sorted[name].iloc[i]
                state_index += 1

        ax.set_xticks(first_positions)
        ax.set_xticklabels(first_labels, rotation=90, fontsize=8)

        ax.set_xlabel(name)
        ax.set_ylabel(col)
        ax.set_title(f"Weighted Pen Parade of {name} by {col}")

    plt.tight_layout()
    return ax


def pengram(
    gdf,
    col,
    name,
    figsize=(8, 6),
    k=5,
    scheme="quantiles",
    xticks=True,
    legend=True,
    leg_pos="lower right",
    fmt="{:.2f}",
    query=None,
    ax=None,
    inset_size="30%",
):
    """
    Pen's Parade combined with a choropleth map.

    This function generates a Pen’s Parade plot combined with a choropleth
    map. Both plots are placed within the same subplot, with the choropleth
    map as the main plot and the Pen's Parade as an inset.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the data to plot.
    col : str
        The column to plot on the y-axis.
    name : str
        The name of the geographic units (e.g., states, regions).
    figsize : tuple, optional
        The size of the figure as a tuple (width, height). Default is (8, 6).
    k : int, optional
        Number of classes for the classification scheme. Default is 5.
    scheme : str, optional
        Classification scheme to use (e.g., 'Quantiles'). Default is
        'quantiles'.
    xticks : bool, optional
        Whether to show x-axis ticks. Default is True.
    legend : bool, optional
        Whether to show the map legend. Default is True.
    leg_pos : str, optional
        The position of the legend on the choropleth map. Default is
        "lower right".
    fmt : str, optional
        Format string for legend labels. Default is "{:.2f}".
    query : list, optional
        Specific geographic units to highlight. Default is an empty list.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes instance to plot on. If None, a new figure and axes
        will be created. Default is None.
    inset_size : str, optional
        Size of the inset plot as a percentage of the main plot. Default is "30%".

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib Axes objects for the combined choropleth and Pen's parade.
    """
    sns, mc, pd = _check_deps()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Main plot: Choropleth map
    _ = gdf.plot(
        column=col,
        scheme=scheme,
        k=k,
        ax=ax,
        legend=legend,
        legend_kwds={"loc": leg_pos, "fmt": fmt},
    )
    ax.axis("off")

    if query:
        highlight = gdf[gdf[name].isin(query)]
        highlight.boundary.plot(ax=ax, edgecolor="red", linewidth=2)

    # Inset plot: Pen's Parade
    inset_ax = inset_axes(ax, width=inset_size, height=inset_size, loc="upper right")

    binned = mc.classify(gdf[col], scheme, k=k)
    gdf["_bin"] = binned.yb

    sgdf = gdf.sort_values(by=col, ascending=True).reset_index(drop=True)

    sns.barplot(
        x=sgdf.index, y=col, hue="_bin", data=sgdf, palette="viridis", ax=inset_ax
    )
    inset_ax.set_ylabel(col)
    inset_ax.set_xlabel(name)
    plt.xticks(rotation=90)
    inset_ax.set_title("Pen's Parade", fontsize=10)

    inset_ax.set_xticks(sgdf.index)
    inset_ax.set_xticklabels(sgdf[name], rotation=90, fontsize=8)

    if not xticks:
        inset_ax.set(xticks=[])
        inset_ax.set(xlabel="")

    if query:
        for obs in query:
            if obs in sgdf[name].values:
                obs_idx = sgdf[sgdf[name] == obs].index[0]
                rect = patches.Rectangle(
                    (obs_idx - 0.5, 0),
                    1,
                    sgdf.loc[obs_idx, col],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                inset_ax.add_patch(rect)

    inset_ax.get_legend().remove()

    # plt.tight_layout()
    return ax, inset_ax
