import seaborn as sns
import matplotlib.pyplot as plt
import mapclassify as mc
import matplotlib.patches as patches
import math
import pandas as pd
import numpy as np


def pen(df, col, x, ascending=True,
        xticks=True, figsize=[8, 6]):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    dbfs = df.sort_values(col, ascending=ascending).reset_index(drop=True)
    sns.barplot(x=x, y=col, data=dbfs, ax=ax)
    ax.set_ylabel(col)
    ax.set_xlabel(x)
    plt.xticks(rotation=90)
    ax.set_xticks(dbfs.index)
    ax.set_xticklabels(dbfs[x], rotation=90)
    # Optionally, remove the x-axis labels
    if not xticks:
        ax.set(xticks=[])
        ax.set(xlabel='')

    # Remove the legend from the barplot
    # ax.get_legend().remove()

    # Adjust layout for better visualization
    plt.tight_layout()
    return fig


def pengram_unified(gdf, col, name, weight=None, figsize=(8, 6), k=5,
                    scheme='Quantiles', xticks=True, leg_pos="lower right",
                    orientation='r', fmt="{:.2f}", ratio=[3, 1],
                    query=[],
                    total_bars=100):

    if weight is None:

        if orientation == 'r':
            nrow = 1
            ncol = 2
            dim_ratios = 'width_ratios'
        elif orientation == 'b':
            nrow = 2
            ncol = 1
            dim_ratios = 'height_ratios'

        fig, (ax1, ax2) = plt.subplots(nrow, ncol, figsize=figsize,
                                       gridspec_kw={dim_ratios: ratio})

        _ = gdf.plot(column=col, scheme=scheme, k=k, ax=ax1, legend=True,
                     legend_kwds={'loc': leg_pos, 'fmt': fmt})
        ax1.axis('off')

        if query:
            highlight = gdf[gdf[name].isin(query)]
            highlight.boundary.plot(ax=ax1, edgecolor='red', linewidth=2)

        binned = mc.Quantiles(gdf[col], k=k)
        gdf['_bin'] = binned.yb

        sgdf = gdf.sort_values(by=col, ascending=True).reset_index(drop=True)

        sns.barplot(x=sgdf.index, y=col, hue='_bin',
                    data=sgdf, palette='viridis', ax=ax2)
        ax2.set_ylabel(col)
        ax2.set_xlabel(name)
        plt.xticks(rotation=90)
        ax2.set_title("Pen's Parade")

        ax2.set_xticks(sgdf.index)
        ax2.set_xticklabels(sgdf[name], rotation=90)

        if not xticks:
            ax2.set(xticks=[])
            ax2.set(xlabel='')

        if query:
            for obs in query:
                if obs in sgdf[name].values:
                    obs_idx = sgdf[sgdf[name] == obs].index[0]
                    rect = patches.Rectangle((obs_idx - 0.5, 0), 1,
                                             sgdf.loc[obs_idx, col],
                                             linewidth=2, edgecolor='red',
                                             facecolor='none')

                    ax2.add_patch(rect)

        ax2.get_legend().remove()

        plt.tight_layout()
        return fig
    else:
        # Calculate number of bars for each unit
        df = gdf
        df['NumBars'] = (df[weight] / df[weight].sum() *
                         total_bars).apply(math.ceil).astype(int)

        # Create a new DataFrame with repeated rows
        repeated_rows = []
        for _, row in df.iterrows():
            repeated_rows.extend([row] * row['NumBars'])

        df_repeated = pd.DataFrame(repeated_rows)

        # Sort by the specified column
        df_sorted = df_repeated.sort_values(by=col).reset_index(drop=True)

        # Assign a unique color to each unit
        unique_states = df[name].unique()
        colors = plt.cm.get_cmap('tab20', len(unique_states))
        color_map = {state: colors(i) for i, state in enumerate(unique_states)}
        bar_colors = df_sorted[name].map(color_map)

        # Plotting
        fig, ax = plt.subplots()

        # Create the bars
        bar_positions = np.arange(len(df_sorted))
        bar_heights = df_sorted[col]
        bar_widths = 1  # Equal width for all bars

        _ = ax.bar(bar_positions, bar_heights, width=bar_widths,
                   color=bar_colors, edgecolor='black')
        tick_width = plt.rcParams['xtick.major.width']

        # Add x-ticks and labels alternatively
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
                    ax.plot([bar_positions[i], bar_positions[i]],
                            [bar_heights[i], text_y_position - 550],
                            color='black', linewidth=tick_width)
                    ax.text(bar_positions[i], text_y_position,
                            df_sorted[name].iloc[i], ha='center',
                            rotation=90, fontsize=8)
                current_state = df_sorted[name].iloc[i]
                state_index += 1

        ax.set_xticks(first_positions)
        ax.set_xticklabels(first_labels, rotation=90, fontsize=8)

        # Labeling the plot
        ax.set_xlabel(name)
        ax.set_ylabel(col)
        ax.set_title(f'Weighted Pen Parade of {name} by {col}')

        plt.show()


def pengram(gdf, col, name, figsize=(8, 6), k=5, scheme='Quantiles',
            xticks=True, leg_pos="lower right", orientation='r',
            fmt="{:.2f}", ratio=[3, 1], query=[]):

    if orientation == 'r':
        nrow = 1
        ncol = 2
        dim_ratios = 'width_ratios'
    elif orientation == 'b':
        nrow = 2
        ncol = 1
        dim_ratios = 'height_ratios'

    fig, (ax1, ax2) = plt.subplots(nrow, ncol, figsize=figsize,
                                   gridspec_kw={dim_ratios: ratio})

    _ = gdf.plot(column=col, scheme=scheme, k=k, ax=ax1, legend=True,
                 legend_kwds={'loc': leg_pos, 'fmt': fmt})
    ax1.axis('off')

    if query:
        highlight = gdf[gdf[name].isin(query)]
        highlight.boundary.plot(ax=ax1, edgecolor='red', linewidth=2)

    binned = mc.Quantiles(gdf[col], k=k)
    gdf['_bin'] = binned.yb

    sgdf = gdf.sort_values(by=col, ascending=True).reset_index(drop=True)

    sns.barplot(x=sgdf.index, y=col, hue='_bin',
                data=sgdf, palette='viridis', ax=ax2)
    ax2.set_ylabel(col)
    ax2.set_xlabel(name)
    plt.xticks(rotation=90)
    ax2.set_title("Pen's Parade")

    ax2.set_xticks(sgdf.index)
    ax2.set_xticklabels(sgdf[name], rotation=90)

    if not xticks:
        ax2.set(xticks=[])
        ax2.set(xlabel='')

    if query:
        for obs in query:
            if obs in sgdf[name].values:
                obs_idx = sgdf[sgdf[name] == obs].index[0]
                rect = patches.Rectangle((obs_idx - 0.5, 0), 1,
                                         sgdf.loc[obs_idx, col],
                                         linewidth=2, edgecolor='red',
                                         facecolor='none')

                ax2.add_patch(rect)

    ax2.get_legend().remove()

    plt.tight_layout()
    return fig


def pen_weighted(df, column, weight, name, normalize=True, total_bars=100):
    # Calculate number of bars for each unit
    df['NumBars'] = (df[weight] / df[weight].sum() *
                     total_bars).apply(math.ceil).astype(int)

    # Create a new DataFrame with repeated rows
    repeated_rows = []
    for _, row in df.iterrows():
        repeated_rows.extend([row] * row['NumBars'])

    df_repeated = pd.DataFrame(repeated_rows)

    # Sort by the specified column
    df_sorted = df_repeated.sort_values(by=column).reset_index(drop=True)

    # Assign a unique color to each unit
    unique_states = df[name].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_states))
    color_map = {state: colors(i) for i, state in enumerate(unique_states)}
    bar_colors = df_sorted[name].map(color_map)

    # Plotting
    fig, ax = plt.subplots()

    # Create the bars
    bar_positions = np.arange(len(df_sorted))
    bar_heights = df_sorted[column]
    bar_widths = 1  # Equal width for all bars

    _ = ax.bar(bar_positions, bar_heights, width=bar_widths,
               color=bar_colors, edgecolor='black')
    tick_width = plt.rcParams['xtick.major.width']

    # Add x-ticks and labels alternatively
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
                ax.plot([bar_positions[i], bar_positions[i]],
                        [bar_heights[i], text_y_position - 550],
                        color='black', linewidth=tick_width)
                ax.text(bar_positions[i], text_y_position,
                        df_sorted[name].iloc[i], ha='center',
                        rotation=90, fontsize=8)
            current_state = df_sorted[name].iloc[i]
            state_index += 1

    ax.set_xticks(first_positions)
    ax.set_xticklabels(first_labels, rotation=90, fontsize=8)

    # Labeling the plot
    ax.set_xlabel(name)
    ax.set_ylabel(column)
    ax.set_title(f'Weighted Pen Parade of {name} by {column}')

    plt.show()
