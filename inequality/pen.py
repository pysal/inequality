import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import mapclassify as mc
import matplotlib.patches as patches


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


def pengram(gdf, col, name, figsize=(8, 6), k=5, scheme='Quantiles', xticks=True,
            leg_pos="lower right", orientation='r', fmt="{:.2f}", ratio=[3, 1],
            query=[]):

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

    base = gdf.plot(column=col, scheme=scheme, k=k, ax=ax1, legend=True,
                    legend_kwds={'loc': leg_pos, 'fmt': fmt})
    ax1.axis('off')

    if query:
        highlight = gdf[gdf[name].isin(query)]
        highlight.boundary.plot(ax=ax1, edgecolor='red', linewidth=2)

    binned = mc.Quantiles(gdf[col], k=k)
    gdf['_bin'] = binned.yb

    sgdf = gdf.sort_values(by=col, ascending=True).reset_index(drop=True)

    sns.barplot(x=sgdf.index, y=col, hue='_bin', data=sgdf, palette='viridis', ax=ax2)
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
                rect = patches.Rectangle((obs_idx - 0.5, 0), 1, sgdf.loc[obs_idx, col],
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax2.add_patch(rect)

    ax2.get_legend().remove()

    plt.tight_layout()
    return fig

