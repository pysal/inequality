import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import mapclassify as mc


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
    #ax.get_legend().remove()

    # Adjust layout for better visualization
    plt.tight_layout()
    return fig



def pengram(gdf, col, x, figsize=(8, 6), k=5, scheme='Quantiles', xticks=True,
        leg_pos="lower right", orientation = 'r',
        fmt="{:.2f}", ratio=[3,1]):


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': ratio})

    # Plot the map on the first axis
    gdf.plot(column=col, scheme=scheme, k=k, ax=ax1, legend=True,
             legend_kwds={'loc': leg_pos,
                          'fmt': fmt},
             )
    ax1.axis('off')

    # Classify the data
    binned = mc.Quantiles(gdf[col], k=k)
    gdf['_bin'] = binned.yb

    # Sort the GeoDataFrame by the specified column
    sgdf = gdf.sort_values(by=col, ascending=True).reset_index(drop=True)

    # Create the bar plot on the second axis
    sns.barplot(x=sgdf.index, y=col, hue='_bin',
                data=sgdf, palette='viridis', ax=ax2)
    ax2.set_ylabel(col)
    ax2.set_xlabel(x)
    plt.xticks(rotation=90)
    ax2.set_title("Pen's Parade")

    ax2.set_xticks(sgdf.index)
    ax2.set_xticklabels(sgdf[x], rotation=90)
    # Optionally, remove the x-axis labels
    if not xticks:
        ax2.set(xticks=[])
        ax2.set(xlabel='')

    # Remove the legend from the barplot
    ax2.get_legend().remove()

    # Adjust layout for better visualization
    plt.tight_layout()
    return fig
