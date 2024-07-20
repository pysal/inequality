import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import mapclassify as mc


def pen(gdf, col, x, figsize=(8, 6), k=5, scheme='Quantiles', xticks=True,
        leg_pos="lower right"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # Plot the map on the first axis
    gdf.plot(column=col, scheme=scheme, k=k, ax=ax1, legend=True,
             legend_kwds={'loc': leg_pos})
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
