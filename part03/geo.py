#!/usr/bin/python3.10
# coding=utf-8
#%%
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
# muzete pridat vlastni knihovny
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def figureShow(show: bool, fig: plt.figure):
    """Show given figure.

    Args:
        show (bool): show or not.
        fig (plt.figure): figure to show.
    """
    if show == True:
        #plt.show()
        fig.show()

def figureSave(loc: str, fig: plt.figure):
    """Save given figure.

    Args:
        loc (str): saving location.
        fig (plt.figure): figure to save.
    """
    if loc is not None:
        fig.savefig(loc)

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""

    # Create GeoDataFrame. Columns 'd' and 'e' are x and y coordinates:
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs="EPSG:5514")

    # Delete rows where are invalid cooradinate points:
    gdf = gdf[~(gdf['geometry'].is_empty | gdf['geometry'].isna())].reset_index()

    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami s alkoholem pro roky 2018-2021 """

    years = [2018, 2019, 2020, 2021]

    _gdf = gdf.copy()

    # Datetime accessor will be used:
    _gdf["p2a"] = pd.to_datetime(_gdf["p2a"])

    # Filter DataFrame on conditions specified in assignmnent:
    _gdf = _gdf[(_gdf['region'] == 'JHM') & (_gdf['p11'] >= 3) & (_gdf['p2a'].dt.year.isin(years))]

    # Graph will be plotted for each year:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot graphs:
    year_iter = iter(years)
    for row in axes:
        for ax in row:
            year = next(year_iter)
            _gdf[_gdf['p2a'].dt.year == year].plot(ax=ax, markersize=10)
            contextily.add_basemap(ax, crs=_gdf.crs.to_string(),
                                    source=contextily.providers.OpenStreetMap.DE)
            ax.set_title(str(year) + ' JHM kraj')
            ax.set_xticks([])
            ax.set_yticks([])

    # Figure show handling:
    figureShow(show_figure, fig)

    # Figure save handling:
    figureSave(fig_location, fig)
    
def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """

    roads = [1, 2, 3]

    _gdf = gdf.copy()

    # Filter DataFrame on conditions specified in assignmnent:
    _gdf = _gdf[(_gdf['p36'].isin(roads)) & (_gdf['region'] == 'JHM')]

    # Use coordinates to train model: 
    x = pd.Series(_gdf['geometry'].apply(lambda p: p.x))
    y = pd.Series(_gdf['geometry'].apply(lambda p: p.y))
    X = np.column_stack((x, y))

    # Train model:
    model = sklearn.cluster.KMeans(n_clusters = 20, random_state = 1).fit(X)
    #model = sklearn.cluster.MiniBatchKMeans(n_clusters = 20).fit(X)

    # Column representing clusters:
    _gdf['cluster'] = model.labels_

    # Count accidents in clusters:
    _gdf['cluster_value'] = _gdf['cluster'].map(_gdf['cluster'].value_counts())

    # One plot for accidents count in chosen region:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title("Nehody v JHM kraji na silnicích 1., 2. a 3. třídy")
    ax.set_axis_off()

    # Plot number of accidents in clusters:
    _gdf.plot(ax=ax, markersize=3, column="cluster_value", legend=True)
    contextily.add_basemap(ax, crs=_gdf.crs.to_string(), 
                            source=contextily.providers.OpenStreetMap.DE)

    # Figure show handling:
    figureShow(show_figure, fig)

    # Figure save handling:
    figureSave(fig_location, fig)
    
#%%
if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl"))
    plot_geo(gdf, "geo1.png")
    plot_cluster(gdf, "geo2.png")
