#!/usr/bin/env python3.9
# coding=utf-8
# %%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    """Load CSV files data into Pandas DataFrame from ZIP file 
    with structure specified in assignment. Add column region
    representing abbreviation of given region.

    Args:
        filename (str): ZIP file name

    Returns:
        pd.DataFrame: DataFrame containing data from ZIP
         with additional column region
    """

    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    def getRegion(csvName: str) -> str:
        """Get region abbreviation of given CSV file from dict 'regions'.

        Args:
            csvName (str): CSV file name

        Returns:
            str: region abbreviation or empty string 
        """

        # If regions dict values contain given csv name
        # return corresponding key:
        for k in regions.keys():
            if regions[k] == csvName[:-4]:
                return k

        # If regions dict does not contain given csv name
        # return empty string:
        return ""

    def get_dataframe(filename: str) -> pd.DataFrame:
        """Load CSV files from ZIP and concatenate their data in DataFrame.

        Args:
            filename (str): ZIP file name

        Returns:
            pd.DataFrame: DataFrame containing data from ZIP
        """

        # List which will contain DataFrames:
        dfList = []

        # Open ZIP:
        with zipfile.ZipFile(filename) as dataZip:
            # Open each sub zip:
            for yearZipName in dataZip.namelist():
                with zipfile.ZipFile(dataZip.open(yearZipName)) as yearZip:
                    # Open each CSV in ZIP except chodci:
                    for csvName in yearZip.namelist():
                        # Get region abbreviation for CSV file:
                        region = getRegion(csvName)
                        # Go to next CSV if region is empty string:
                        if region == "":
                            continue
                        with yearZip.open(csvName) as csv:
                            # Read CSV into DataFrame:
                            df = pd.read_csv(csv, sep=';', header=None, names=headers,
                                             dtype='unicode', encoding='cp1250')
                            # Add column region representing region abbreviation:
                            df["region"] = region
                            # Append DataFrame to list:
                            dfList.append(df)

        # Return DataFrame concatenated of data of every CSV:
        return pd.concat(dfList, axis=0)

    # Return DataFrame containing data from ZIP file:
    return get_dataframe(filename)

# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): DataFrame loaded by load_data()
        verbose (bool, optional): If True compute deep memory size of given
            and returned DataFrame columns. Defaults to False.

    Returns:
        pd.DataFrame: modified given DataFrame as new instance
    """

    intCols = ("p1", "p13a", "p13b", "p13c","p14", "p33d", 
                "p34", "p53", "n", "r", "s", "l",)
    floatCols = ("a", "b", "d", "e", "f", "g")
    stringCols = ("h", "i", "p2b")
    notCategoryCols = ("date", "p47")

    # Create new DataFrame containing data of given DataFrame:
    _df = df.copy()

    # Add column "date" with datetime64 type:
    _df["date"] = pd.to_datetime(_df["p2a"])

    # Remove duplicates by column "p1":
    _df = _df.drop_duplicates(subset=["p1"])

    # If needed change column's datatype:
    for col in _df:
        # Numeric columns:
        if col in intCols or col in floatCols:
            _df[col] = _df[col].replace(',', '.', regex=True)
            _df[col] = pd.to_numeric(_df[col], errors='coerce')
            continue
        # String columns:
        if col in stringCols:
            continue
        # If column has reasonably little unique
        #  values use category datatype:
        if _df[col].nunique() < 500:
            # Except specified columns:
            if col not in notCategoryCols:
                _df[col] = _df[col].astype("category")

    # If True compute deep memory size of given
    # and modified DataFrames:
    if verbose == True:
        # Given size:
        oSize = np.round(df.memory_usage(deep=True).sum() / np.power(10, 6), 2)
        # Modified size:
        nSize = np.round(_df.memory_usage(
            deep=True).sum() / np.power(10, 6), 2)
        # Print sizes:
        print(f'orig_size={str(oSize)} MB')
        print(f'new_size={str(nSize)} MB')

    # Return parsed DataFrame:
    return _df

# Ukol 3: počty nehod v jednotlivých regionech podle viditelnosti


def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """Plot accidents count by visibility in chosen regions.

    Args:
        df (pd.DataFrame): parsed dataframe.
        fig_location (str, optional): Save figure to file if given. Defaults to None.
        show_figure (bool, optional): Show figure. Defaults to False.
    """

    # Chosen regions:
    regions = ['JHC', 'JHM', 'PAK', 'LBK']

    # Dictionary used for categorizing:
    visibility = {
        '1': 've dne - nezhoršená',
        '2': 've dne - zhoršená',
        '3': 've dne - zhoršená',
        '4': 'v noci - zhoršená',
        '5': 'v noci - nezhoršená',
        '6': 'v noci - zhoršená',
        '7': 'v noci - nezhoršená',
    }

    # Work with copy of given dataframe:
    _df = df.copy()

    # Keep records of chosen regions:
    _df = _df[_df['region'].isin(regions)]
    _df['region'] = _df['region'].cat.remove_unused_categories()

    # Categorize p19 col data using dict visibility:
    _df['visibility'] = pd.CategoricalIndex(_df['p19'].copy()).map(visibility)

    # Keep important columns:
    _df = _df[["visibility", "region"]]

    # Add column for counting rows after grouping:
    _df['count'] = 1

    # Group and count rows by visibility and region:
    _df = _df.groupby(['region', 'visibility']).agg('count').reset_index()

    # Barplot count of accidents based on visibility in chosen regions.
    # Four graphs will by displayed, 2 in row: 
    g = sns.catplot(x='visibility', y='count', col='region', data=_df, kind='bar',
                     col_wrap=2, units='count', saturation=0.5, aspect=1.3)

    # Region name as title of graph:
    g.set_titles("Kraj: {col_name}")
    # Set axis labels. 
    # Set rotation for tick labels, due to long text:
    g.tick_params(axis='x', **{"labelrotation": 40})
    for ax in g.axes.flat:
        ax.set_xlabel('Viditelnost')
        ax.set_ylabel('Počet nehod')
        ax.xaxis.label.set_visible(True)
        ax.yaxis.label.set_visible(True)
        ax.tick_params(labelleft=True, labelbottom=True)
        # Graph border:
        for spine in ax.spines.values():
            spine.set_visible(True)
        # Values on top of bars:
        for container in ax.containers:
            ax.bar_label(container, )
    g.tight_layout()

    #sns.set(rc={'figure.figsize':(8, 8)})
    plt.rcParams['figure.figsize'] = (8, 8)
    # Figure show handling:
    if show_figure == True:
        plt.show()
    
    # Figure save handling:
    if fig_location is not None:
        g.savefig(fig_location)

# Ukol4: druh srážky jedoucích vozidel


def plot_direction(df: pd.DataFrame, fig_location: str = None,
                   show_figure: bool = False):
    """Plot accidents count in chosen regions based on collision kind
    over months.

    Args:
        df (pd.DataFrame): parsed dataframe.
        fig_location (str, optional): Save figure to file if given. Defaults to None.
        show_figure (bool, optional): Show figure. Defaults to False.
    """

    # Chosen regions:
    regions = ['JHC', 'JHM', 'PAK', 'LBK']

    # Dictionary used for categorizing:
    collision = {
        "1": "čelní",
        "2": "boční",
        "3": "boční",
        "4": "zezadu",
    }

    # Work with copy of given dataframe:
    _df = df.copy()

    # Keep records of chosen regions:
    _df = _df[_df['region'].isin(regions)]
    _df['region'] = _df['region'].cat.remove_unused_categories()

    # Value 0 in col p7 will not be included(assingment):
    _df = _df[_df['p7'] != "0"]

    # Categorize p7 col data using dict collision:
    _df['collision'] = pd.CategoricalIndex(_df['p7'].copy()).map(collision)

    # Accident month will be needed for plot:
    _df['month'] = _df['date'].dt.month

    # Keep important columns:
    _df = _df[['collision', 'region', 'month']]

    # Add column for counting rows after grouping:
    _df['count'] = 1

    # Count accidents for every month in regions by collision:
    _df = _df.groupby(['region', 'month', 'collision']).agg('count').reset_index()

    # Barplot count of accidents based on collision side in chosen regions over months.
    # Four graphs will by displayed, 2 in row: 
    sns.set_style("darkgrid")
    g = sns.catplot(x='month', y='count', hue='collision' ,col='region', data=_df, 
                    kind='bar', col_wrap=2, saturation=0.5, aspect=1.3, sharex=True, sharey=True)

    # Set titles and ticks:
    g.set_titles("Kraj: {col_name}")
    for ax in g.axes.flat:
        ax.set_xlabel('Viditelnost')
        ax.set_ylabel('Počet nehod')
        ax.xaxis.label.set_visible(True)
        ax.yaxis.label.set_visible(True)
        ax.tick_params(labelleft=True, labelbottom=True)
    g.tight_layout()

    sns.set(rc={'figure.figsize':(8, 8)})
    plt.rcParams['figure.figsize'] = (8, 8)
    # Figure show handling:
    if show_figure == True:
        plt.show()
    
    # Figure save handling:
    if fig_location is not None:
        g.savefig(fig_location)

    sns.set_style(None)

# Ukol 5: Následky v čase


def plot_consequences(df: pd.DataFrame, fig_location: str = None,
                      show_figure: bool = False):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        fig_location (str, optional): _description_. Defaults to None.
        show_figure (bool, optional): _description_. Defaults to False.
    """

    # Chosen regions:
    regions = ['JHC', 'JHM', 'PAK', 'LBK']

    # Tuple used for categorizing:
    collision = {
        "a": "Usmrcení", 
        "b": "Težké zranení",
        "c": "Lehké zranení"
        }

    # Work with copy of given dataframe:
    _df = df.copy()

    # Keep records of chosen regions:
    _df = _df[_df['region'].isin(regions)]
    _df['region'] = _df['region'].cat.remove_unused_categories()

    # Remove rows where all of cols p13a, p13b, p13c contain 0:
    _df = _df[(_df['p13a'] != 0) | (_df['p13b'] != 0) | (_df['p13c'] != 0)]

    def worstConsequence(p13a: int, p13b: int) -> str:
        """Determine collision type based on worst consequence.

        Args:
            p13a (int): number of dead people
            p13b (int): number of heavy injured people

        Returns:
            str: last char of worst consequence column name.
        """

        if p13a != 0:
            return 'a'
        if p13a == 0 and p13b != 0:
            return 'b'
        return 'c'

    # Specify collision kind based on worst consequence:
    _df['collision'] = _df.apply(lambda x: worstConsequence(x['p13a'], x['p13b']), axis=1)

    # Assign str representation to worst consequence:
    _df['collision'] = _df['collision'].astype("category")
    _df['collision'] = _df['collision'].map(collision)

    # Keep only necessary columns:
    _df = _df[['date', 'region', 'collision']]

    # Additional row for grouping:
    _df['count'] = 0

    # Sum different kinds of collisions for every date and region:
    _df = pd.pivot_table(_df, columns='collision', index=['date', 'region'], values='count', aggfunc='count')

    # Sum collision for region in month rate:
    _df = _df.reset_index(level=[1]).groupby(['region']).resample('M').sum(numeric_only=True).reset_index(level=0)

    # Lineplot different accidents kind count:
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # For every region:
    for ax, reg in zip(axes.flat, regions):
        sns.lineplot(data=_df[_df['region'] == reg], ax=ax)
        # Set labels and ticks. Limit x and y axes:
        ax.xaxis.label.set_visible(False)
        ax.set_ylabel('Počet nehod')
        ax.set_title("Kraj: {}".format(reg))
        ax.set_xlim(np.array(['2016-01-01', '2022-01-01'], dtype="datetime64[D]"))
        ax.set_ylim(np.array([0, 280]))
        # Format X ticks to mm/YY:
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([pd.to_datetime(tm, unit='D').strftime(r'%m/%y') for tm in xticks])
    
    # Figure legend:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.legend().remove()
    axes.flat[3].legend(handles, labels, loc="lower left", bbox_to_anchor=(1,1))

    # Figure show handling:
    if show_figure == True:
        plt.show()
    
    # Figure save handling:
    if fig_location is not None:
        fig.savefig(fig_location)

    sns.set_style(None)

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data/data.zip")
    df2 = parse_data(df, True)
    # %%

    plot_visibility(df2, "01_visibility.png", True)
    plot_direction(df2, "02_direction.png", True)
    plot_consequences(df2, "03_consequences.png", True)


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku

# %%
