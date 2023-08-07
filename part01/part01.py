#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Jakub Kasem (xkasem02)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import List, Tuple

def integrate(x: np.array, y: np.array) -> float:
    """
    Evaluate integral defined by given arrays using formula from assignment.

    :param x: ordered vector of all integration points
    :param y: vector of integrated values f(x)
    :return: Value of integral
    """


    # None is not compatible for this function:
    if x is None or y is None:
        return None

    # Given arrays should contain more than 1 element:
    if x.size == 1:
        return None

    # Define lambda function which will multiply and sum given
    # arrays, evaluating integral:
    integralEvaluation = lambda a, b:  np.sum(np.multiply(a, b))

    # Substract Xi and Xi-1 as in assignment formula:
    a = np.subtract(x, np.roll(x, 1))

    # Add Yi-1 and Yi, divide addition by 2 as in assignment formula:
    b = np.divide(np.add(y, np.roll(y, 1)), 2)

    # Slice arrays as like indexing starts from 1.
    # Return evaluated integral:
    return integralEvaluation(a[1:], b[1:])


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Visualize function f_a(x) = a * x^2

    :param a: List of floats where every element represents coefficient 'a' in function
    :param show_figure: if True graph is shown using matplotlib.pyplot.show()
    :param save_path: if stated graph is saved on given location using matplotlib.pyplot.savefig()
    """


    # Lambda function which returns 2D array wi0values of visualized function:
    f = lambda a, x : np.multiply(np.array(a)[:, np.newaxis], np.power(x, 2))

    # Set domain of function as in assignment
    domain = np.linspace(-3, 3, 1000)

    # Set labels for legend and annotations. Values set to 1., 2. and -2. 
    # as specified in assignment:
    legendLabels = [r"y$_{1,0}$(x)", r"y$_{2,0}$(x)", r"y$_{-2,0}$(x)"]
    annotLabels = [
        r"$\int$f$_{1.0}$(x)dx", r"$\int$f$_{2.0}$(x)dx", r"$\int$f$_{-2.0}$(x)dx"]

    # Create figure and sub plot:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    # Plot every point which lambda function returns:
    ax.plot(domain, f(a, domain).T, label=legendLabels)

    # Set sub plot axis limits:
    ax.set_xlim([-3, 4])
    ax.set_ylim([-20, 20])
    # Remove last tick which will not be needed:
    ax.spines['right'].set_position(("data", 4))
    ax.set_xticks(ax.get_xticks()[:-1])

    # Annotate 3 values as specified in assignment:
    ax.annotate(annotLabels[0], xy=(3, f(a, domain)[0][-1]),  xycoords='data',
                xytext=(3, f(a, domain)[0][-1]), textcoords='data',
                horizontalalignment='left'
                )
    ax.annotate(annotLabels[1], xy=(3, f(a, domain)[1][-1]),  xycoords='data',
                xytext=(3, f(a, domain)[1][-1]), textcoords='data',
                horizontalalignment='left'
                )
    ax.annotate(annotLabels[2], xy=(3, f(a, domain)[2][-1]),  xycoords='data',
                xytext=(3, f(a, domain)[2][-1]), textcoords='data',
                horizontalalignment='left'
                )
    
    # Set axis labels as specified in assignment:
    ax.set_ylabel(r"f$_a$(x)")
    ax.set_xlabel('x')
    fig.legend(loc='upper center', ncol=3)

    # Fill color under curves:
    ax.fill_between(domain, f(a, domain)[0], alpha=0.2)
    ax.fill_between(domain, f(a, domain)[1], alpha=0.2)
    ax.fill_between(domain, f(a, domain)[2], alpha=0.2)

    # Figure showing or saving by given booleans:
    if show_figure:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    
    # Close figure
    plt.close(fig)

def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Plot three graphs, each for different function specified in assignment on domain <0, 100>.
    In third graph, part of graph which is above first graph will be different color than part
    which is under first graph.
    
    :param show_figure: if True graph is shown using matplotlib.pyplot.show()
    :param save_path: if stated graph is saved on given location using matplotlib.pyplot.savefig()
    """

    # Graph limit of y axis
    Y_LIM1 = -0.8
    Y_LIM2 = 0.8

    # Lambda functions which returns 2D array wi0values of functions
    # specified in assignment.
    # f1:
    f1 = lambda t: 0.5*np.sin(0.02*np.pi*t)
    # f2:
    f2 = lambda t: 0.25*np.sin(np.pi*t)
    # f3:
    f3 = lambda t: f1(t) + f2(t)

    def setLimits(x: tuple[float], y: tuple[float], axes: List[Axes]):
        """
        Set same limits for given axes.

        :param x: x axis limits
        :param y: y axis limits
        :param axes: axes which limits will be set
        """
        for a in axes:
            a.set_xlim(x)
            a.set_ylim(y)

    def setTicks(x: Tuple, y: Tuple, step: Tuple, axes: List[Axes]):
        """
        Set same ticks for given axes.

        :param x: tuple specifing range for x axis ticks
        :param y: tuple specifing range for y axis ticks
        :param step: tuple specifing step for x and y axis
        :param axes: axes which ticks will be set
        """
        for a in axes:
            a.xaxis.set_ticks(np.arange(x[0], x[1] + step[0], step[0]))
            a.yaxis.set_ticks(np.arange(y[0], y[1] + step[1], step[1]))



    # Set domain of functions as in assignment
    domain = np.linspace(0, 100, 10**5)

    # Create figure and sub plot:
    fig, axes = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(8,8))
    # Axes for f1, f2 and f3:
    ax1, ax2, ax12 = axes

    # Set axis limits for all graphs:
    setLimits([domain[0], domain[-1]], [Y_LIM1, Y_LIM2], [ax1, ax2, ax12])

    # Set axis labels for all graphs:
    ax1.set_ylabel(r"f$_1$(x)")
    ax1.set_xlabel('t')
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel('t')
    ax12.set_ylabel(r"f$_1$(x) + f$_2$(x)")
    ax12.set_xlabel('t')

    # Set axis ticks for all graphs:
    setTicks([domain[0], domain[-1]], [Y_LIM1, Y_LIM2], [20, 0.4], [ax1, ax2, ax12])

    # Divide f3 to 2 parts:
    above, under = f3(domain), f3(domain)
    # Calculate parth of function 3 which is above f1 graph:
    above[above <= f1(domain)] = np.nan
    # Calculate parth of function 3 which is under f1 graph:
    under[under > f1(domain)] = np.nan

    # Plot graphs for functions.
    # f1:
    ax1.plot(domain, f1(domain))
    # f2:
    ax2.plot(domain, f2(domain))
    # f3.
    # Green color for part of function which is above f1 graph:
    ax12.plot(domain, above, color='green')
    # Red color for part of function which is under f1 graph:
    ax12.plot(domain, under, color='red')

    # Figure showing or saving by given booleans:
    if show_figure:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    
    # Close figure
    plt.close(fig)


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """
    Download average temperature table from given url and return it's
    data in structures specified in assignment.
    """

    def strtoi(str):
        """
        Convert string to int.
        """
        try:
            return int(str)
        except:
            return ""

    # Return list initialization:
    retList = []

    # Use GET method on given url:
    resp = requests.get(url)
    
    # Return empty list if status code is not 200:
    if resp.status_code != 200:
        return retList
    
    # Initialize html parser:
    soup = BeautifulSoup(resp.text, "html.parser")
    
    assert(soup.body == soup.find('body'))
    assert(soup.body.div == soup.body.find('div'))
    assert(soup.body.div.div == soup.body.div.find('div'))
    assert(soup.table == soup.body.div.div.find('table'))
    
    # Find all rows in table found in html doc:
    rows = soup.table.find_all('tr')
    # Iterate through all rows:
    for row in rows:
        # Find all divisions in row:
        divisions = row.find_all('td')
        # Initialize array of temperatures:
        temps = []
        # Iterate through divisions which contain temperatures:
        for temp in divisions[2:]:
            # If temperature available, append it as float to array:
            if temp.p is not None:
                try:
                    temps.append(float(temp.text.strip().replace(',', '.')))
                except:
                    continue
        # Append specified dict to return list:
        retList.append(
                {
                    "year": strtoi(divisions[0].text.strip()),
                    "month": strtoi(divisions[1].text.strip()),
                    "temp": np.array(temps)
                }
        )
    return retList

def get_avg_temp(data, year=None, month=None) -> float:
    """
    Calculate average temperature in given data structure 
    specified in assignment.

    :param data: input data
    :param year: filter for year
    :param month: filter for month
    """
    
    # Set starter values:
    tempSum, tempCnt = 0, 0

    # Iterate through rows in given data structure:
    for row in data:
        # Filter:
        if year is not None and year != row['year']:
            continue
        if month is not None and month != row['month']:
            continue
        # Sum values of temperatures and add them to overall summary: 
        tempSum += row['temp'].sum()
        # Add record count to overall summary:
        tempCnt += row['temp'].size
    
    # Return average:
    return np.divide(tempSum, tempCnt)
