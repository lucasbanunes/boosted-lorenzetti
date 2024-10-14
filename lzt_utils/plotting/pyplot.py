
"""Utils for plotting with matplotlib.pyplot"""
from typing import Any, Dict, Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from hep_utils.formulas import norm1
from hep_utils.constants import RINGS_LAYERS
import pandas as pd
import numpy as np
import numpy.typing as npt


def get_plt_color_cycle() -> List[str]:
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def errorarea(x, y, yerr,
              ax: Optional[plt.Axes] = None,
              plot_kwargs: Dict[str, Any] = {},
              fill_between_kwargs: Dict[str, Any] = {}) -> plt.Axes:
    """
    Plots an error area in the plot.

    Parameters
    ----------
    x : np.array
        x values
    y : np.array
        y values
    yerr : np.array
        y errors
    ax : plt.Axes, optional
        Ax to plot the data, by default None
    fill_between_kwargs
        Kwargs for plt.fill_between

    Returns
    -------
    plt.Axes
        The ax where the data was plotted
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plot_kwargs)
    ax.fill_between(x, y-yerr, y+yerr, **fill_between_kwargs)
    return ax


def plot_rings_profile(df: Union[pd.DataFrame, npt.NDArray[np.float_]],
                       ax: Optional[plt.Axes] = None,
                       normalize: bool = True,
                       error_margin: bool = True,
                       label: str = '',
                       add_rings_labels: bool = True
                       ) -> Tuple[plt.Axes,
                                  npt.NDArray[np.float_],
                                  npt.NDArray[np.float_]]:
    """
    Plots the mean profile of the rings in the calorimeter.

    Parameters
    ----------
    df : Union[pd.DataFrame, npt.NDArray[np.float_]]
        Dataframe containing the rings or the rings array
    ax : plt.Axes, optional
        Ax to plot the data, by default None
    normalize : bool, optional
        If True, normalizes the rings with norm1, by default True
    error_margin : bool, optional
        If True, plots the error margin as shaded area,
        by default True
    label : str, optional
        Label for the plot, by default ''
    add_rings_labels : bool, optional
        If True, adds the rings labels in the plot, by default True

    Returns
    -------
    Tuple[plt.Axes,npt.NDArray[np.float_],npt.NDArray[np.float_]]
        ax: plt.Axes
            The ax where the data was plotted
        mean: npt.NDArray[np.float_]
            The mean profile of the rings
        std: npt.NDArray[np.float_]
            The standard deviation of the rings
    """
    if isinstance(df, pd.DataFrame):
        rings = df.values
    if normalize:
        rings = norm1(rings)
    mean = rings.mean(axis=0)
    std = rings.std(axis=0)
    if ax is None:
        ax = plt.gca()
    lines = ax.plot(np.arange(len(mean)), mean,
                    label=label, marker='o', linestyle='-')
    if error_margin:
        ax.fill_between(np.arange(len(mean)), mean - std, mean + std,
                        facecolor=lines[0].get_color(), alpha=0.25)
    if add_rings_labels:
        _, y_up = ax.get_ylim()
        for layer_name, idxs in RINGS_LAYERS.items():
            ax.axvline(idxs[0], color='black', linestyle='--')
            ax.text(idxs[0]+0.5, y_up*0.95, layer_name,
                    verticalalignment='center', fontsize=10)
    ax.axhline(0, color='black', linestyle='--')
    ax.legend()
    ax.set_title('Rings mean profile')
    ax.set_xlim(0, len(mean))
    ax.set_xlabel('Ring index')
    ax.set_ylabel('Normalized energy')
    return ax, mean, std


def plot_all_rings(df: Union[pd.DataFrame, npt.NDArray[np.float_]],
                   ax: Optional[plt.Axes] = None,
                   normalize: bool = True,
                   plot_kwargs: Dict[str, Any] = {},
                   add_rings_labels: bool = True
                   ) -> plt.Axes:
    """
    Plots all the events' rings.

    Parameters
    ----------
    df : Union[pd.DataFrame, npt.NDArray[np.float_]]
        Dataframe containing the rings or the rings array
    ax : plt.Axes, optional
        Ax to plot the data, by default None
    normalize : bool, optional
        If True, normalizes the rings with norm1, by default True
        by default True
    plot_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.plot, by default {}
    add_rings_labels : bool, optional
        If True, adds the rings labels in the plot, by default True

    Returns
    -------
    plt.Axes
        The ax where the data was plotted
    """
    if isinstance(df, pd.DataFrame):
        rings = df.values
    if normalize:
        rings = norm1(rings)
    if ax is None:
        ax = plt.gca()

    if plot_kwargs.get('color') is None:
        plot_kwargs['color'] = get_plt_color_cycle()[0]

    n_rings = rings.shape[1]
    x = np.arange(n_rings)
    for i, ring_vector in enumerate(rings):
        ax.plot(x, ring_vector, **plot_kwargs)
    if add_rings_labels:
        _, y_up = ax.get_ylim()
        for layer_name, idxs in RINGS_LAYERS.items():
            ax.axvline(idxs[0], color='black', linestyle='--')
            ax.text(idxs[0]+0.5, y_up*0.95, layer_name,
                    verticalalignment='center', fontsize=10)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('All Rings')
    ax.set_xlim(0, n_rings)
    ax.set_xlabel('Ring index')
    if normalize:
        ax.set_ylabel('Normalized energy')
    else:
        ax.set_ylabel('Energy')
    return ax


def histplot(data: Union[pd.Series, npt.NDArray[np.number]],
             nbins: int = 100,
             bin_max: Optional[float] = None,
             bin_min: Optional[float] = None,
             ax: Optional[plt.Axes] = None,
             metrics: bool = False,
             legend_kwargs: Dict[str, Any] = {},
             ax_set: Dict[str, Any] = {},
             hist_kwargs: Dict[str, Any] = {}
             ) -> Tuple[plt.Axes, Dict[str, Any]]:
    """
    Plots a histogram of the data.

    Parameters
    ----------
    data : Union[pd.Series, npt.NDArray[np.number]]
        Data to plot
    nbins : int, optional
        Number of equally spaced bins, by default 100.
        If bins arg is passed in hist_kwargs, this is ignored.
    bin_max : float, optional
        Maximum value of the bins, by default None.
        If bin_max is not passed the max value from the data is used.
    bin_min : float, optional
        Minimum value of the bins, by default None.
        If bin_min is not passed the min value from the data is used.
    ax : plt.Axes, optional
        Ax to plot the data, by default plt.gca()
    metrics : bool, optional
        If True writes mean, std, min, max and samples values in the legend,
        by default False
    legend_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.legend, by default {}.
        Used only when metrics is True.
    ax_set : Dict[str, Any], optional
        Kwargs for plt.Axes.set, by default {}
    hist_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.hist, by default {}

    Returns
    -------
    Tuple[plt.Axes, Dict[str, Any]]
        plt.Axes
            The ax where the data was plotted
        Dict[str, Any]
            Dictionary with the metrics of the data.
            Empty dict if metrics is False.
    """
    if ax is None:
        ax = plt.gca()
    if bin_max is None:
        bin_max = data.max()
    else:
        data = data[data <= bin_max]
    if bin_min is None:
        bin_min = data.min()
    else:
        data = data[data >= bin_min]
    if hist_kwargs.get('bins') is None:
        hist_kwargs['bins'] = np.linspace(bin_min, bin_max, nbins)
    ax.hist(data, **hist_kwargs)
    ax.set(**ax_set)
    if metrics:
        metrics_dict = {
            'Samples': len(data),
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': bin_min,
            'Max': bin_max
        }
        for key, value in metrics_dict.items():
            if isinstance(value, (int, np.integer)):
                ax.plot([], [], ' ', label=f'{key}: {value}')
            else:
                ax.plot([], [], ' ', label=f'{key}: {value:.2f}')
        ax.legend(**legend_kwargs)
        return ax, metrics_dict
    else:
        return ax, {}


def categorical_histplot(data: Union[pd.Series, npt.NDArray[np.number]],
                         ax: Optional[plt.Axes] = None,
                         metrics: bool = False,
                         percentage: bool = False,
                         legend_kwargs: Dict[str, Any] = {},
                         ax_set: Dict[str, Any] = {},
                         bar_kwargs: Dict[str, Any] = {}
                         ) -> Tuple[plt.Axes, Dict[str, Any]]:
    """
    Plots a categorical histogram of the data.
    This hist differs from histplot because it counts the number of samples
    and the ticks are centered.

    Parameters
    ----------
    data : Union[pd.Series, npt.NDArray[np.number]]
        Data to plot
    ax : plt.Axes, optional
        Ax to plot the data, by default plt.gca()
    metrics : bool, optional
        If True writes mean, std, min, max and samples values in the legend,
        by default False
    percentage : bool, optional
        If True, the y-axis is in percentage, by default False
    legend_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.legend, by default {}.
        Used only when metrics is True.
    ax_set : Dict[str, Any], optional
        Kwargs for plt.Axes.set, by default {}
    bar_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.bar, by default {}

    Returns
    -------
    Tuple[plt.Axes, Dict[str, Any]]
        plt.Axes
            The ax where the data was plotted
        Dict[str, Any]
            Dictionary with the metrics of the data.
            Empty dict if metrics is False.
    """
    if ax is None:
        ax = plt.gca()
    categories, counts = np.unique(data, return_counts=True)
    n_samples = np.sum(counts)
    if percentage:
        counts = 100*counts/n_samples
    bar_kwargs['align'] = 'center'
    ax.bar(categories, counts, **bar_kwargs)
    ax.set(**ax_set)
    if metrics:
        metrics_dict = {
            'Samples': n_samples,
        }
        for key, value in metrics_dict.items():
            ax.plot([], [], ' ', label=f'{key}: {value}')
        ax.legend(**legend_kwargs)
        return ax, metrics_dict
    else:
        return ax, {}
