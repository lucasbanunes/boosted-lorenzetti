
"""Utils for plotting with matplotlib.pyplot"""
from typing import Any, Dict, Optional, Union, Tuple, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
from numbers import Number

from ..norms import norm1
from ..constants import RINGS_LAYERS


def get_plt_color_cycle() -> List[str]:
    """
    Get the current color cycle from matplotlib.

    Returns
    -------
    List[str]
        A list of color codes.
    """
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


def plot_rings_profile(data: Union[pd.DataFrame, npt.NDArray[np.floating]],
                       ax: Optional[plt.Axes] = None,
                       normalize: bool = True,
                       error_margin: bool = True,
                       add_rings_labels: bool = True,
                       ax_set_kwargs: Dict[str, Any] = {},
                       **plot_kwargs
                       ) -> Tuple[plt.Axes,
                                  npt.NDArray[np.floating],
                                  npt.NDArray[np.floating]]:
    """
    Plots the mean profile of the rings in the calorimeter.

    Parameters
    ----------
    data : Union[pd.DataFrame, npt.NDArray[np.floating]]
        Dataframe containing the rings or the rings array
    ax : plt.Axes, optional
        Ax to plot the data, by default None
    normalize : bool, optional
        If True, normalizes the rings with norm1, by default True
    error_margin : bool, optional
        If True, plots the error margin as shaded area,
        by default True
    add_rings_labels : bool, optional
        If True, adds the rings labels in the plot, by default True
    plot_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.plot

    Returns
    -------
    Tuple[plt.Axes,npt.NDArray[np.floating],npt.NDArray[np.floating]]
        ax: plt.Axes
            The ax where the data was plotted
        mean: npt.NDArray[np.floating]
            The mean profile of the rings
        std: npt.NDArray[np.floating]
            The standard deviation of the rings
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    if normalize:
        data = norm1(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if ax is None:
        ax = plt.gca()
    lines = ax.plot(np.arange(len(mean)), mean, marker='o', linestyle='-',
                    **plot_kwargs)
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
    if 'title' not in ax_set_kwargs:
        if normalize:
            ax_set_kwargs['title'] = 'Normalized Rings $\\mu \\pm \\sigma$'
        else:
            ax_set_kwargs['title'] = 'Rings $\\mu \\pm \\sigma$'
    if 'xlabel' not in ax_set_kwargs:
        ax_set_kwargs['xlabel'] = 'Ring index'
    if 'ylabel' not in ax_set_kwargs:
        if normalize:
            ax_set_kwargs['ylabel'] = 'Normalized energy'
        else:
            ax_set_kwargs['ylabel'] = 'Energy'
    if 'xlim' not in ax_set_kwargs:
        ax_set_kwargs['xlim'] = (0, len(mean))
    ax.set(**ax_set_kwargs)
    return ax, mean, std


def plot_all_rings(df: Union[pd.DataFrame, npt.NDArray[np.floating]],
                   ax: Optional[plt.Axes] = None,
                   normalize: bool = True,
                   plot_kwargs: Dict[str, Any] = {},
                   ax_set_kwargs: Dict[str, Any] = {},
                   add_rings_labels: bool = True
                   ) -> plt.Axes:
    """
    Plots all the events' rings.

    Parameters
    ----------
    df : Union[pd.DataFrame, npt.NDArray[np.floating]]
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
    if 'label' in plot_kwargs:
        label = plot_kwargs.pop('label')
    else:
        label = ''
    for i, ring_vector in enumerate(rings):
        lines = ax.plot(x, ring_vector, **plot_kwargs)
    if label:
        ax.plot([], [], label=label, color=lines[0].get_color())
    if add_rings_labels:
        _, y_up = ax.get_ylim()
        for layer_name, idxs in RINGS_LAYERS.items():
            ax.axvline(idxs[0], color='black', linestyle='--')
            ax.text(idxs[0]+0.5, y_up*0.95, layer_name,
                    verticalalignment='center', fontsize=10)
    ax.axhline(0, color='black', linestyle='--')
    if 'title' not in ax_set_kwargs:
        ax_set_kwargs['title'] = 'All rings'
    if 'xlabel' not in ax_set_kwargs:
        ax_set_kwargs['xlabel'] = 'Ring index'
    if 'ylabel' not in ax_set_kwargs:
        if normalize:
            ax_set_kwargs['ylabel'] = 'Normalized energy'
        else:
            ax_set_kwargs['ylabel'] = 'Energy'
    if 'xlim' not in ax_set_kwargs:
        ax_set_kwargs['xlim'] = (0, n_rings)
    ax.set(**ax_set_kwargs)
    if label:
        ax.legend()
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
        real_max = bin_max
    else:
        real_max = data.max()
        data = data[data <= bin_max]
    if bin_min is None:
        bin_min = data.min()
        real_min = bin_min
    else:
        real_min = data.min()
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
            'Min': real_min,
            'Max': real_max
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


def plot_roc_curve(tpr: npt.NDArray[np.floating],
                   fpr: npt.NDArray[np.floating],
                   ax: Optional[plt.Axes] = None,
                   plot_kwargs: Dict[str, Any] = {},
                   add_diagonal: bool = True,
                   diagonal_kwargs: Dict[str, Any] = {},
                   axes_set: Dict[str, Any] = {}
                   ) -> List[plt.Line2D]:
    """
    Plots the ROC curve.

    Parameters
    ----------
    tpr : npt.NDArray[np.floating]
        True positive rate
    fpr : npt.NDArray[np.floating]
        False positive rate
    ax : plt.Axes, optional
        Ax to plot the data, by default plt.gca()
    plot_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.plot, by default {}
    add_diagonal : bool, optional
        If True, adds the diagonal line, by default True
    diagonal_kwargs : Dict[str, Any], optional
        Kwargs for plt.Axes.plot, by default {}
    axes_set : Dict[str, Any], optional
        Kwargs for plt.Axes.set, by default {}

    Returns
    -------
    List[plt.Line2D]
        List with the lines plotted

    """
    if ax is None:
        ax = plt.gca()
    if add_diagonal:
        if diagonal_kwargs.get('color') is None:
            diagonal_kwargs['color'] = 'black'
        if diagonal_kwargs.get('linestyle') is None:
            diagonal_kwargs['linestyle'] = '--'
        ax.plot([0, 1], [0, 1], **diagonal_kwargs)
    if axes_set.get('title') is None:
        axes_set['title'] = 'ROC curve'
    if axes_set.get('xlabel') is None:
        axes_set['xlabel'] = 'False positive rate'
    if axes_set.get('ylabel') is None:
        axes_set['ylabel'] = 'True positive rate'
    if axes_set.get('xlim') is None:
        axes_set['xlim'] = [0, 1]
    if axes_set.get('ylim') is None:
        axes_set['ylim'] = [0, 1]
    ax.set(**axes_set)
    lines = ax.plot(fpr, tpr, **plot_kwargs)
    return lines


def hist2dplot(data: pd.DataFrame,
               x: str,
               y: str,
               xlabel: str = None,
               xlim: Tuple[Number, Number] | None = None,
               ylabel: str = None,
               ylim: Tuple[Number, Number] | None = None,
               data_label: str = None,
               norm: str = 'log',
               figsize=None,
               marginal_ticks: bool = False,
               vmin=None,
               vmax=None,
               density: bool = True,
               title: str | None = None,
               corner_text: str | None = None,
               xaxis_set: Dict[str, Any] = {},
               yaxis_set: Dict[str, Any] = {},
               joint_axis_set: Dict[str, Any] = {},
               xaxis_hist_kwargs: Dict[str, Any] = {},
               yaxis_hist_kwargs: Dict[str, Any] = {},
               joint_hist_kwargs: Dict[str, Any] = {},):

    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    data_label = '' if data_label is None else data_label

    if norm == 'log':
        plot_scale = 'log'
        norm = mpl.colors.LogNorm(vmin, vmax)
    elif norm == 'linear':
        plot_scale = 'linear'
        norm = mpl.colors.Normalize(vmin, vmax)
    else:
        raise NotImplementedError(
            f'Norm {norm} not implemented. Use "log" or "linear"'
        )

    jgrid = sns.JointGrid(marginal_ticks=marginal_ticks)
    xaxis_hist_kwargs['density'] = density
    if 'color' not in xaxis_hist_kwargs:
        xaxis_hist_kwargs['color'] = 'k'
    if 'histtype' not in xaxis_hist_kwargs:
        xaxis_hist_kwargs['histtype'] = 'step'

    x_hists, xbins, x_patches = jgrid.ax_marg_x.hist(
        data[x],
        **xaxis_hist_kwargs)

    xaxis_set['yscale'] = plot_scale
    xaxis_set['xlabel'] = ''
    xaxis_set['ylabel'] = 'Density' if density else 'Counts'
    xaxis_set['xlim'] = xlim

    jgrid.ax_marg_x.set(**xaxis_set)

    yaxis_hist_kwargs['density'] = density
    # Rotates the histogram to match the joint axis
    yaxis_hist_kwargs['orientation'] = 'horizontal'
    if 'color' not in yaxis_hist_kwargs:
        yaxis_hist_kwargs['color'] = 'k'
    if 'histtype' not in yaxis_hist_kwargs:
        yaxis_hist_kwargs['histtype'] = 'step'

    y_hists, ybins, y_patches = jgrid.ax_marg_y.hist(
        data[y],
        **yaxis_hist_kwargs)

    yaxis_set['xscale'] = plot_scale
    yaxis_set['xlabel'] = 'Density' if density else 'Counts'
    yaxis_set['ylabel'] = ''
    yaxis_set['ylim'] = ylim

    jgrid.ax_marg_y.set(**yaxis_set)

    joint_hist_kwargs['density'] = density
    joint_hist_kwargs['bins'] = [
        xaxis_hist_kwargs.get('bins', None),
        yaxis_hist_kwargs.get('bins', None)
    ]
    joint_hist_kwargs['norm'] = norm
    joint_hist_kwargs['cmin'] = 1e-12

    joint = jgrid.ax_joint.hist2d(
        data[x],
        data[y],
        range=[
            jgrid.ax_marg_x.get_xlim(),
            jgrid.ax_marg_y.get_ylim()],
        **joint_hist_kwargs)

    joint_axis_set['xlabel'] = xlabel
    joint_axis_set['ylabel'] = ylabel

    if 'xlim' in xaxis_set:
        joint_axis_set['xlim'] = xlim

    if 'xlim' in yaxis_set:
        joint_axis_set['ylim'] = ylim

    # Setting the colorbar to the right, it orignaly stays between
    # the join ax and the marg_y ax
    # May become a function in the future
    cbar_ax = plt.colorbar(
        joint[-1],
        ax=jgrid.ax_joint,
        use_gridspec=True,
        fraction=0.1,
        label='Density')
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = jgrid.ax_joint.get_position()
    pos_marg_x_ax = jgrid.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    jgrid.ax_joint.set_position([
        pos_joint_ax.x0,
        pos_joint_ax.y0,
        pos_marg_x_ax.width,
        pos_joint_ax.height
    ])
    # reposition the colorbar using new x positions and y
    # positions of the joint ax
    jgrid.figure.axes[-1].set_position([
        .83,
        pos_joint_ax.y0,
        .07,
        pos_joint_ax.height
    ])

    jgrid.ax_joint.set(**joint_axis_set)
    if title:
        jgrid.figure.suptitle(title)
    if figsize:
        jgrid.figure.set_figwidth(figsize[0])
        jgrid.figure.set_figheight(figsize[1])

    if corner_text:
        jgrid.figure.text(
            0.7, 0.9,
            corner_text,
            va='top', wrap=True)

    return_dict = dict(marg_x=[x_hists, xbins, x_patches],
                       marg_y=[y_hists, ybins, y_patches],
                       joint=joint,
                       cbar_ax=cbar_ax)

    # if sub_ax_kwargs is not None:
    #     sub_ax = jgrid.figure.add_axes(sub_ax_kwargs.pop('pos'))
    #     sub_ax.hist2d(data[x], data[y], bins=[xbins, ybins],
    #                   range=[x_range, y_range],
    #                   cmin=1e-12, cmap=cmap, density=True, norm=norm)
    #     sub_ax.set(**sub_ax_kwargs)
    #     return_dict['sub_ax'] = sub_ax

    return jgrid, return_dict
