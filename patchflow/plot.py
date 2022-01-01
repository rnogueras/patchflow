#%%
"""Functions for plotting imagery and labels."""

from pathlib import Path
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import rasterio
import rasterio.plot

warnings.filterwarnings("ignore")


def plot_imagery(
    raster,
    window=None,
    bands=[1, 2, 3],
    normalize_bands=False,
    raster_shape=True,
    show_axis=False,
    ax=None,
    **kwargs,
):
    """Plot pixel value array.

    Parameters
    ----------

    raster : array, str, Path
        Image raster to plot. If string or pathlib.Path object, it will
        be interpreted as a path and open using rasterio.
    window : rasterio.windows.Window
        A rasterio window to plot only a subset of the raster. Ignored
        if the raster comes as array.
    bands : list, optional
        Define which bands will be displayed and in which order.
        Positions in the list correspond to red, green and blue
        respectively. Default: `red: 1, green: 2, blue: 3`.
    normalize_bands : bool, optional
        If true, min-max normalization is performed on each band before
        plotting. False by default.
    show_axis : bool, optional
        If true, the axis of the image will be displayed. False by
        default.
    ax : matplotlib Axes, optional
        Axes to plot on, otherwise uses current axes.
    **kwargs : key, value pairings, optional
        These will be passed to the matplotlib.pyplot.imshow
        function.
        See full list at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    Returns
    -------
    ax : matplotlib Axes
        Axes with plot.
    """

    if isinstance(raster, (str, Path)):
        with rasterio.open(raster) as dataset:
            raster = dataset.read(bands, window=window)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    if raster_shape:
        raster = rasterio.plot.reshape_as_image(raster)

    if normalize_bands:
        for index, band in enumerate(raster):
            raster[:, :, index] = rasterio.plot.adjust_band(band)

    ax.imshow(raster, **kwargs)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


def plot_labels(
    labels,
    window=None,
    ignore=[0],
    cmap="Set1",
    alpha=0.7,
    legend=True,
    label_names=None,
    show_axis=False,
    ax=None,
    **kwargs,
):
    """Plot pixel label array.

    Parameters
    ----------

    labels : array, str, Path
        Categorical raster to plot. If string or pathlib. Path object,
        it will be interpreted as a path and open using rasterio. If
        array, it is expected to have two dimensions.
    window : rasterio.windows.Window, optional
        A rasterio window to plot only a subset of the raster. Ignored
        if the raster comes as array.
    ignore : list of int, optional
        List of values that will not be displayed. Label 0 by default.
    cmap : str, cmap, optional
        cmap name or object used to display the labels. `Set1` by
        default.
    alpha : float, optional
        Real number between 0 and 1 to control the transparency of the
        displayed labels. By default, 0.7.
    legend : bool, optional
        If true, a legend showing the color of each label will be
        displayed.
    legend_names : list of str, optional
        List of names of the labels to be displayed in the legend. If
        not provided, the label values will be displayed instead.
    show_axis : bool, optional
        If true, the axis of the image will be displayed. False by
        default.
    ax : matplotlib Axes, optional
        Axes to plot on, otherwise uses current axes.
    **kwargs : key, value pairings, optional
        These will be passed to the matplotlib.pyplot.imshow
        function.
        See full list at:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    Returns
    -------
    ax : matplotlib Axes
        Axes with plot.
    """
    cmap = matplotlib.cm.get_cmap(cmap)

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "vmax" not in kwargs:
        kwargs["vmax"] = cmap.N

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as dataset:
            labels = dataset.read(1, window=window)

    if ignore is not None:
        labels = np.where(np.isin(labels, ignore), np.nan, labels)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    plt.imshow(labels, cmap=cmap, alpha=alpha, **kwargs)

    if legend:

        if label_names is None:
            non_nan_values = labels[~np.isnan(labels)]
            label_names = [str(value) for value in np.unique(non_nan_values)]

        categories = [
            matplotlib.patches.Patch([0], [0], color=color, alpha=alpha)
            for color in cmap.colors
        ]

        ax.legend(categories, label_names)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax
