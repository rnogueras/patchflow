"""Functions for plotting imagery and labels."""

from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import rasterio
import rasterio.plot

import warnings

warnings.filterwarnings("ignore")


def plot_raster(
    raster,
    window=None,
    bands=[1, 2, 3],
    show_axis=False,
    ax=None,
    **kwargs,
):
    """Plot pixel value array.

    Parameters
    ----------

    raster : array, str, Path
        Image to plot. If path as string or Path object, it will be
        open using rasterio.
    bands : list
        Define which bands will be displayed and in which order.
    reshape : bool, optional
        Rasters are usually shaped as `(bands, width, height)` while
        images often come shaped as `(width, height, channels)`. Set
        this to True if the raster array comes shaped as an image.
    ax : matplotlib Axes, optional
        Axes to plot on, otherwise uses current axes.
    **kwargs : key, value pairings, optional
        These will be passed to the rasterio.plot.show function.
        See full list at:
        https://rasterio.readthedocs.io/en/latest/api/rasterio.plot.html

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

    rasterio.plot.show(raster, ax=ax, **kwargs)

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
    show_axis=False,
    label_names=None,
    ax=None,
    **kwargs,
):

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as dataset:
            labels = dataset.read(1, window=window)

    if ignore is not None:
        labels = np.where(np.isin(labels, ignore), np.nan, labels)

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    cmap = matplotlib.cm.get_cmap(cmap)

    if "vmax" not in kwargs:
        kwargs["vmax"] = cmap.N

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
