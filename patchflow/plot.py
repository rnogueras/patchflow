"""Functions for plotting imagery and labels."""
from pathlib import Path
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import rasterio
import rasterio.plot

from raster import get_raster_proportions

# TODO: Catch the warning isolatedly
warnings.filterwarnings("ignore")


color_dict = dict(
    blue="#577590",
    turquoise="#43AA8B",
    green="#90BE6D",
    dark_green="#104547",
    red="#F94144",
    yellow="#F8961E",
)

STANDARD_CMAP = matplotlib.colors.ListedColormap(list(color_dict.values()))


def plot_imagery(
    raster,
    window=None,
    bands=[1, 2, 3],
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
        with rasterio.open(raster) as src:
            raster = src.read(bands, window=window)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    if raster_shape:
        raster = rasterio.plot.reshape_as_image(raster)

    ax.imshow(raster, **kwargs)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


def plot_labels(
    labels,
    window=None,
    ignore=[],
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
    legend : bool, optional
        If true, a legend showing the color of each label will be
        displayed.
    label_names : list of str, optional
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

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"
        
    if "cmap" not in kwargs:
        kwargs["cmap"] = STANDARD_CMAP
        
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.7

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)

    labels = labels.squeeze()
    
    label_values = np.unique(labels)
    cmap_index = label_values / (len(label_values) - 1)

    if ignore:
        # set alphas of ignored values to 0
        cmap_array = kwargs["cmap"](cmap_index)
        cmap_array[np.isin(label_values, ignore), -1] = 0 
        kwargs["cmap"] = matplotlib.colors.ListedColormap(cmap_array)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    ax.imshow(labels, **kwargs)

    if legend:

        cmap_valid_indexes = cmap_index[~np.isin(label_values, ignore)]
        valid_color_codes = kwargs["cmap"](cmap_valid_indexes)

        categories = [
            matplotlib.patches.Patch(
                [0], [0], color=color_code, alpha=kwargs["alpha"]
            )
            for color_code in valid_color_codes
        ]
        
        if label_names is None:
            label_names = label_values[~np.isin(label_values, ignore)]

        ax.legend(categories, label_names)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


# TODO: docstring
def plot_histogram(
    raster,
    window=None,
    bands=[1, 2, 3],
    show_axis=True,
    ax=None,
    **kwargs,
):

    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"

    if "color" not in kwargs:
        kwargs["color"] = [
            color_dict["red"], color_dict["green"], color_dict["blue"]
        ]

    if "stacked" not in kwargs:
        kwargs["stacked"] = True
    
    if "fill" not in kwargs:
        kwargs["fill"] = True
    
    if isinstance(raster, (str, Path)):
        with rasterio.open(raster) as src:
            raster = src.read(bands, window=window)
    
    # TODO: Handle nodata here, convert to nans and then ignore them
    value_range = np.nanmin(raster), np.nanmax(raster)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    ax.hist(
        x=raster.reshape(raster.shape[0], -1).T,
        range=value_range,
        **kwargs,
    )

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


# TODO: docstring
# TODO: fix x axis (it shows a continuous variable instead of a discrete one)
def plot_proportions(
    labels,
    cmap=STANDARD_CMAP,
    window=None,
    ax=None,
    show_axis=True,
    **kwargs,
):
    
    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)
        
    class_proportions = get_raster_proportions(labels)
    labels = list(class_proportions.keys())
    proportions = list(class_proportions.values())
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    cmap = cmap.reversed()
        
    show = False
    if not ax:
        show = True
        ax = plt.gca()
    
    ax.bar(
        x=labels, 
        height=proportions,
        color=cmap(rescale(proportions)),
        **kwargs
    )
    
    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()


# TODO: docstring
def describe(
    raster,
    labels,
    window=None,
    bands=[1, 2, 3],
    figure_size=(10, 8),
    raster_params={},
    label_params={},
    hist_params={},
    bar_params={},
) -> None:
    """Descriptive plot about the image and labels provided."""

    if "ignore" not in label_params:
        label_params["ignore"] = []
    
    if "legend" not in label_params:
        label_params["legend"] = False
        
    if "alpha" not in label_params:
        label_params["alpha"] = 1
        
    if isinstance(raster, (str, Path)):
        with rasterio.open(raster) as src:
            raster = src.read(bands, window=window)
            
    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)

    plt.figure(figsize=figure_size)

    ax_1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=3)
    plot_imagery(raster=raster, ax=ax_1, **raster_params)

    ax_2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=3)
    plot_labels(labels=labels, ax=ax_2, **label_params)

    ax_3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    plot_histogram(raster=raster, ax=ax_3, **hist_params)

    ax_4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    plot_proportions(labels=labels, ax=ax_4, **bar_params)
    
    plt.tight_layout()
    plt.show()
