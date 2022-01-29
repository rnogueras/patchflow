"""Functions for plotting imagery and labels."""
from typing import Optional, Sequence, Any, Type
from pathlib import Path
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import rasterio
import rasterio.plot

from patchflow.raster import RasterSourceType, WindowType, get_raster_proportions

# TODO: Catch the warning isolatedly
warnings.filterwarnings("ignore")

ColorMapType = Type[matplotlib.colors.ListedColormap]

COLOR_CODES = dict(
    blue="#577590",
    turquoise="#43AA8B",
    green="#90BE6D",
    dark_green="#104547",
    red="#F94144",
    yellow="#F8961E",
)

STANDARD_CMAP = matplotlib.colors.ListedColormap(list(COLOR_CODES.values()))


def plot_imagery(
    imagery: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    raster_shape: bool = True,
    show_axis: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot pixel value array.

    Args:
        imagery: Imagery raster to plot. If a string or a Path is
            provided, it will be interpreted as a path to the file
            and open using rasterio.
        window: A rasterio window to plot only a subset of the raster.
            Ignored if the raster comes as array.
        bands: Define which bands will be displayed and in which order.
            Positions in the list correspond to red, green and blue
            respectively. Default: [1, 2, 3].
        raster_shape: Whether the raster dimensions match the raster
            standard (band, width, height). Default: True.
        show_axis: If true, the axis of the image will be displayed.
            False by default.
        ax: Axes to plot on. Otherwise, use current axes.
        **kwargs: These will be passed to the matplotlib`s imshow function.
            See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    Returns:
        Axes with plot.
    """

    if isinstance(imagery, (str, Path)):
        with rasterio.open(imagery) as src:
            imagery = src.read(bands, window=window)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    if raster_shape:
        imagery = rasterio.plot.reshape_as_image(imagery)

    ax.imshow(imagery, **kwargs)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


def plot_labels(
    labels: RasterSourceType,
    window: Optional[WindowType] = None,
    transparent: Optional[Sequence[int]] = None,
    legend: bool = True,
    label_names: Optional[Sequence[str]] = None,
    show_axis: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot pixel label array.

    Args:
        labels: Discrete raster to plot. If string or pathlib. Path object,
            it will be interpreted as a path and open using rasterio. If
            array, it is expected to have two dimensions.
        window: A rasterio window to plot only a subset of the raster. Ignored
            if the raster comes as array.
        transparent: List of label values to make transparent. The alpha
            of this values will be set to 0.
        legend: If true, a legend showing the color of each label will be
            displayed.
        label_names: List of names of the labels to be displayed in the legend.
            If not provided, the label values will be displayed instead.
        show_axis: If true, the axis of the image will be displayed. False by
            default.
        ax: Axes to plot on, otherwise uses current axes.
        kwargs: These will be passed to the matplotlib`s imshow function.
            See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    Returns:
        Axes with plot.
    """

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "cmap" not in kwargs:
        kwargs["cmap"] = STANDARD_CMAP

    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.7
        
    # TODO: Add transparent default

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)

    labels = labels.squeeze()

    label_values = np.unique(labels)
    cmap_index = label_values / (len(label_values) - 1)

    if transparent is not None:
        # set alpha of values in disabled list to 0
        cmap_array = kwargs["cmap"](cmap_index)
        cmap_array[np.isin(label_values, transparent), -1] = 0
        kwargs["cmap"] = matplotlib.colors.ListedColormap(cmap_array)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    ax.imshow(labels, **kwargs)

    if legend:

        cmap_valid_indexes = cmap_index[~np.isin(label_values, transparent)]
        valid_color_codes = kwargs["cmap"](cmap_valid_indexes)

        categories = [
            matplotlib.patches.Patch(
                [0], [0], color=color_code, alpha=kwargs["alpha"]
            )
            for color_code in valid_color_codes
        ]

        if label_names is None:
            label_names = label_values[~np.isin(label_values, transparent)]

        ax.legend(categories, label_names)

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


def plot_histogram(
    imagery: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    show_axis: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot histogram of the raster bands.

    Args:
        imagery: Imagery raster to plot. If a string or a Path is
            provided, it will be interpreted as a path to the file
            and open using rasterio.
        window: A rasterio window to
            plot only a subset of the raster. Ignored if the raster
            comes as array. Defaults to None.
        bands: Define which bands will be displayed and in which order.
            Positions in the list correspond to red, green and blue
            respectively. Defaults to [1, 2, 3].
        show_axis: If true, the axis of the image will be
            displayed.
            False by default. Defaults to True.
        ax: Axes to plot on. Otherwise,
            use current axes. Defaults to None.
        **kwargs: These will be passed to the matplotlib`s hist function.
            See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Returns:
        Axes with plot.
    """

    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"

    if "color" not in kwargs:
        kwargs["color"] = [
            COLOR_CODES["red"],
            COLOR_CODES["green"],
            COLOR_CODES["blue"],
        ]

    if "stacked" not in kwargs:
        kwargs["stacked"] = True

    if "fill" not in kwargs:
        kwargs["fill"] = True

    if isinstance(imagery, (str, Path)):
        with rasterio.open(imagery) as src:
            imagery = src.read(bands, window=window)

    # TODO: Handle nodata here, convert to nans and then ignore them
    value_range = np.nanmin(imagery), np.nanmax(imagery)

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    ax.hist(
        x=imagery.reshape(imagery.shape[0], -1).T,
        range=value_range,
        **kwargs,
    )

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


# TODO: fix x axis (it shows a continuous variable instead of a discrete one)
def plot_proportions(
    labels: RasterSourceType,
    cmap: ColorMapType = STANDARD_CMAP,
    window: Optional[WindowType] = None,
    ax: Optional[plt.Axes] = None,
    show_axis: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Plot proportion of each class present in a labels raster.

    Args:
        labels: Categorical raster to plot. If string or Path object,
            it will be interpreted as a path and open using rasterio.
        cmap: Matplotlib color map. If not provided, a default cmap
            will be used.
        window: A rasterio window to plot only a subset of the raster.
            Ignored if the raster comes as array.
        show_axis: If true, the axis of the image will be displayed.
            False by default.
        ax: Axes to plot on. Otherwise, use current axes.
        **kwargs: These will be passed to the matplotlib`s bar function.
            See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

    Returns:
        Axes with plot.
    """

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)

    class_proportions = get_raster_proportions(labels)
    label_values = list(class_proportions.keys())
    proportions = list(class_proportions.values())
    # TODO: put rescale function in raster module
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    cmap = cmap.reversed()

    show = False
    if not ax:
        show = True
        ax = plt.gca()

    ax.bar(
        x=label_values,
        height=proportions,
        color=cmap(rescale(proportions)),
        **kwargs,
    )

    if not show_axis:
        ax.axis("off")

    if show:
        plt.show()

    return ax


def describe(
    imagery: RasterSourceType,
    labels: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    figure_size: Sequence[int] = (10, 8),
) -> None:
    """Describe labelled raster plotting imagery, labels, histogram
    and proportions together in a grid.

    Args:
        imagery: Imagery raster to plot. If a string or a Path is
            provided, it will be interpreted as a path to the file
            and open using rasterio.
        labels: Categorical raster to plot. If string or Path object,
            it will be interpreted as a path and open using rasterio.
        window: A rasterio window to plot only a subset of the raster.
            Ignored for rasters that come as array. Defaults to None.
        bands: Define which bands will be displayed and in which order.
            Positions in the list correspond to red, green and blue
            respectively. Default: `red: 1, green: 2, blue: 3`.
        figure_size: Width and height of the figure in inches.
            Defaults to (10, 8).

    Returns:
        None
    """

    if isinstance(imagery, (str, Path)):
        with rasterio.open(imagery) as src:
            imagery = src.read(bands, window=window)

    if isinstance(labels, (str, Path)):
        with rasterio.open(labels) as src:
            labels = src.read(1, window=window)

    plt.figure(figsize=figure_size)

    ax_1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=3)
    plot_imagery(imagery=imagery, bands=bands, ax=ax_1)

    ax_2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=3)
    plot_labels(labels=labels, ax=ax_2, alpha=1, legend=False)

    ax_3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    plot_histogram(imagery=imagery, ax=ax_3)

    ax_4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    plot_proportions(labels=labels, ax=ax_4)

    plt.tight_layout()
    plt.show()
