"""Functions for plotting imagery and labels."""
from typing import Optional, Sequence, Any, Type
import warnings

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import rasterio
import rasterio.plot

from patchflow.data import read_source
from patchflow.raster import (
    RasterSourceType,
    WindowType,
    ParamsType,
    get_proportions,
    rescale,
)

# TODO: Handle nodata

# TODO: Catch the warning isolatedly
warnings.filterwarnings("ignore")

ColorMapType = Type[matplotlib.colors.ListedColormap]

COLOR_CODES = dict(
    grey="#a3a3a3",
    blue="#277da1",
    green="#90be6d",
    turoquoise="#43aa8b",
    yellow="#f9c74f",
    red="#f94144",
)

STANDARD_CMAP = matplotlib.colors.ListedColormap(list(COLOR_CODES.values()))


def show_imagery(
    source: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    raster_shape: bool = True,
    show_axis: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot pixel value array.

    Args:
        source: Path or array to read the data from.
        window: A rasterio Window to plot only a subset of the raster.
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

    raster = read_source(source=source, window=window, bands=bands)

    if ax is None:
        ax = plt.gca()

    if raster_shape:
        raster = rasterio.plot.reshape_as_image(raster)

    ax.imshow(raster, **kwargs)

    if not show_axis:
        ax.axis("off")

    return ax


def show_labels(
    source: RasterSourceType,
    window: Optional[WindowType] = None,
    transparent: Sequence[int] = (0,),
    legend: bool = True,
    label_names: Optional[Sequence[str]] = None,
    show_axis: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot pixel label array.

    Args:
        source: Path or array to read the data from.
        window: A rasterio Window to plot only a subset of the raster.
        transparent: List of label values to make transparent. The alpha
            of this values will be set to 0. By default, hide label 0.
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

    defaults = dict(interpolation="nearest", alpha=0.7, cmap=STANDARD_CMAP)
    kwargs = {**defaults, **kwargs}

    raster = read_source(source=source, window=window).squeeze()

    label_values = np.unique(raster)
    cmap_index = label_values / (len(label_values) - 1)

    if transparent:
        # set alpha of values in transparent list to 0
        cmap_array = kwargs["cmap"](cmap_index)
        cmap_array[np.isin(label_values, transparent), -1] = 0
        kwargs["cmap"] = matplotlib.colors.ListedColormap(cmap_array)

    if ax is None:
        ax = plt.gca()

    ax.imshow(raster, **kwargs)

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

    return ax


def show_histogram(
    source: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    show_axis: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot histogram of the raster bands.

    Args:
        source: Path or array to read the data from.
        window: A rasterio Window to plot only a subset of the raster.
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

    defaults = dict(
        histtype="step",
        color=(COLOR_CODES["red"], COLOR_CODES["green"], COLOR_CODES["blue"]),
        stacked=True,
        fill=True,
    )

    raster = read_source(source=source, window=window, bands=bands)
    value_range = np.nanmin(raster), np.nanmax(raster)

    if ax is None:
        ax = plt.gca()

    ax.hist(
        x=raster.reshape(raster.shape[0], -1).T,
        range=value_range,
        **{**defaults, **kwargs},
    )

    if not show_axis:
        ax.axis("off")

    return ax


def show_proportions(
    source: RasterSourceType,
    window: Optional[WindowType] = None,
    ax: Optional[plt.Axes] = None,
    show_axis: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Plot proportion of each class present in a labels raster.

    Args:
        source: Path or array to read the data from.
        window: A rasterio Window to plot only a subset of the raster.
        show_axis: If true, the axis of the image will be displayed.
            False by default.
        ax: Axes to plot on. Otherwise, use current axes.
        **kwargs: These will be passed to the matplotlib`s bar function.
            See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

    Returns:
        Axes with plot.
    """

    raster = read_source(source=source, window=window).squeeze()

    class_proportions = get_proportions(raster)
    label_values = [str(key) for key in class_proportions.keys()]
    proportions = list(class_proportions.values())

    defaults = dict(color=STANDARD_CMAP, x=label_values)
    kwargs = {**defaults, **kwargs}
    kwargs["color"] = kwargs["color"].reversed()
    kwargs["color"] = kwargs["color"](rescale(proportions))

    if ax is None:
        ax = plt.gca()

    ax.bar(height=proportions, **kwargs)

    if not show_axis:
        ax.axis("off")

    return ax


def describe(
    imagery_source: RasterSourceType,
    label_source: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, 2, 3),
    figure_size: Sequence[int] = (10, 8),
) -> None:
    """Describe labelled raster plotting imagery, labels, histogram
    and proportions together in a grid.

    Args:
        imagery_source: Path or array to read the data from.
        label_source: Path or array to read the data from.
        window: A rasterio Window to plot only a subset of the raster.
        bands: Define which bands will be displayed and in which order.
            Positions in the list correspond to red, green and blue
            respectively. Default: `red: 1, green: 2, blue: 3`.
        figure_size: Width and height of the figure in inches.
            Defaults to (10, 8).

    Returns:
        None
    """
    
    imagery_raster = read_source(
        source=imagery_source, window=window, bands=bands
    )
    label_raster = read_source(source=label_source, window=window)

    plt.figure(figsize=figure_size)

    ax_1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=3)
    show_imagery(source=imagery_raster, bands=bands, ax=ax_1)

    ax_2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=3)
    show_labels(
        source=label_raster, ax=ax_2, alpha=1, legend=False, transparent=[]
    )

    ax_3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    show_histogram(source=imagery_raster, ax=ax_3)

    ax_4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    show_proportions(source=label_raster, ax=ax_4)

    plt.tight_layout()
    plt.show()


def add_grid(
    patch_shape: Sequence[int],
    grid_shape: Optional[Sequence[int]] = None,
    patch_ids: Optional[Sequence[int]] = None,
    grid_params: Optional[ParamsType] = None,
    text_params: Optional[ParamsType] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Add grid and (if provided) patch ids to plot.

    Args:
        patch_shape: The patch shape in pixels. E.g.: (128, 128)
        grid_shape: The grid shape in patches. E.g.: (10, 10). If not
            provided, the grid shape will be calculated from the ax.
            Defaults to None.
        patch_ids: Sequence of numbers corresponding to the patch
            ids present in the tile. If not provided, no ids will
            be showed. Defaults to None.
        grid_params: Dictionary (argument: value) to be passed to the
            matplotlib`s grid function. Defaults to None. See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
        text_params: Dictionary (argument: value) to be passed to the
            matplotlib`s grid function. Defaults to None. See full list at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
        ax: Axes to plot on. Otherwise, use current axes.

    Returns:
        Axes with plot.
    """

    grid_defaults = dict(color="white", linewidth=2)
    text_defaults = dict(
        color="white", size="x-large", ha="center", va="center"
    )

    if ax is None:
        ax = plt.gca()

    patch_width, patch_height = patch_shape

    if grid_shape is not None:
        grid_width, grid_height = grid_shape
    else:
        grid_width = int(np.abs(np.subtract(*(ax.get_xlim())) / patch_width))
        grid_height = int(np.abs(np.subtract(*(ax.get_ylim()))) / patch_height)

    # Define grid
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(patch_width))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(patch_height))
    ax.grid(**{**grid_defaults, **(grid_params or {})})
    ax.axis("on") # Axis must be enabled to view the grid

    # Plot ids
    if patch_ids is not None:
        for row in range(grid_height):
            coordinate_y = row * patch_height + patch_height / 2
            for column in range(grid_width):
                coordinate_x = column * patch_width + patch_width / 2
                patch_position = column + row * grid_width
                ax.text(
                    coordinate_x,
                    coordinate_y,
                    f"{patch_ids[patch_position]}",
                    **{**text_defaults, **(text_params or {})},
                )

    return ax
