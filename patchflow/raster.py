"""Functions for working with rasters."""
from typing import Union, Type, Tuple, Any, Dict
from pathlib import Path

import numpy as np
import rasterio


# TODO: Handle nodata


RasterSourceType = Union[str, Path, np.ndarray]
WindowType = Type[rasterio.windows.Window]
ParamsType = Dict[str, Any]  # TODO: Take this away


def get_proportions(array: np.ndarray) -> Dict[int, float]:
    """Calculate pixel proportion per value in raster.

    Args:
        array: Discrete array.

    Returns:
        Dictionary containing each value present in the input array
        and its proportion.
    """

    values, counts = np.unique(array, return_counts=True)
    proportions = counts / array.size

    return dict(zip(values, proportions))


def rescale(
    array: np.ndarray, percentiles: Tuple[int, int] = (0, 100)
) -> np.ndarray:
    """Rescale array.

    Args:
        array: Array to rescale.
        percentiles: lower and upper percentiles to impose limits
            on the range of values to be taken into account for
            rescaling to (0, 100).

    Returns:
        Rescaled array.
    """
    low_percentile, top_percentile = percentiles

    minimum = np.nanpercentile(array, low_percentile)
    maximum = np.nanpercentile(array, top_percentile)

    nominator = array - minimum
    denominator = maximum - minimum

    return np.divide(
        nominator,
        denominator,
        out=np.zeros_like(nominator),
        where=denominator != 0,
    )


def pad_raster(
    raster: np.ndarray,
    out_shape: Tuple[int, int],
    **kwargs: Any,
) -> np.ndarray:
    """Pad array to match the out_shape.

    Args:
        raster: Two or three dimensional array shaped as a raster.
        out_shape: Shape (width, height) to which the input raster will
            be padded.
        kwargs: These will be passsed to the numpy.pad function.
            See full list at:
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns:
        Padded array
    """

    if not 1 < len(raster.shape) < 4:
        raise ValueError(
            f"Invalid shape: {raster.shape}. Can only work with 2 or 3"
            " dimensional arrays."
        )

    if len(raster.shape) == 2:
        raster = np.expand_dims(raster, axis=0)

    if "mode" not in kwargs:
        kwargs["mode"] = "symmetric"

    # Using np.ceil to deal with uneven numbers (it is
    # better to get a larger array and crop it afterwards).
    dim_0 = int(np.ceil((out_shape[0] - raster.shape[1]) / 2))
    dim_1 = int(np.ceil((out_shape[1] - raster.shape[2]) / 2))
    pad_width = (dim_0, dim_0), (dim_1, dim_1)

    padded_raster = np.array(
        [np.pad(band, pad_width=pad_width, **kwargs) for band in raster]
    )
    return padded_raster[:, : out_shape[0], : out_shape[1]]
