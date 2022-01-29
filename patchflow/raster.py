#%%
"""Functions for working with rasters."""
from typing import Union, Type, Tuple, Any
from pathlib import Path

import rasterio
import numpy as np


RasterSourceType = Union[str, Path, np.ndarray]
WindowType = Type[rasterio.windows.Window]


def get_raster_proportions(raster):
    """Calculate pixel proportion per value in raster.

    Parameters
    ----------
    raster : np.array
        An array containing the rasterized labels.

    Returns
    -------
    Dictionary
        value: proportion
    """
    values, counts = np.unique(raster, return_counts=True)
    proportions = counts / raster.size

    return {value: proportion for value, proportion in zip(values, proportions)}


def pad_raster(
    raster: np.ndarray,
    out_shape: Tuple[int, int],
    **kwargs: Any,
) -> np.ndarray:
    """Pad array to match the out_shape.

    Args:
        raster: Two or three dimensional array shaped as raster.
        out_shape: Shape (width, height) to which the input raster will
            be padded. 
        kwargs: These will be passsed to the numpy.pad function.
            See full list at:
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Returns:
        padded array

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
