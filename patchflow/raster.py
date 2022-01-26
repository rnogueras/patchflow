"""Functions for working with rasters."""
import numpy as np


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


# TODO: add documentation
def pad(raster, out_shape, method="symmetric"):
    """Pad array to match the out_shape. The input array
    must be shaped as a raster, namely: (bands, width, height).
    """
    if not raster.shape[0] or not raster.shape[1] or not raster.shape[2]:
        raise ValueError(
            f"Invalid shape: {raster.shape}. The input array must have"
            " three positive dimensions: (bands, width, height)."
        )

    # Using np.ceil to deal with uneven numbers (it is
    # better to get a larger array and crop it afterwards).
    pad_width = (
        int(np.ceil((out_shape[0] - raster.shape[1]) / 2)),
        int(np.ceil((out_shape[1] - raster.shape[2]) / 2)),
    )
    padded_raster = np.array(
        [np.pad(band, pad_width=pad_width, mode=method) for band in raster]
    )
    return padded_raster[:, : out_shape[0], : out_shape[1]]
