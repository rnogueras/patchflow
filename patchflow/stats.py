"""Functions for working with rasters."""
import numpy as np


def get_proportions(raster):
    """Calculate pixel proportion per value in raster.

    Parameters
    ----------
    raster : np.array
        An array containing the rasterized labels.

    Returns
    -------
    list of tuples
        value, proportion
    """
    values, counts = np.unique(raster, return_counts=True)
    proportions = counts / raster.size

    # TODO: convert this into a dictionary, update generator as well
    return [pair for pair in zip(values, proportions)]
