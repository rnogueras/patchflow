"""Functions for accessing the data."""

import os
from pathlib import Path

import pandas as pd
import rasterio


def generate_paired_paths(
    directory: str, imagery_folder_name="imagery", labels_folder_name="labels"
):
    """Retrieve imagery / labels paired data paths.

    Paired files are expected to be named the same in both
    directories. Only the file names found in both folders
    will be returned. No order is preserved.

    Parameters
    ----------
    directory : str, Path
        Path to the data main directory.
    imagery_folder : str, optional
        Name of the folder where the imagery is stored.
        `imagery` by default.
    labels_folder : str, optional
        Name of the folder where the labels are stored.
        `labels` by default.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the found paired paths.

    """

    if isinstance(directory, str):
        directory = Path(directory)

    imagery_folder_path = directory / imagery_folder_name
    labels_folder_path = directory / labels_folder_name

    valid_file_names = set(os.listdir(imagery_folder_path)) & set(
        os.listdir(labels_folder_path)
    )

    return pd.DataFrame(
        {
            "imagery_path": (
                imagery_folder_path / name for name in valid_file_names
            ),
            "labels_path": (
                labels_folder_path / name for name in valid_file_names
            ),
        }
    )
