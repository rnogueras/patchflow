#%%
"""Functions for accessing the data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_tile_paths(
    directory: str, imagery_folder_name="imagery", labels_folder_name="labels"
):
    """Generate imagery / labels paired data paths to retrieve
    the data from.

    Paired files are expected to be named the same in both
    directories. Only the file names found in both folders will
    be returned.

    Parameters
    ----------
    directory : str, Path
        Path to the data main directory.
    imagery_folder : str, optional
        Name of the folder where the imagery is stored. `imagery`
        by default.
    labels_folder : str, optional
        Name of the folder where the labels are stored. `labels`
        by default.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the found paired paths.

    """

    if isinstance(directory, str):
        directory = Path(directory)

    imagery_folder_path = directory / imagery_folder_name
    labels_folder_path = directory / labels_folder_name

    valid_file_names = list(
        set(os.listdir(imagery_folder_path))
        & set(os.listdir(labels_folder_path))
    )

    valid_file_names.sort()

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


def generate_patch_ids(
    paired_paths,
    tile_shape,
    patch_shape,
    validation_size=0.2,
    test_size=0.1,
    shuffle=True,
    verbose=True,
    random_seed=None,
):
    """Calculate and generate a set of indexes based on the tile
    and patch sizes.

    This can be useful when the number of tiles is small or when
    you want to generate validation and test sets that have a
    distribution strictly similar to the training set. The function
    generates a single set of patches from the tiles provided and
    then splits them into the three subsets following the specified
    criteria. This ensures that the patches in each set come from
    any of the original tiles without the distribution of the tiles
    skewing the result. In no case are the patches generated, only
    the indexes needed to calculate the windows are created in this
    step, so both the amount of processing and the size of the output
    are very light.

    Parameters
    ----------
    paired_paths : pd.DataFrame
        The output of the generate_tile_paths function.
    tile_shape : tuple of integers
        Shape of the tile in pixels, provided as a tuple containing
        a pair of integers. E.g.: (1000, 1000).
    patch_shape :
        Shape of the tile in pixels, provided as a tuple containing
        a pair of integers. E.g.: (128, 128).
    validation_size : float, optional
        A real number between 0 and 1 specifying the proportion of
        patches to place in the validation test. 0.2 by default.
    test_size : float, optional
        A real number between 0 and 1 specifying the proportion of
        patches to place in the validation test. 0.1 by default.
    shuffle : bool, optional
        Whether to shuffle patches before splitting them into subsets.
    verbose : bool, optional
        Whether to print the size in patches of each subset. True
        by default.

    Returns
    -------
    tuple of dictionaries
        Three dictionaries each of which contains the paired_paths,
        tile_shape, patche_shape and patch_indexes of a different
        subset, so that it can be easily passed to a different
        instance of the PatchFlowGenerator using the ** operator.
    """

    grid_size = np.prod(np.array(tile_shape) // np.array(patch_shape))
    patch_ids = np.arange(len(paired_paths) * grid_size)

    if shuffle:
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
            rng.shuffle(patch_ids)
        else:
            np.random.shuffle(patch_ids)

    test_length = int(len(patch_ids) * test_size)
    validation_length = int(len(patch_ids) * validation_size)

    # Extract test and validation indexes
    # and leave the rest for training
    test_ids = patch_ids[0:test_length]
    validation_ids = patch_ids[test_length : test_length + validation_length]
    training_ids = patch_ids[test_length + validation_length :]

    if verbose:
        print(f"{len(training_ids)} patches have been set up for training.")
        print(f"{len(validation_ids)} patches have been set up for validating.")
        print(f"{len(test_ids)} patches have been set up for testing.")

    return [
        {
            "paired_paths": paired_paths,
            "tile_shape": tile_shape,
            "patch_shape": patch_shape,
            "patch_ids": id_subset,
        }
        for id_subset in [
            training_ids,
            validation_ids,
            test_ids,
        ]
    ]
