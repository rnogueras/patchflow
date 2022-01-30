"""Functions for accessing the data."""
from typing import Dict, List, Union, Any, Sequence, Optional
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from patchflow.raster import RasterSourceType, WindowType


def pair_paths(
    directory: Union[str, Path],
    imagery_folder_name: str = "imagery",
    labels_folder_name: str = "labels",
) -> pd.DataFrame:
    """Generate imagery & labels paired data paths to retrieve
    the data from.

    Paired files are expected to be named the same in both
    directories. Only the file names found in both folders will
    be returned.

    Args:
        directory: Path to the data main directory.
        imagery_folder_name: Name of the folder where the imagery is
            stored. Defaults to `imagery`.
        labels_folder_name: Name of the folder where the labels are
            stored. Defaults to `labels`.

    Returns:
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


def tag_patches(
    paired_paths: pd.DataFrame,
    tile_shape: Sequence[int],
    patch_shape: Sequence[int],
    validation_size: float = 0.2,
    test_size: float = 0.1,
    shuffle: bool = True,
    verbose: bool = True,
    random_seed: bool = None,
) -> List[Dict[str, Any]]:
    """Generate training, validation and test subsets of patch
    indexes from the tile and patch sizes provided.

    This function generates a single set of patches from the paired
    paths provided and then splits them into three subsets following 
    the specified criteria. This ensures that the patches in each set
    may come from any tile, so their distribution do not skew the 
    results. In no case are the patches generated, only the indexes
    needed to calculate the windows are created in this step, so both
    the amount of processing and the size of the output are very light.

    Args:
        paired_paths: The output of the generate_tile_paths function.
        tile_shape: Shape of the tile in pixels, provided as a tuple
            containing a pair of integers. E.g.: (1000, 1000).
        patch_shape: Shape of the tile in pixels, provided as a tuple
            containing a pair of integers. E.g.: (128, 128).
        validation_size: A real number between 0 and 1 specifying the
            proportion of patches to place in the validation test.
            0.2 by default.
        test_size: A real number between 0 and 1 specifying the
            proportion of patches to place in the validation test.
            0.1 by default.
        shuffle: Whether to shuffle patches before splitting them into
            subsets.
        verbose: Whether to print the size in patches of each subset.
            True by default.
        random_seed: Random seed to shuffle the generated ids.

    Returns:
        Three dictionaries each of which contains the paired_paths,
        tile_shape, patche_shape and patch_indexes of a different
        subset, so that they can be easily passed to an instance of
        the PatchFlowGenerator using the ** operator.
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
        print(
            f"{len(validation_ids)} patches have been set up for validating."
        )
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


def read_source(
    source: RasterSourceType,
    window: Optional[WindowType] = None,
    bands: Sequence[int] = (1, )
) -> np.ndarray:
    """Read raster data source.

    Args:
        source: Path or array to read the data from.
        window: Window object to read a subset of the raster only.
        bands: Defines which bands will be displayed and in which order.
            Positions in the list correspond to red, green and blue
            respectively. Default: read first band only.

    Returns:
        Raster.
    """
    
    if isinstance(source, (str, Path)):
        with rasterio.open(source) as src:
            return src.read(bands, window=window)

    elif isinstance(source, np.ndarray):
        if window is not None:
            
            if len(source.shape) == 2:
                source = np.expand_dims(source, axis=0)

            source = source[
                :,
                window.col_off : window.col_off + window.width,
                window.row_off : window.row_off + window.height
            ]

        return source
    
    raise TypeError(
        f"The source type {type(source)} is incorrect."
        " Only string, Path or np.ndarray types are allowed."
    )