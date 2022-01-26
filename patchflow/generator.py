"""PatchFlow class."""
from typing import Optional, Sequence
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import rasterio
import skimage.transform
from tensorflow import keras

from raster import get_raster_proportions, pad
from plot import plot_imagery, plot_labels


# TODO: Add documentation
class PatchFlowGenerator(keras.utils.Sequence):
    """Patch generator to feed Keras segmentation models."""

    def __init__(
        self,
        paired_paths: pd.DataFrame,
        tile_shape: Sequence[int],
        patch_shape: Sequence[int],
        patch_ids: Optional[Sequence[int]] = None,
        batch_size: int = 32,
        bands: Sequence[int] = [1, 2, 3],
        filler_label: int = 0,
        padding_method: str = "symmetric",
        output_shape: Optional[Sequence[int]] = None,
        resizing_method: str = "constant",
        rescaling_factor: Optional[float] = None,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ):
        """Initialize PatchFlowGenerator. The class can be instanciated
        in two different ways:

            - Providing a paired paths dataframe outputted by the
                `generate_paired_paths` function along with arbitrary
                tile_shape and patch_shape arguments. This way, the
                patch ids are automatically generated from the specified
                tile shape and the desired patch shapes during the
                initialization.

            - Passing one of the dictionaries outputted by the
                `generate_patch_ids` function using the ** operator.

        The patch grids generated over the tiles can be checked using the
        `plot_grid` method.

        Args:
            paired_paths: A dataframe containing the imagery and labels
                paired paths of each tile as outputted by the
                `generate_paired_paths` function.
            tile_shape: Tile shape in pixels. E.g.: (5000, 5000). The
                generator assumes that all tiles have the same shape.
                The patch grid will be built from the top left corner.
                The area outside the provided tile shape will not be
                processed by the generator.
            patch_shape: Patch shape in pixels. E.g.: (128, 128).
            patch_ids: A sequence containing the patch ids that will
                be used to calculate the patch windows over the imagery.
                Defaults to None.
            batch_size: Number of patches generated per batch. Defaults
                to 32.
            bands: Imagery bands to read during the data generation.
                Defaults to [1, 2, 3].
            filler_label: Pixel label for the filler class. This value
                will be used to label any empty area found during the
                data generation. E.g.: incomplete tiles. Defaults to 0.
            padding_method: Numpy padding mode. Defaults to "symmetric".
                See full list at:
                https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            output_shape: Shape of the output images. If the output
                shape is different from the patch shape, the images
                will be resized during the data generation. Defaults
                to None.
            resizing_method: Name of the mode to be used for resizing
                the images when the patch size and output size are
                different. Defaults to "constant". See full list at:
                https://scikit-image.org/docs/dev/api/skimage.transform.html
            rescaling_factor: Real number used to rescale the pixel
                values. E.g.: 1 / 255. Defaults to None.
            shuffle: Whether to shuffle the patch ids at the end of each
                epoch. Defaults to True.
            random_seed: Pass an integer for reproducible output. Defaults
                to None.
        """

        # Paths
        self.paired_paths = paired_paths

        # Patch location
        self.patch_shape = patch_shape
        self.tile_shape = tile_shape
        self.grid_shape = self.init_grid_shape()
        self.grid_size = np.prod(self.grid_shape)
        self.patch_ids = self.init_patch_ids(patch_ids)

        # Process config
        self.batch_size = batch_size
        self.bands = bands
        self.rescaling_factor = rescaling_factor
        self.filler_label = filler_label
        self.padding_method = padding_method
        self.output_shape = output_shape
        if self.output_shape is None:
            self.output_shape = self.patch_shape
        self.resizing_method = resizing_method

        # Iteration
        self.iterator = 0
        self.shuffle = shuffle
        self.rng = self.init_rng(random_seed)
        if self.shuffle:
            self.shuffle_generator()

    def init_patch_ids(self, patch_ids):
        """Initialize patch ids."""
        if patch_ids is None:
            patch_ids = np.arange(len(self.paired_paths) * self.grid_size)
            print(
                f"{len(patch_ids)} patches have been set up in this generator."
            )
        return patch_ids

    # TODO: Add greedy mode
    def init_grid_shape(self):
        """Initialize grid shape."""
        grid_shape = np.array(self.tile_shape) // np.array(self.patch_shape)
        return grid_shape.tolist()

    def init_rng(self, random_seed):
        """Initialize random number generator."""
        if random_seed is None:
            return None
        return np.random.default_rng(random_seed)

    def __len__(self):
        """Return number of batches in the sequence."""
        return math.ceil(len(self.patch_ids) / self.batch_size)

    def __iter__(self):
        """Create a generator that iterates over the Sequence."""
        for index in range(len(self)):
            yield self[index]

    def __next__(self):
        """Enable getting batches using next()."""
        iterator = self.iterator
        self.iterator += 1
        return self[iterator]

    def __getitem__(self, index):
        """Get batch of patches by indexing the Sequence."""

        if index >= len(self):
            raise IndexError("Batch index out of range.")

        self.current_batch = self.patch_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        return self.load_batch()

    def reset_generator(self):
        """Reset generator iterator."""
        self.iterator = 0

    def shuffle_generator(self):
        """Shuffle patch ids."""
        if self.rng is not None:
            self.rng.shuffle(self.patch_ids)
            return
        np.random.shuffle(self.patch_ids)

    def unshuffle_generator(self):
        """Unshuffle generator patch ids."""
        np.ndarray.sort(self.patch_ids)

    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        self.patch_ids = np.arange(len(self.patch_ids))
        if self.shuffle:
            self.shuffle_generator()

    def estimate_proportions(self, number_of_batches=10, number_of_classes=2):
        """Estimate class proportions from a random sample of batches."""

        proportion_array = np.zeros(number_of_classes, dtype=float)
        progress_bar = keras.utils.Progbar(
            number_of_batches, unit_name="batch"
        )

        for index in range(number_of_batches):

            batch = next(self)

            for label_array in batch[1]:
                raster_proportions = get_raster_proportions(label_array)

                for label, proportion in raster_proportions.items():
                    proportion_array[label] += proportion

            progress_bar.update(index + 1)

        return proportion_array / np.sum(proportion_array)

    def plot_batch(
        self,
        batch_id=None,
        grid_width=5,
        grid_height=5,
        figure_size=(14, 14),
        imagery_kwargs={},
        labels_kwargs={},
    ):
        """Plot imagery and labels of a set of patches from the next batch."""

        if batch_id is not None:
            X_batch, Y_batch = self[batch_id]
        else:
            X_batch, Y_batch = next(self)

        if "legend" not in labels_kwargs:
            labels_kwargs["legend"] = False

        plt.figure(figsize=figure_size)

        for index in range(grid_height * grid_width):
            ax = plt.subplot(grid_height, grid_width, index + 1)
            ax.set_title(self.current_batch[index])
            plot_imagery(
                X_batch[index], raster_shape=False, ax=ax, **imagery_kwargs
            )
            plot_labels(Y_batch[index], ax=ax, **labels_kwargs)

        plt.show()

    def load_batch(self):
        """Load and preprocess a batch of patches."""

        Y = np.empty((self.batch_size, *self.output_shape, 1), dtype=np.uint8)
        X = np.empty((self.batch_size, *self.output_shape, len(self.bands)))

        for index, patch_id in enumerate(self.current_batch):

            patch_meta = self.get_patch_meta(patch_id)

            with rasterio.open(patch_meta["labels_path"]) as src:
                labels = src.read([1], window=patch_meta["window"])
            with rasterio.open(patch_meta["imagery_path"]) as src:
                imagery = src.read(self.bands, window=patch_meta["window"])

            labels_shape = labels.squeeze().shape
            imagery_shape = imagery[0, :, :].shape

            if labels_shape == imagery_shape != self.patch_shape:
                if labels_shape[0] and labels_shape[1]:
                    labels = pad(
                        raster=labels,
                        out_shape=(*self.patch_shape,),
                        method=self.padding_method,
                    )
                    imagery = pad(
                        raster=imagery,
                        out_shape=(*self.patch_shape,),
                        method=self.padding_method,
                    )
                else:
                    labels = np.full(
                        shape=(1, *self.patch_shape),
                        fill_value=self.filler_label,
                        dtype=np.uint8,
                    )
                    imagery = np.zeros(
                        shape=(len(self.bands), *self.patch_shape),
                        dtype=np.uint8,
                    )

            elif labels_shape != imagery_shape != self.patch_shape:
                labels = np.full(
                    shape=(1, *self.patch_shape),
                    fill_value=self.filler_label,
                    dtype=np.uint8,
                )
                imagery = np.zeros(
                    shape=(len(self.bands), *self.patch_shape), dtype=np.uint8
                )

            if self.rescaling_factor is not None:
                imagery = imagery * self.rescaling_factor

            labels = rasterio.plot.reshape_as_image(labels)
            imagery = rasterio.plot.reshape_as_image(imagery)

            if labels.squeeze().shape != self.output_shape != None:
                labels = skimage.transform.resize(
                    image=labels,
                    output_shape=(*self.output_shape, 1),
                    mode=self.resizing_method,
                    preserve_range=True,
                )
                imagery = skimage.transform.resize(
                    image=imagery,
                    output_shape=(*self.output_shape, len(self.bands)),
                    mode=self.resizing_method,
                )

            Y[index] = labels
            X[index] = imagery

        return X, Y

    def get_patch_meta(self, patch_id):
        """Locate patch in the dataset."""

        tile_id = patch_id // self.grid_size
        column_id = patch_id % self.grid_shape[0]
        row_id = patch_id // self.grid_shape[0] % self.grid_shape[1]

        window = rasterio.windows.Window(
            column_id * self.patch_shape[0],
            row_id * self.patch_shape[1],
            self.patch_shape[0],
            self.patch_shape[1],
        )

        paths = self.paired_paths.iloc[tile_id]

        return {
            "patch_id": patch_id,
            "tile": tile_id,
            "column": column_id,
            "row": row_id,
            "window": window,
            "imagery_path": paths["imagery_path"],
            "labels_path": paths["labels_path"],
        }

    def plot_grid(
        self,
        patch_id=None,
        tile_id=None,
        show_labels=True,
        patch_id_color="white",
        patch_id_size="x-large",
        grid_color="white",
        linewidth=3,
        figsize=(10, 10),
        imagery_kwargs={},
        labels_kwargs={},
    ):
        """Plot tile and its grid of patches."""

        if patch_id is not None:
            tile_id = self.get_patch_meta(patch_id)["tile"]

        if patch_id is None and tile_id is None:
            raise ValueError("Either patch_id or tile_id must be provided.")

        sorted_patch_ids = np.arange(len(self.patch_ids))
        tile_patch_ids = sorted_patch_ids[
            tile_id * self.grid_size : (tile_id + 1) * self.grid_size
        ]

        fig, ax = plt.subplots(figsize=figsize)
        # Remove whitespace around the image
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Define grid
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(base=self.patch_shape[0])
        )
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(base=self.patch_shape[1])
        )
        ax.grid(
            which="major",
            axis="both",
            linestyle="-",
            color=grid_color,
            linewidth=linewidth,
        )

        if "show_axis" not in imagery_kwargs:
            imagery_kwargs["show_axis"] = True

        if "show_axis" not in labels_kwargs:
            labels_kwargs["show_axis"] = True

        # Plot imagery and labels
        paths = self.paired_paths.iloc[tile_id]
        plot_imagery(paths["imagery_path"], ax=ax, **imagery_kwargs)
        if show_labels:
            plot_labels(paths["labels_path"], ax=ax, **labels_kwargs)

        # Plot ids
        for row in range(self.grid_shape[1]):
            y_coord = row * self.patch_shape[1] + self.patch_shape[1] / 2
            for column in range(self.grid_shape[0]):
                x_coord = (
                    column * self.patch_shape[0] + self.patch_shape[0] / 2
                )
                patch_position = column + row * self.grid_shape[0]
                ax.text(
                    x_coord,
                    y_coord,
                    f"{tile_patch_ids[patch_position]}",
                    color=patch_id_color,
                    size=patch_id_size,
                    ha="center",
                    va="center",
                )
