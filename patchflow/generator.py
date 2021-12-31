"""PatchFlowGenerator class."""

import math

import numpy as np
from matplotlib import pyplot as plt

import rasterio
from rasterio.plot import show
from rasterio.windows import Window

import tensorflow.keras as keras
import skimage.transform
from tensorflow.keras.utils import Progbar

from raster import get_proportions, polygonize_raster


# TODO: add random seed
class PatchFlowGenerator(keras.utils.data_utils.Sequence):
    """Patch generator to feed Keras segmentation models."""

    def __init__(
        self,
        paired_paths,
        tile_shape,
        patch_shape,
        patch_indexes=None,
        batch_size=32,
        bands=[1, 2, 3],
        output_shape=None,
        rescaling_factor=None,
        shuffle=True,
    ):
        """Initialize data generator."""
        self.patch_indexes = patch_indexes
        self.paired_paths = paired_paths
        self.tile_shape = np.array(tile_shape)
        self.patch_shape = np.array(patch_shape)
        self.tile_shape_in_patches = self.tile_shape // self.patch_shape
        self.tile_size_in_patches = np.prod(self.tile_shape_in_patches)
        self.batch_size = batch_size
        self.bands = bands
        self.output_shape = output_shape
        self.rescaling_factor = rescaling_factor
        self.shuffle = shuffle
        self.iterator = 0

        if self.patch_indexes is None:
            self.patch_indexes = np.arange(
                len(self.paired_paths) * self.tile_size_in_patches
            )
            print(
                f"{len(self.patch_indexes)} patches have been set up "
                "for this generator."
            )

        if self.output_shape is None:
            self.output_shape = patch_shape

        if self.shuffle:
            self.shuffle_generator()

    def __len__(self):
        """Return number of batches per epoch."""
        return math.ceil(len(self.patch_indexes) / self.batch_size)

    def __iter__(self):
        """Make the generator iterable."""
        for index in range(len(self)):
            yield self[index]

    def __next__(self):
        """Enable getting batches using next()."""
        iterator = self.iterator
        self.iterator += 1
        return self[iterator]

    def __getitem__(self, index):
        """Get batch of patches by indexing."""

        if index >= len(self):
            raise IndexError("Batch index out of range.")

        batch_patch_indexes = self.patch_indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        return self._generate_batch(batch_patch_indexes)

    def _generate_batch(
        self, batch_patch_indexes, return_X=True):
        """Load and process patches."""

        # Initialize arrays
        Y = np.empty((self.batch_size, *self.output_shape, 1), dtype=np.uint8)
        if return_X:
            X = np.empty((self.batch_size, *self.output_shape, len(self.bands)))

        for index, patch_index in enumerate(batch_patch_indexes):

            # Locate patch in the dataset
            tile_index = patch_index // self.tile_size_in_patches
            column_index = patch_index % self.tile_shape_in_patches[0]
            row_index = (
                patch_index
                // self.tile_shape_in_patches[0]
                % self.tile_shape_in_patches[1]
            )

            # Get paired data paths
            patch_paths = self.paired_paths.iloc[tile_index]

            # Load labels
            with rasterio.open(patch_paths["labels_path"]) as src:

                patch_width = src.width // self.tile_shape_in_patches[0]
                patch_height = src.height // self.tile_shape_in_patches[1]

                window = Window(
                    column_index * patch_width,
                    row_index * patch_height,
                    patch_width,
                    patch_height,
                )
                window_shape = (window.width, window.height)

                label_array = src.read([1], window=window)

            # Load imagery
            if return_X:
                with rasterio.open(patch_paths["imagery_path"]) as src:
                    imagery_array = src.read(self.bands, window=window)

            # TODO: better padding
            # Handle incomplete tiles
            invalid_labels_shape = label_array.squeeze().shape != window_shape

            if invalid_labels_shape:
                label_array = np.resize(
                    label_array, new_shape=(*window_shape, 1)
                )

                if return_X:
                    imagery_array = np.resize(
                        imagery_array,
                        new_shape=(*window_shape, len(self.bands)),
                    )

            # Rearrange axis to match Keras standard
            parameters = {"source": 0, "destination": -1}
            label_array = np.moveaxis(label_array, **parameters)
            if return_X:
                imagery_array = np.moveaxis(imagery_array, **parameters)

            # Rescale values
            if return_X:
                if self.rescaling_factor is not None:
                    imagery_array = imagery_array * self.rescaling_factor

            # Resize output
            if self.output_shape is not None:
                if label_array.shape != self.output_shape:

                    # Resize labels
                    label_array = skimage.transform.resize(
                        image=label_array,
                        output_shape=(*self.output_shape, 1),
                        mode="constant",
                        preserve_range=True,
                    )

                    # Resize imagery
                    if return_X:
                        imagery_array = skimage.transform.resize(
                            image=imagery_array,
                            output_shape=(*self.output_shape, len(self.bands)),
                            mode="constant",
                        )

                # Collect outputs
                Y[index] = label_array
                if return_X:
                    X[index] = imagery_array

        # Return batch
        if not return_X:
            return Y

        return X, Y

    def estimate_proportions(self, number_of_classes=2, number_of_batches=10):
        """Estimate class proportions from a random sample of batches."""

        proportion_array = np.zeros(number_of_classes, dtype=float)
        progress_bar = Progbar(number_of_batches, unit_name="batch")

        for index in range(number_of_batches):

            batch_patch_indexes = np.random.choice(
                self.patch_indexes, self.batch_size
            )
            batch = self._generate_batch(batch_patch_indexes, return_X=False)

            for label_array in batch:
                array_class_proportions = get_proportions(label_array)

                for proportion in array_class_proportions:
                    proportion_array[proportion[0]] += proportion[1]

            progress_bar.update(index + 1)

        return proportion_array / np.sum(proportion_array)

    def plot_batch(
        self,
        grid_width=5,
        grid_height=5,
        figure_size=(14, 14),
        facecolor="red",
        edgecolor="none",
        alpha=0.4,
    ):
        """Plot imagery and labels of a set of patches from the next batch."""

        # Pick next batch
        imagery_batch, label_batch = next(self)

        # Plot patches
        plt.figure(figsize=figure_size)
        for index in range(grid_height * grid_width):

            # Initialize subplot
            ax = plt.subplot(grid_height, grid_width, index + 1)

            # TODO: Add option to change band positions
            # Plot imagery
            imagery = rasterio.plot.reshape_as_raster(imagery_batch[index])
            show(source=imagery, ax=ax)

            # Plot labels
            labels = rasterio.plot.reshape_as_raster(label_batch[index])
            polygonized_labels = polygonize_raster(labels)
            if any(polygonized_labels.label > 0):
                polygonized_labels[polygonized_labels.label > 0].plot(
                    ax=ax,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )

            plt.axis("off")

    def reset_generator(self):
        """Reset generator iterator."""
        self.iterator = 0

    def shuffle_generator(self):
        """Shuffle generator indexes."""
        np.random.shuffle(self.patch_indexes)

    def unshuffle_generator(self):
        """Unshuffle generator indexes."""
        np.ndarray.sort(self.patch_indexes)

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.patch_indexes = np.arange(len(self.patch_indexes))
        if self.shuffle:
            self.shuffle_generator()










