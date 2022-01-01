#%%
"""PatchFlow class."""

import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker
import rasterio
import skimage.transform
from tensorflow import keras

from stats import get_proportions
from plot import plot_imagery, plot_labels


# TODO: Add documentation
# TODO: Add filler value atribute and use it for plots
# TODO: Add infinite iterator method (in keras.utils.Sequence)
class PatchFlow(keras.utils.Sequence):
    """Patch generator to feed Keras segmentation models."""

    def __init__(
        self,
        paired_paths,
        tile_shape,
        patch_shape,
        patch_ids=None,
        batch_size=32,
        bands=[1, 2, 3],
        output_shape=None,
        rescaling_factor=None,
        shuffle=True,
        random_seed=None,
    ):
        """Initialize data generator."""

        # Paths
        self.paired_paths = paired_paths

        # Patch location
        self.patch_shape = np.array(patch_shape)
        self.tile_shape = np.array(tile_shape)
        self.grid_shape = self.tile_shape // self.patch_shape
        self.grid_size = np.prod(self.grid_shape)
        self.patch_ids = self.init_patch_ids(patch_ids)

        # Process config
        self.batch_size = batch_size
        self.bands = bands
        self.output_shape = output_shape
        if self.output_shape is None:
            self.output_shape = self.patch_shape
        self.rescaling_factor = rescaling_factor

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

    def init_rng(self, random_seed):
        """Initialize random number generator."""
        if random_seed is None:
            return None
        return np.random.default_rng(random_seed)

    def __len__(self):
        """Return number of batches per epoch."""
        return math.ceil(len(self.patch_ids) / self.batch_size)

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
        """Update generator after each epoch."""
        self.patch_ids = np.arange(len(self.patch_ids))
        if self.shuffle:
            self.shuffle_generator()

    # TODO: Add more statistics.
    # E.g.: min and max number of object pixels per class, mean, deviation...
    def estimate_proportions(self, number_of_batches=10, number_of_classes=2):
        """Estimate class proportions from a random sample of batches."""

        proportion_array = np.zeros(number_of_classes, dtype=float)
        progress_bar = keras.utils.Progbar(number_of_batches, unit_name="batch")

        for index in range(number_of_batches):

            batch_patch_ids = np.random.choice(self.patch_ids, self.batch_size)
            batch = self.load_batch(batch_patch_ids, return_X=False)

            for label_array in batch:
                array_class_proportions = get_proportions(label_array)

                for proportion in array_class_proportions:
                    proportion_array[proportion[0]] += proportion[1]

            progress_bar.update(index + 1)

        return proportion_array / np.sum(proportion_array)

    # TODO: Add some more plot arguments
    # TODO: Add kwargs
    # TODO: Add general legend for all the label colors in the plot grid
    def plot_batch(
        self,
        batch_id=None,
        grid_width=5,
        grid_height=5,
        figure_size=(14, 14),
    ):
        """Plot imagery and labels of a set of patches from the next batch."""

        if batch_id is not None:
            X_batch, Y_batch = self[batch_id]
        else:
            X_batch, Y_batch = next(self)

        # Plot
        plt.figure(figsize=figure_size)
        for index in range(grid_height * grid_width):
            ax = plt.subplot(grid_height, grid_width, index + 1)
            plot_imagery(X_batch[index], raster_shape=False, ax=ax)
            plot_labels(Y_batch[index], legend=False, ax=ax)
            ax.set_title(self.current_batch[index])

        plt.show()

    def load_batch(self, return_X=True):
        """Load and preprocess a batch of patches."""

        # TODO: The 0 class corresponds to the `everything-else`
        # class. This is because we initialize Y as an array of
        # zeros, so this is going to be the label over the empty
        # areas. An argument specifying the filler class could
        # be added, the Y array should initialize with that number.

        # Initialize output arrays
        Y = np.empty((self.batch_size, *self.output_shape, 1), dtype=np.uint8)
        if return_X:
            X = np.empty((self.batch_size, *self.output_shape, len(self.bands)))

        for index, patch_id in enumerate(self.current_batch):

            # Get patch meta
            patch_meta = self.get_patch_meta(patch_id)

            # Load data
            with rasterio.open(patch_meta["labels_path"]) as dataset:
                label_array = dataset.read([1], window=patch_meta["window"])
            if return_X:
                with rasterio.open(patch_meta["imagery_path"]) as dataset:
                    imagery_array = dataset.read(
                        self.bands, window=patch_meta["window"]
                    )

            # TODO: better padding
            # Handle incomplete tiles
            invalid_labels_shape = (
                label_array.squeeze().shape != patch_meta["window_shape"]
            )
            if invalid_labels_shape:
                label_array = np.resize(
                    label_array, new_shape=(*patch_meta["window_shape"], 1)
                )
                if return_X:
                    imagery_array = np.resize(
                        imagery_array,
                        new_shape=(
                            *patch_meta["window_shape"],
                            len(self.bands),
                        ),
                    )

            # Reshape data as image
            label_array = rasterio.plot.reshape_as_image(label_array)
            if return_X:
                imagery_array = rasterio.plot.reshape_as_image(imagery_array)

            # Rescale values
            if return_X:
                if self.rescaling_factor is not None:
                    imagery_array = imagery_array * self.rescaling_factor

            # Resize output
            # TODO: add attribute to change the `mode` parameter
            if self.output_shape is not None:
                if label_array.shape != self.output_shape:

                    label_array = skimage.transform.resize(
                        image=label_array,
                        output_shape=(*self.output_shape, 1),
                        mode="constant",
                        preserve_range=True,
                    )

                    if return_X:
                        imagery_array = skimage.transform.resize(
                            image=imagery_array,
                            output_shape=(*self.output_shape, len(self.bands)),
                            mode="constant",
                        )

                Y[index] = label_array
                if return_X:
                    X[index] = imagery_array

        if not return_X:
            return Y

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
            "window_shape": (window.width, window.height),
            "imagery_path": paths["imagery_path"],
            "labels_path": paths["labels_path"],
        }

    def plot_tile_grid(
        self,
        tile_id,
        show_labels=False,
        patch_id_color="white",
        patch_id_size="x-large",
        grid_color="white",
        linewidth=3,
        figsize=(10, 10),
    ):
        """Plot tile and its grid of patches."""
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

        # Plot imagery and labels
        paths = self.paired_paths.iloc[tile_id]
        plot_imagery(paths["imagery_path"], ax=ax, show_axis=True)
        if show_labels:
            plot_labels(paths["labels_path"], ax=ax, show_axis=True)

        # Plot ids
        for row in range(self.grid_shape[1]):
            y_coord = row * self.patch_shape[1] + self.patch_shape[1] / 2
            for column in range(self.grid_shape[0]):
                x_coord = column * self.patch_shape[0] + self.patch_shape[0] / 2
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
