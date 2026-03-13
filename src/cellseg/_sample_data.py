"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy as np
from skimage import data


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    return [(np.random.rand(512, 512), {})]


def _generate_3d_cells():
    """Generate a synthetic 3D image of cells using blobs."""
    # Create a 3D volume with some blobs to simulate cells
    blobs = data.binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype(
        float
    )
    # Add some intensity variation
    image = blobs * np.random.uniform(0.5, 1.0, blobs.shape)
    # Add a bit of noise
    image += np.random.normal(0, 0.1, image.shape)
    image = np.clip(image, 0, 1)
    return image


def sample_data_3d():
    """Return a sample 3D image."""
    return [(_generate_3d_cells(), {'name': 'Sample 3D Cells'}, 'image')]


def sample_data_4d():
    """Return a sample 4D time series (T, Z, Y, X) with moving cells."""
    t_frames = 5
    base_volume = _generate_3d_cells()
    # Create a time series by slightly shifting the cells
    time_series = []
    for t in range(t_frames):
        shift = t * 2  # shift by 2 pixels each frame
        # Simple translation for demonstration - in reality you might use scipy.ndimage.shift
        # Here we'll just roll the array for simplicity
        shifted = np.roll(base_volume, shift=shift, axis=(1, 2))
        time_series.append(shifted)
    stack_4d = np.stack(time_series, axis=0)
    return [(stack_4d, {'name': 'Sample 4D Cells'}, 'image')]
