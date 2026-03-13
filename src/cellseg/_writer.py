"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from skimage import io

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    """Writes a single image layer.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """

    # implement your writer logic here ...

    # return path to any file(s) that were successfully written
    return [path]


def write_multiple(path: str, data: list[FullLayerData]) -> list[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """

    # implement your writer logic here ...

    # return path to any file(s) that were successfully written
    return [path]


def napari_write_labels(
    path: str, data: Union[np.ndarray, list[np.ndarray]], meta: dict
):
    """Write a labels layer to a file.

    Parameters
    ----------
    path : str
        Path to the file to write.
    data : array or list of arrays
        The image data. For a single layer, this is a numpy array.
    meta : dict
        Metadata associated with the layer.

    Returns
    -------
    str or None
        The path to the file if successful, otherwise None.
    """
    # Handle both single layer and multiple layers (list)
    if isinstance(data, list):
        # For simplicity, we only handle single layer saving via this writer
        return None

    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        return None

    # Save the labels as uint16 TIFF
    try:
        io.imsave(path, data.astype(np.uint16))
        return path
    except (OSError, ValueError) as e:
        print(f'Error saving labels: {e}')
        return None


# Optional: Add a function to get the writer for specific layer types
def napari_get_writer(layer_types):
    """Return the writer function if the layer type is 'labels'."""
    if 'labels' in layer_types:
        return napari_write_labels
    return None
