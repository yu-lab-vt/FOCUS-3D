"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

from __future__ import annotations

import os

import numpy as np
from napari.types import LayerData
from skimage import io


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # the get_reader function should make as many checks as possible
    # (without loading the full file) to determine if it can read
    # the path. Here, we check the dtype of the array by loading
    # it with memmap, so that we don't actually load the full array into memory.
    # We pretend that this reader can only read integer arrays.
    try:
        arr = np.load(path, mmap_mode='r')
        if arr.dtype != np.int_:
            return None
    # napari_get_reader should never raise an exception, because napari
    # raises its own specific errors depending on what plugins are
    # available for the given path, so we catch
    # the OSError that np.load might raise if the file is malformed
    except OSError:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = 'image'  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def read_4d_folder(path: str) -> list[LayerData]:
    """Reads all .tif files in a folder as a time series (T x Z x Y x X).

    Parameters
    ----------
    path : str
        Path to the folder containing the image files.

    Returns
    -------
    List[LayerData]
        A list containing a single tuple: (data, metadata, layer_type).
        The data is a 4D numpy array (T, Z, Y, X). Metadata includes layer name.
    """
    # Get all .tif files, sorted to maintain correct time order
    files = sorted(
        [f for f in os.listdir(path) if f.endswith(('.tif', '.tiff'))]
    )
    if not files:
        return []

    # Load the first image to determine the spatial dimensions (Z, Y, X)
    first_img = io.imread(os.path.join(path, files[0]))
    # Determine the number of time points (T)
    t_frames = len(files)
    # Initialize a 4D array (T, Z, Y, X)
    stack_4d = np.zeros((t_frames,) + first_img.shape, dtype=first_img.dtype)

    # Load each file into the corresponding time frame
    for t, fname in enumerate(files):
        img = io.imread(os.path.join(path, fname))
        stack_4d[t] = img

    # Prepare metadata for the layer
    layer_data = (stack_4d, {'name': '4D Time Series'}, 'image')
    return [layer_data]
