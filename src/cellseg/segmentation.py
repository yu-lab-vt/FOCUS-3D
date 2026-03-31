import traceback

import dask.array as da
import dask_image.ndmeasure
import numpy as np
from qtpy.QtCore import QObject, QThread, Signal
from scipy import ndimage as ndi


class SegmentationWorker(QObject):
    """Worker for running 3D segmentation in a background thread."""

    progress = Signal(int, str)  # progress value, message
    finished = Signal(
        object, str
    )  # result (labels array), error message (empty if success)
    cancelled = Signal()

    def __init__(self, image_data, mode, threshold=None, checkpoint=None):
        super().__init__()
        self.image_data = image_data
        self.mode = mode  # 0: manual, 1: auto
        self.threshold = threshold
        self.checkpoint = checkpoint
        self._is_cancelled = False
        # self.chk_contour.blockSignals(True)
        # self.chk_contour.setChecked(False)
        # self.chk_contour.blockSignals(False)
        # self._update_curation_controls()

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.mode == 0:  # manual threshold
                self.progress.emit(0, 'Applying threshold...')
                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                # Use the existing _threshold_segmentation logic
                labels = self._threshold_segmentation(
                    self.image_data, self.threshold
                )
                self.progress.emit(100, 'Segmentation completed.')
                self.finished.emit(labels, '')
            elif self.mode == 1:  # auto (placeholder)
                self.progress.emit(
                    0, 'Running auto-segmentation (not yet implemented)...'
                )
                # Simulate long work
                QThread.sleep(2)
                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                self.finished.emit(None, 'Auto-segmentation not implemented.')
            else:
                self.finished.emit(None, 'Invalid segmentation mode.')
        except (ValueError, OSError) as e:
            self.finished.emit(
                None, f'Segmentation error: {e}\n{traceback.format_exc()}'
            )

    def _threshold_segmentation(self, image, thresh):
        """
        Perform threshold-based 3D segmentation using manual threshold.
        Supports numpy, zarr, or dask arrays. For large zarr arrays, uses dask
        for memory-efficient processing.
        """

        # Determine input type and convert to dask if not numpy
        if isinstance(image, np.ndarray):
            # For numpy, process directly with scipy (fast for in-memory data)
            binary = image > thresh
            labeled, _ = ndi.label(binary)
            return labeled.astype(np.int32)
        else:
            # For zarr or other array-like, convert to dask array for chunked processing
            if not isinstance(image, da.Array):
                # Try to convert to dask, preserving chunks if possible
                if hasattr(image, 'chunks'):
                    image = da.from_array(image, chunks=image.chunks)
                else:
                    # Use auto-chunking as fallback
                    image = da.from_array(image, chunks='auto')

            # Binary threshold (dask supports elementwise comparison)
            binary = image > thresh

            # Connected components labeling using dask-image
            labeled, num_features = dask_image.ndmeasure.label(binary)

            # Convert to int32
            labeled = labeled.astype(np.int32)

            return labeled
