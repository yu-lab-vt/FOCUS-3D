import traceback

import dask.array as da
import numpy as np
from dask import compute, delayed
from qtpy.QtCore import QObject, Signal
from skimage.measure import regionprops


class QuantitativeWorker(QObject):
    """Worker for computing quantitative statistics in the background."""

    progress = Signal(int, str)
    finished = Signal(dict, str)  # stats dict, error message

    def __init__(self, labels_data, image_data):
        super().__init__()
        self.labels_data = labels_data
        self.image_data = image_data
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(0, 'Computing label areas...')
            if self._is_cancelled:
                self.finished.emit({}, 'Cancelled')
                return

            # Use the existing compute_areas_dask or numpy fallback
            if isinstance(self.labels_data, da.Array):
                label_areas = self.compute_areas_dask(self.labels_data)
                labels_list, areas_list = (
                    zip(*label_areas, strict=True) if label_areas else ([], [])
                )
            else:
                labels = np.asarray(self.labels_data)
                props = regionprops(
                    labels,
                    intensity_image=np.asarray(self.image_data)
                    if self.image_data is not None
                    else None,
                )
                labels_list = [p.label for p in props]
                areas_list = [p.area for p in props]

            if self._is_cancelled:
                self.finished.emit({}, 'Cancelled')
                return

            self.progress.emit(50, 'Computing intensity statistics...')
            # ... compute other stats (min/max, histogram) ...
            # For simplicity, we return the raw lists; the main thread will build the dialog.
            stats = {
                'labels': labels_list,
                'areas': areas_list,
                'num_cells': len(labels_list),
                'max_area': max(areas_list) if areas_list else 0,
                'min_area': min(areas_list) if areas_list else 0,
                # intensity stats can be added here
            }
            self.progress.emit(100, 'Done.')
            self.finished.emit(stats, '')
        except (ValueError, OSError) as e:
            self.finished.emit(
                {}, f'Statistics error: {e}\n{traceback.format_exc()}'
            )

    def compute_areas_dask(self, labels):
        """Return a Dask bag of (label, area) for each label > 0."""
        # Get unique labels (excluding 0) - this is a Dask array
        unique_labels = da.unique(labels)
        # Filter out background (label 0)
        unique_labels = unique_labels[unique_labels != 0]

        # Define a function to count pixels of a specific label in a chunk
        def count_label_in_chunk(block, label):
            return np.sum(block == label)

        # For each label, apply count over all chunks and sum results
        def area_for_label(label):
            # Apply count function to each chunk and reduce by sum
            counts = da.map_blocks(
                lambda block: np.array([count_label_in_chunk(block, label)]),
                labels,
                dtype=np.int64,
                chunks=(1,),  # each chunk returns a scalar
            )
            return da.sum(counts).compute()  # compute to get scalar

        # Create delayed tasks for each label
        tasks = {
            int(label): delayed(area_for_label)(label)
            for label in unique_labels.compute()
        }
        # Compute all tasks in parallel
        results = compute(*tasks.values())
        # Return as list of (label, area)
        return list(zip(tasks.keys(), results, strict=True))
