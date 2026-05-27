from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from contextlib import suppress
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from napari.layers import Image, Labels
from napari.utils import notifications
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QColorDialog,
    QInputDialog,
    QProgressDialog,
)
from skimage.draw import polygon as draw_polygon


class DeleteAllWorker(QObject):
    finished = Signal(int)  # label_val
    error = Signal(str)
    progress = Signal(int, str)  # percent, message

    def __init__(self, data, label_val, current_z):
        super().__init__()
        self.data = data
        self.label_val = label_val
        self.current_z = (
            current_z  # Store the current slice index for range limiting
        )

    def run(self):
        try:
            delete_N = 25
            total_slices = self.data.shape[0]
            # Calculate slice range: 50 slices above and below current slice
            z_start = max(0, self.current_z - delete_N)
            z_end = min(
                total_slices, self.current_z + delete_N
            )  # +51 because end is exclusive
            slices_to_process = z_end - z_start
            found = False

            if isinstance(self.data, np.ndarray):
                # Process only the relevant slice range
                for idx, z in enumerate(range(z_start, z_end)):
                    if QThread.currentThread().isInterruptionRequested():
                        return
                    slice_data = self.data[z]
                    mask = slice_data == self.label_val
                    if np.any(mask):
                        slice_data[mask] = 0
                        # For numpy, slice_data is a view, so changes are in-place
                        found = True
                    # Emit progress based on processed slices within the range
                    progress_percent = int(100 * (idx + 1) / slices_to_process)
                    self.progress.emit(
                        progress_percent,
                        f'Processing slice {z + 1}/{total_slices} (range {z_start + 1}-{z_end})',
                    )

            elif isinstance(self.data, zarr.Array):
                # For zarr, process slice by slice within the range (simpler than chunk-based)
                for idx, z in enumerate(range(z_start, z_end)):
                    if QThread.currentThread().isInterruptionRequested():
                        return
                    slice_data = self.data[z]  # returns a numpy array
                    mask = slice_data == self.label_val
                    if np.any(mask):
                        slice_data[mask] = 0
                        self.data[z] = slice_data
                        found = True
                    progress_percent = int(100 * (idx + 1) / slices_to_process)
                    self.progress.emit(
                        progress_percent,
                        f'Processing slice {z + 1}/{total_slices} (range {z_start + 1}-{z_end})',
                    )
            else:
                self.error.emit(f'Unsupported data type: {type(self.data)}')
                return

            if found:
                self.finished.emit(self.label_val)
            else:
                # No label found in the processed range, still emit to notify
                self.finished.emit(self.label_val)
        except (ValueError, RuntimeError, OSError) as e:
            self.error.emit(str(e))


class DeleteInsideWorker(QObject):
    """Worker for deleting all labels intersecting a polygon across all Z."""

    finished = Signal(int)  # number of deleted labels
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, data, shapes, current_z):
        super().__init__()
        self.data = data
        self.shapes = shapes  # list of polygons (each as (N,2) or (N,3) if 3D)
        self.current_z = current_z
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        """
        Delete all labels that intersect the drawn polygon across all Z slices.
        Optimized by processing each slice only once, using np.isin to delete multiple labels simultaneously.
        """
        try:
            # --- Step 1: Generate polygon mask on the current slice ---
            slice_shape = self.data.shape[1:3]  # (Y, X)
            combined_mask = np.zeros(slice_shape, dtype=bool)
            for poly in self.shapes:
                # If polygon vertices include Z coordinate (ndim=3), discard the Z column
                if poly.shape[1] == 3:
                    poly = poly[:, 1:]  # keep only (y, x)

                poly = _smooth_closed_polygon_yx(
                    poly,
                    samples_per_segment=16,
                )

                rr, cc = draw_polygon(poly[:, 0], poly[:, 1], slice_shape)
                combined_mask[rr, cc] = True

            # --- Step 2: Identify which labels are inside the polygon on the current slice ---
            current_slice = self.data[self.current_z]
            labels_in_mask = np.unique(current_slice[combined_mask])
            labels_in_mask = labels_in_mask[
                labels_in_mask != 0
            ]  # exclude background

            if len(labels_in_mask) == 0:
                self.finished.emit(0)  # no labels to delete
                return

            # --- Step 3: Delete those labels across all slices, processing one slice at a time ---
            total_slices = self.data.shape[0]
            delete_N = 25
            z_start = max(0, self.current_z - delete_N)
            z_end = min(total_slices, self.current_z + delete_N + 1)
            total_steps = z_end - z_start + 1
            # for z in range(total_slices):
            for step, z in enumerate(range(z_start, z_end), start=1):
                if QThread.currentThread().isInterruptionRequested():
                    return

                # Get the current slice data (as a numpy array; for zarr this is a copy)
                slice_data = self.data[z]

                # Create a boolean mask marking all pixels that belong to any label in labels_in_mask
                # np.isin is efficient for a small set of labels and returns a boolean array of same shape
                mask = np.isin(slice_data, labels_in_mask)

                if np.any(mask):
                    # Set those pixels to background (0)
                    slice_data[mask] = 0

                    # If the underlying data is a zarr array, we need to write the modified slice back
                    if isinstance(self.data, zarr.Array):
                        self.data[z] = slice_data
                    # For numpy arrays, slice_data is a view (if contiguous) and changes are already in‑place,
                    # so no explicit write‑back is needed.

                # Update progress
                self.progress.emit(
                    int(100 * step / total_steps),
                    f'Processing slice {z + 1}/{total_slices}',
                )

            # --- Step 4: Notify completion with the number of labels deleted ---
            self.finished.emit(len(labels_in_mask))

        except (ValueError, RuntimeError, OSError) as e:
            self.error.emit(str(e))


class RefineSelectedCellWorker(QObject):
    """Worker for refining one selected cell in a background thread."""

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        image_crop,
        label_crop,
        label_val,
        bbox,
        gaussian_sigma=(0.0, 3.0, 3.0),
        min_seed_voxels=5,
        shift=(3, 6, 6),
        curve_thres=0.0,
        im_resolution=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.image_crop = image_crop
        self.label_crop = label_crop
        self.label_val = int(label_val)
        self.bbox = bbox
        self.gaussian_sigma = gaussian_sigma
        self.min_seed_voxels = int(min_seed_voxels)
        self.shift = shift
        self.curve_thres = float(curve_thres)
        self.im_resolution = im_resolution

    def run(self):
        try:
            self.progress.emit(5, 'Preparing local crop...')

            cell_mask_crop = self.label_crop == self.label_val
            if not np.any(cell_mask_crop):
                raise ValueError(
                    f'Label {self.label_val} not found in local crop.'
                )

            other_labels_crop = (self.label_crop > 0) & (
                self.label_crop != self.label_val
            )

            self.progress.emit(35, 'Running local graph-cut refinement...')
            from focus3d.basic.Princut_refine.local_refine import (
                refine_single_cell_graphcut,
            )

            refined_mask_crop = refine_single_cell_graphcut(
                image_crop=self.image_crop,
                cell_mask_crop=cell_mask_crop,
                other_labels_crop=other_labels_crop,
                gaussian_sigma=self.gaussian_sigma,
                min_seed_voxels=self.min_seed_voxels,
                shift=self.shift,
                curve_thres=self.curve_thres,
                im_resolution=self.im_resolution,
            )

            refined_mask_crop = np.asarray(refined_mask_crop).astype(bool)

            if refined_mask_crop.shape != cell_mask_crop.shape:
                raise ValueError(
                    f'Refined mask shape mismatch: expected {cell_mask_crop.shape}, '
                    f'got {refined_mask_crop.shape}.'
                )

            if not np.any(refined_mask_crop):
                raise ValueError('Refinement produced an empty mask.')

            self.progress.emit(80, 'Building replacement label crop...')

            new_label_crop = self.label_crop.copy()
            new_label_crop[new_label_crop == self.label_val] = 0

            refined_mask_crop[other_labels_crop] = False
            new_label_crop[refined_mask_crop] = self.label_val

            result = {
                'bbox': self.bbox,
                'label_val': self.label_val,
                'old_label_crop': self.label_crop,
                'new_label_crop': new_label_crop,
                'old_voxels': int(np.sum(cell_mask_crop)),
                'new_voxels': int(np.sum(refined_mask_crop)),
            }

            self.progress.emit(100, 'one-click segmentation finished.')
            self.finished.emit(result)

        except Exception:
            tb = traceback.format_exc()
            print('\n[ClickedLocalRefineWorker ERROR]\n', tb, flush=True)
            self.error.emit(tb)


def _get_one_click_backend_name():
    """
    Select backend for one-click segmentation.

    Default:
        Windows -> inference_win.py
        Linux   -> inference.py

    Override:
        CELLSEG_FOCUS3D_BACKEND=windows
        CELLSEG_FOCUS3D_BACKEND=detectron2
    """
    backend = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()

    if backend in {'windows', 'win', 'nod2', 'no_detectron2', 'pytorch'}:
        return 'windows'

    if backend in {'detectron2', 'd2', 'linux'}:
        return 'detectron2'

    return 'windows' if os.name == 'nt' else 'detectron2'


def _make_one_click_model_cache_key(
    config_file,
    weights_path,
    cuda_visible_devices=None,
):
    backend_name = _get_one_click_backend_name()

    return (
        backend_name,
        str(Path(config_file).expanduser().resolve()),
        str(Path(weights_path).expanduser().resolve()),
        str(cuda_visible_devices),
    )


class OneClickModelLoadWorker(QObject):
    """
    Worker for loading the one-click segmentation model in a background thread.

    This avoids blocking the napari UI when the user enters one-click mode.
    """

    progress = Signal(int, str)
    finished = Signal(object, object)  # model, cache_key
    error = Signal(str)

    def __init__(
        self,
        config_file,
        weights_path,
        cache_key,
        cuda_visible_devices=None,
    ):
        super().__init__()
        self.config_file = str(config_file)
        self.weights_path = str(weights_path)
        self.cache_key = cache_key
        self.cuda_visible_devices = cuda_visible_devices

    def run(self):
        try:
            self.progress.emit(5, 'Preparing one-click segmentation model...')

            if self.cuda_visible_devices:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(
                    self.cuda_visible_devices
                )

            backend_name = _get_one_click_backend_name()

            self.progress.emit(
                20,
                f'Importing {backend_name} one-click backend...',
            )

            if backend_name == 'windows':
                from focus3d.segmentation.FOCUS3D.inference_win import (
                    build_predictor,
                    setup_cfg,
                )
            else:
                from focus3d.segmentation.FOCUS3D.inference import (
                    build_predictor,
                    setup_cfg,
                )

            self.progress.emit(45, 'Building model configuration...')
            cfg = setup_cfg(self.config_file, self.weights_path)

            self.progress.emit(65, 'Loading model weights...')
            model = build_predictor(cfg)

            self.progress.emit(100, 'One-click segmentation model loaded.')
            self.finished.emit(model, self.cache_key)

        except Exception:
            self.error.emit(traceback.format_exc())


class ClickedLocalRefineWorker(QObject):
    """
    Worker for clicked one-click segmentation.

    The model must already be loaded and passed in.
    This worker only runs infer_clicked_instance(...).
    """

    progress = Signal(int, str)
    finished = Signal(object)  # result
    error = Signal(str)

    def __init__(
        self,
        image_input,
        coord_zyx,
        config_file,
        weights_path,
        model=None,
        patch_size=(32, 96, 96),
        score_thresh=0.05,
        click_prob_thresh=0.10,
        mask_thresh=0.50,
        min_voxels=20,
    ):
        super().__init__()
        self.image_input = image_input
        self.coord_zyx = tuple(int(v) for v in coord_zyx)
        self.config_file = str(config_file)
        self.weights_path = str(weights_path)
        self.model = model
        if self.model is None:
            raise ValueError(
                'ClickedLocalRefineWorker requires a pre-loaded model. '
                'Please call _load_one_click_segmentation_model_once(self) before starting the worker.'
            )
        self.patch_size = tuple(int(v) for v in patch_size)
        self.score_thresh = float(score_thresh)
        self.click_prob_thresh = float(click_prob_thresh)
        self.mask_thresh = float(mask_thresh)
        self.min_voxels = int(min_voxels)

    def run(self):
        try:
            self.progress.emit(
                5, 'Preparing clicked one-click segmentation...'
            )

            # curation.py is in src/focus3d/basic/curation.py
            # Mask2former root is src/focus3d/segmentation/Mask2former
            self.progress.emit(15, 'Importing clicked inference modules...')

            from focus3d.segmentation.FOCUS3D.click_inference import (
                infer_clicked_instance,
            )

            backend_name = _get_one_click_backend_name()

            print(
                '[ClickedLocalRefineWorker] backend=',
                backend_name,
                '| infer_clicked_instance module=',
                infer_clicked_instance.__module__,
                flush=True,
            )

            model = self.model
            if model is None:
                raise RuntimeError(
                    'One-click segmentation model is not loaded. '
                    'This should not happen if _enter_local_refinement_mode() loaded the model correctly.'
                )

            self.progress.emit(
                45, f'Running clicked inference at zyx={self.coord_zyx}...'
            )

            result = infer_clicked_instance(
                image=self.image_input,
                coord_zyx=self.coord_zyx,
                model=model,
                patch_size=self.patch_size,
                score_thresh=self.score_thresh,
                click_prob_thresh=self.click_prob_thresh,
                mask_thresh=self.mask_thresh,
                min_voxels=self.min_voxels,
                return_full_size=True,
                show_result=False,
            )

            if not isinstance(result, dict):
                raise RuntimeError(
                    'infer_clicked_instance did not return a dict.'
                )

            result['coord_zyx'] = self.coord_zyx

            if not result.get('success', False):
                reason = result.get('reason', 'unknown reason')
                raise RuntimeError(f'Clicked inference failed: {reason}')

            if result.get('instance_mask_full', None) is None:
                raise RuntimeError(
                    'Clicked inference succeeded, but instance_mask_full is missing.'
                )

            self.progress.emit(100, 'Clicked one-click segmentation finished.')
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


def _smooth_closed_polygon_yx(poly_yx, samples_per_segment=16, eps=1e-6):
    """
    Convert sparse polygon control points into a dense smooth closed curve.

    Input:
        poly_yx: (N, 2), columns are (Y, X)
    Output:
        smooth_yx: (M, 2), dense closed contour points in (Y, X)

    Uses closed Catmull-Rom spline.
    No scipy dependency.
    """
    pts = np.asarray(poly_yx, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts

    pts = pts[:, :2]
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if pts.shape[0] < 3:
        return pts

    # If napari stores the first point again as the last point, remove it.
    if np.linalg.norm(pts[0] - pts[-1]) < eps:
        pts = pts[:-1]

    # Remove consecutive duplicated points.
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= eps:
            cleaned.append(p)

    pts = np.asarray(cleaned, dtype=np.float64)

    if pts.shape[0] < 3:
        return pts

    n = pts.shape[0]
    samples_per_segment = max(4, int(samples_per_segment))

    smooth = []

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)

        for t in ts:
            t2 = t * t
            t3 = t2 * t

            # Uniform Catmull-Rom spline.
            q = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            smooth.append(q)

    return np.asarray(smooth, dtype=np.float64)


def _get_first_image_layer_for_refine(self):
    active = self.viewer.layers.selection.active
    if isinstance(active, Image):
        return active

    for layer in self.viewer.layers:
        if isinstance(layer, Image):
            return layer

    return None


def _refine_selected_cell(self):
    """
    Start background refinement for the currently selected cell.
    """
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        notifications.show_error('No labels layer found.')
        return

    image_layer = self._get_first_image_layer_for_refine()
    if image_layer is None:
        notifications.show_error('No image layer found.')
        return

    label_val = int(labels_layer.selected_label)
    if label_val == 0:
        notifications.show_warning(
            'Please select a non-background cell first.'
        )
        return

    label_data = labels_layer.data
    image_data = image_layer.data

    if label_data.ndim != 3 or image_data.ndim != 3:
        notifications.show_error(
            'Refine Selected Cell expects 3D image and 3D labels.'
        )
        return

    if tuple(label_data.shape) != tuple(image_data.shape):
        notifications.show_error(
            f'Image and label shape mismatch: image={image_data.shape}, '
            f'label={label_data.shape}.'
        )
        return

    bbox = self._get_selected_cell_bbox_3d(
        labels_layer=labels_layer,
        label_val=label_val,
        expand_factor=2.0,
        min_size=(4, 20, 20),
        max_empty_gap=1,
        require_current_slice=True,
    )

    if bbox is None:
        notifications.show_warning(
            f'Label {label_val} not found in 3D volume.'
        )
        return

    z0, z1, y0, y1, x0, x1 = bbox

    image_crop = np.asarray(image_data[z0:z1, y0:y1, x0:x1]).copy()
    label_crop = np.asarray(label_data[z0:z1, y0:y1, x0:x1]).copy()

    if not np.any(label_crop == label_val):
        notifications.show_warning(
            f'Label {label_val} not found in local crop.'
        )
        return

    self._cell_refine_active = True
    self._update_curation_controls()

    self.cell_refine_progress = QProgressDialog(
        'Refining selected cell...', 'Cancel', 0, 100, self
    )
    self.cell_refine_progress.setWindowTitle('Cell Refinement')
    self.cell_refine_progress.setAutoClose(True)
    self.cell_refine_progress.setAutoReset(True)
    self.cell_refine_progress.show()

    self.cell_refine_thread = QThread()
    self.cell_refine_worker = RefineSelectedCellWorker(
        image_crop=image_crop,
        label_crop=label_crop,
        label_val=label_val,
        bbox=bbox,
        gaussian_sigma=(0.0, 3.0, 3.0),
        min_seed_voxels=5,
        shift=(3, 6, 6),
        curve_thres=0.0,
        im_resolution=(1.0, 1.0, 1.0),
    )
    self.cell_refine_worker.moveToThread(self.cell_refine_thread)

    self.cell_refine_worker.progress.connect(
        lambda val, msg: self.cell_refine_progress.setValue(val)
    )
    self.cell_refine_worker.progress.connect(
        lambda val, msg: self.cell_refine_progress.setLabelText(msg)
    )
    self.cell_refine_worker.finished.connect(
        self._on_refine_selected_cell_finished
    )
    self.cell_refine_worker.error.connect(self._on_refine_selected_cell_error)

    self.cell_refine_progress.canceled.connect(
        self.cell_refine_thread.requestInterruption
    )
    self.cell_refine_progress.canceled.connect(
        self._on_refine_selected_cell_cancelled
    )

    self.cell_refine_thread.started.connect(self.cell_refine_worker.run)
    self.cell_refine_worker.finished.connect(self.cell_refine_thread.quit)
    self.cell_refine_worker.error.connect(self.cell_refine_thread.quit)
    self.cell_refine_worker.finished.connect(
        self.cell_refine_worker.deleteLater
    )
    self.cell_refine_worker.error.connect(self.cell_refine_worker.deleteLater)
    self.cell_refine_thread.finished.connect(
        self.cell_refine_thread.deleteLater
    )

    self.cell_refine_thread.start()


def _on_refine_selected_cell_finished(self, result):
    """
    Write the refined crop back to the labels layer in the main thread.
    """
    if (
        hasattr(self, 'cell_refine_progress')
        and self.cell_refine_progress is not None
    ):
        self.cell_refine_progress.close()

    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        self._cell_refine_active = False
        self._update_curation_controls()
        notifications.show_error('No labels layer found after refinement.')
        return

    bbox = result['bbox']
    label_val = int(result['label_val'])
    old_label_crop = result['old_label_crop']
    new_label_crop = result['new_label_crop']

    z0, z1, y0, y1, x0, x1 = bbox

    if not hasattr(self, '_cell_refine_undo_stack'):
        self._cell_refine_undo_stack = []

    self._cell_refine_undo_stack.append(
        {
            'labels_layer': labels_layer,
            'label_val': label_val,
            'bbox': bbox,
            'old_crop': old_label_crop,
        }
    )

    max_undo = 10
    if len(self._cell_refine_undo_stack) > max_undo:
        self._cell_refine_undo_stack.pop(0)

    labels_layer.data[z0:z1, y0:y1, x0:x1] = new_label_crop
    labels_layer.selected_label = label_val
    labels_layer.refresh()

    self._append_log_entry(
        operation='refine selected cell',
        label_id_or_count=label_val,
        z_index=int(self.viewer.dims.current_step[0]),
        note=(
            f'Refined selected cell {label_val}. '
            f'bbox=(z:{z0}-{z1}, y:{y0}-{y1}, x:{x0}-{x1}); '
            f'old_voxels={result["old_voxels"]}, '
            f'new_voxels={result["new_voxels"]}.'
        ),
    )

    self._cell_refine_active = False
    self._update_curation_controls()
    self.viewer.layers.selection.active = labels_layer

    notifications.show_info(
        f'Refined cell {label_val}: '
        f'{result["old_voxels"]} → {result["new_voxels"]} voxels. Ctrl+Z to undo.'
    )


def _on_refine_selected_cell_error(self, error_msg):
    """
    Handle background refinement failure.
    """
    if (
        hasattr(self, 'cell_refine_progress')
        and self.cell_refine_progress is not None
    ):
        self.cell_refine_progress.close()

    self._cell_refine_active = False
    self._update_curation_controls()

    notifications.show_error(f'Refine selected cell failed: {error_msg}')


def _on_refine_selected_cell_cancelled(self):
    """
    Handle user cancellation.

    The current worker is not force-killed during graph cut. The cancel button
    only requests interruption and closes the progress dialog when the worker
    exits.
    """
    notifications.show_info('Cell refinement cancellation requested.')


def _undo_cell_refinement(self, viewer=None):
    """
    Undo latest Refine Selected Cell operation.
    """
    if not hasattr(self, '_cell_refine_undo_stack'):
        notifications.show_info('No cell refinement history.')
        return

    if len(self._cell_refine_undo_stack) == 0:
        notifications.show_info('No cell refinement operation to undo.')
        return

    item = self._cell_refine_undo_stack.pop()

    labels_layer = item['labels_layer']
    bbox = item['bbox']
    old_crop = item['old_crop']
    label_val = item['label_val']

    if labels_layer is None or labels_layer not in self.viewer.layers:
        notifications.show_warning(
            'The original labels layer no longer exists.'
        )
        return

    z0, z1, y0, y1, x0, x1 = bbox

    labels_layer.data[z0:z1, y0:y1, x0:x1] = old_crop
    labels_layer.selected_label = label_val
    labels_layer.refresh()

    notifications.show_info(f'Undo refinement for cell {label_val}.')


def _choose_label_dtype(max_label: int):
    """
    Choose a compact unsigned dtype for label volume.
    """
    if max_label <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_label <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        return np.uint64


def _prepare_labels_for_curation(
    self, labels, output_dir=None, name='curation_labels'
):
    """
    Convert segmentation labels to an editable backend for manual curation.

    Preferred backend:
        zarr.Array

    Why:
        - editable by slice / bbox
        - lower memory pressure than full numpy
        - compatible with current manual curation code
    """
    import shutil
    from pathlib import Path

    import dask.array as da
    import numpy as np
    import zarr

    # Already editable and memory-resident.
    if isinstance(labels, np.ndarray):
        max_label = int(labels.max()) if labels.size > 0 else 0
        dtype = _choose_label_dtype(max_label)
        labels = labels.astype(dtype, copy=False)

        # For small arrays, numpy is fine.
        # For consistency and low memory pressure, still save to zarr if output_dir is available.
        if output_dir is None:
            return labels

    # Decide where to store editable zarr.
    if output_dir is None or str(output_dir).strip() == '':
        output_dir = Path.cwd() / 'focus3d_curation_output'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = output_dir / f'{name}.zarr'

    # Remove old curation zarr from previous segmentation result.
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    # Case 1: Dask labels.
    if isinstance(labels, da.Array):
        max_label = int(da.max(labels).compute())
        dtype = _choose_label_dtype(max_label)
        labels = labels.astype(dtype)

        # Use chunks optimized for z-slice/bbox editing.
        chunks = labels.chunksize
        if chunks is None:
            chunks = (1, min(512, labels.shape[1]), min(512, labels.shape[2]))

        if len(labels.shape) == 3:
            chunks = (
                min(4, labels.shape[0]),
                min(512, labels.shape[1]),
                min(512, labels.shape[2]),
            )

        labels = labels.rechunk(chunks)
        da.to_zarr(labels, str(zarr_path), overwrite=True, compute=True)

        editable = zarr.open(str(zarr_path), mode='r+')
        return editable

    # Case 2: numpy labels, save as zarr to reduce long-term RAM pressure.
    labels_np = np.asarray(labels)
    max_label = int(labels_np.max()) if labels_np.size > 0 else 0
    dtype = _choose_label_dtype(max_label)
    labels_np = labels_np.astype(dtype, copy=False)

    if labels_np.ndim == 3:
        chunks = (
            min(4, labels_np.shape[0]),
            min(512, labels_np.shape[1]),
            min(512, labels_np.shape[2]),
        )
    else:
        chunks = None

    editable = zarr.open(
        str(zarr_path),
        mode='w',
        shape=labels_np.shape,
        dtype=labels_np.dtype,
        chunks=chunks,
    )
    editable[:] = labels_np
    return editable


def _get_selected_cell_bbox_3d(
    self,
    labels_layer,
    label_val,
    expand_factor=2.0,
    min_size=(4, 20, 20),
    max_empty_gap=1,
    require_current_slice=True,
):
    """
    Find selected cell bbox by scanning around the current Z slice.

    This avoids scanning the full 3D label volume.

    Parameters
    ----------
    labels_layer : napari.layers.Labels
        Label layer containing the selected cell.

    label_val : int
        Selected cell ID.

    expand_factor : float
        Enlarge the detected bbox by this factor.

    min_size : tuple[int, int, int]
        Minimum crop size in (Z, Y, X).

    max_empty_gap : int
        Stop scanning in one direction after this many consecutive empty slices.
        For example, max_empty_gap=1 stops after one empty slice.

    require_current_slice : bool
        If True, the selected label must exist on the current Z slice.
        This is recommended for interactive refinement.
    """
    data = labels_layer.data

    if label_val is None or int(label_val) == 0:
        return None

    Z, Y, X = data.shape
    current_z = int(self.viewer.dims.current_step[0])
    current_z = max(0, min(current_z, Z - 1))

    def slice_bbox(z):
        """Return 2D bbox of label_val on one slice, or None."""
        sl = np.asarray(data[z])
        yy, xx = np.where(sl == label_val)

        if len(yy) == 0:
            return None

        return (
            int(yy.min()),
            int(yy.max()) + 1,
            int(xx.min()),
            int(xx.max()) + 1,
        )

    current_bbox = slice_bbox(current_z)

    if current_bbox is None:
        if require_current_slice:
            notifications.show_warning(
                f'Label {label_val} is not visible on current Z slice {current_z}. '
                'Please click the cell on a slice where it is visible.'
            )
            return None

        # Fallback: local search around current_z only, not full volume.
        # You can increase this radius if needed.
        search_radius = 5
        for dz in range(1, search_radius + 1):
            for z in (current_z - dz, current_z + dz):
                if z < 0 or z >= Z:
                    continue
                bbox = slice_bbox(z)
                if bbox is not None:
                    current_z = z
                    current_bbox = bbox
                    break
            if current_bbox is not None:
                break

        if current_bbox is None:
            return None

    z_hits = []
    y0_list, y1_list = [], []
    x0_list, x1_list = [], []

    def add_slice(z, bbox):
        y0, y1, x0, x1 = bbox
        z_hits.append(int(z))
        y0_list.append(int(y0))
        y1_list.append(int(y1))
        x0_list.append(int(x0))
        x1_list.append(int(x1))

    add_slice(current_z, current_bbox)

    # Scan backward from current_z.
    empty_gap = 0
    for z in range(current_z - 1, -1, -1):
        bbox = slice_bbox(z)
        if bbox is None:
            empty_gap += 1
            if empty_gap > max_empty_gap:
                break
            continue

        empty_gap = 0
        add_slice(z, bbox)

    # Scan forward from current_z.
    empty_gap = 0
    for z in range(current_z + 1, Z):
        bbox = slice_bbox(z)
        if bbox is None:
            empty_gap += 1
            if empty_gap > max_empty_gap:
                break
            continue

        empty_gap = 0
        add_slice(z, bbox)

    if len(z_hits) == 0:
        return None

    z0 = min(z_hits)
    z1 = max(z_hits) + 1
    y0 = min(y0_list)
    y1 = max(y1_list)
    x0 = min(x0_list)
    x1 = max(x1_list)

    def expand_one_axis(a0, a1, max_len, factor, min_len):
        center = 0.5 * (a0 + a1)
        length = max(1.0, float(a1 - a0))

        new_length = length * float(factor)
        new_length = max(new_length, float(min_len))

        b0 = int(np.floor(center - new_length / 2.0))
        b1 = int(np.ceil(center + new_length / 2.0))

        # Shift the box back into the valid range while preserving size
        # as much as possible.
        if b0 < 0:
            b1 -= b0
            b0 = 0

        if b1 > max_len:
            b0 -= b1 - max_len
            b1 = max_len

        b0 = max(0, b0)
        b1 = min(max_len, b1)

        return b0, b1

    min_z, min_y, min_x = min_size

    z0, z1 = expand_one_axis(z0, z1, Z, expand_factor, min_z)
    y0, y1 = expand_one_axis(y0, y1, Y, expand_factor, min_y)
    x0, x1 = expand_one_axis(x0, x1, X, expand_factor, min_x)

    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return None

    return z0, z1, y0, y1, x0, x1


def _start_one_click_model_loading(
    self,
    config_file,
    weights_path,
    cache_key,
    cuda_visible_devices=None,
):
    """
    Start loading the one-click segmentation model in a QThread.

    UI remains responsive while setup_cfg/build_predictor are running.
    """
    # Avoid starting duplicated loading threads.
    if getattr(self, '_local_refine_loading', False):
        notifications.show_info(
            'One-click segmentation model is already loading.'
        )
        return

    self._local_refine_loading = True
    self._local_refine_pending_cache_key = cache_key
    self._local_refine_pending_config_file = str(config_file)
    self._local_refine_pending_weights_path = str(weights_path)

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(False)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(False)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText(
            'Loading one-click segmentation model...'
        )
        self.local_refine_status_label.setStyleSheet('color: #ffcc66;')

    self.local_refine_model_progress = QProgressDialog(
        'Loading one-click segmentation model...',
        'Cancel',
        0,
        100,
        self,
    )
    self.local_refine_model_progress.setWindowTitle(
        'One-click Segmentation Model Loading'
    )
    self.local_refine_model_progress.setAutoClose(False)
    self.local_refine_model_progress.setAutoReset(False)
    self.local_refine_model_progress.show()

    self.local_refine_model_thread = QThread()
    self.local_refine_model_worker = OneClickModelLoadWorker(
        config_file=config_file,
        weights_path=weights_path,
        cache_key=cache_key,
        cuda_visible_devices=cuda_visible_devices,
    )

    self.local_refine_model_worker.moveToThread(self.local_refine_model_thread)

    self.local_refine_model_thread.started.connect(
        self.local_refine_model_worker.run
    )

    self.local_refine_model_worker.progress.connect(
        lambda val, msg: self.local_refine_model_progress.setValue(int(val))
    )
    self.local_refine_model_worker.progress.connect(
        lambda val, msg: self.local_refine_model_progress.setLabelText(
            str(msg)
        )
    )

    self.local_refine_model_worker.finished.connect(
        self._on_one_click_model_loaded
    )
    self.local_refine_model_worker.error.connect(
        self._on_one_click_model_load_error
    )

    # Loading a torch model cannot always be interrupted safely.
    # We only close the dialog and mark the result as ignored.
    self.local_refine_model_worker.finished.connect(
        self.local_refine_model_thread.quit
    )
    self.local_refine_model_worker.error.connect(
        self.local_refine_model_thread.quit
    )

    self.local_refine_model_worker.finished.connect(
        self.local_refine_model_worker.deleteLater
    )
    self.local_refine_model_worker.error.connect(
        self.local_refine_model_worker.deleteLater
    )
    self.local_refine_model_thread.finished.connect(
        self.local_refine_model_thread.deleteLater
    )

    self.local_refine_model_thread.start()


def _on_one_click_model_loaded(self, model, cache_key):
    """
    Called in the main thread after the one-click model is loaded.
    """
    if (
        hasattr(self, 'local_refine_model_progress')
        and self.local_refine_model_progress is not None
    ):
        self.local_refine_model_progress.close()

    self._local_refine_loading = False

    # If config/checkpoint changed during loading, ignore stale model.
    pending_key = getattr(self, '_local_refine_pending_cache_key', None)
    if pending_key != cache_key:
        if hasattr(self, 'btn_enter_local_refinement'):
            self.btn_enter_local_refinement.setEnabled(True)
        if hasattr(self, 'btn_exit_local_refinement'):
            self.btn_exit_local_refinement.setEnabled(False)
        if hasattr(self, 'local_refine_status_label'):
            self.local_refine_status_label.setText(
                'One-click segmentation inactive'
            )
            self.local_refine_status_label.setStyleSheet('color: #aaaaaa;')

        notifications.show_warning(
            'One-click segmentation model was loaded, but config/checkpoint changed. '
            'Please enter one-click mode again.'
        )
        return

    self._local_refine_model = model
    self._local_refine_model_key = cache_key
    self._local_refine_config_file = getattr(
        self,
        '_local_refine_pending_config_file',
        None,
    )
    self._local_refine_weights_path = getattr(
        self,
        '_local_refine_pending_weights_path',
        None,
    )

    self._local_refine_mode_active = True
    self._local_refine_busy = False

    self._install_local_refinement_callbacks()

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(False)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(True)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText(
            'Active: click a cell in the viewer'
        )
        self.local_refine_status_label.setStyleSheet('color: #66cc66;')

    notifications.show_info(
        'One-click segmentation model loaded. Click a cell to segment it.'
    )


def _on_one_click_model_load_error(self, error_msg):
    """
    Called when model loading fails.
    """
    if (
        hasattr(self, 'local_refine_model_progress')
        and self.local_refine_model_progress is not None
    ):
        self.local_refine_model_progress.close()

    self._local_refine_loading = False
    self._local_refine_mode_active = False
    self._local_refine_busy = False

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(True)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(False)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText(
            'One-click segmentation inactive'
        )
        self.local_refine_status_label.setStyleSheet('color: #aaaaaa;')

    notifications.show_error(
        f'Failed to load one-click segmentation model:\n{error_msg}'
    )


def _cancel_one_click_model_loading(self):
    """
    Mark current model loading result as ignored.

    Torch model loading cannot always be force-stopped safely, so we do not kill
    the thread. We just ignore its result when it finishes.
    """
    self._local_refine_load_cancelled = True
    self._local_refine_loading = False

    if (
        hasattr(self, 'local_refine_model_progress')
        and self.local_refine_model_progress is not None
    ):
        self.local_refine_model_progress.close()

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(True)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(False)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText(
            'One-click segmentation inactive'
        )
        self.local_refine_status_label.setStyleSheet('color: #aaaaaa;')

    notifications.show_info(
        'One-click segmentation model loading cancellation requested.'
    )


def _enter_local_refinement_mode(self):
    """
    Enter one-click segmentation mode.

    Model loading is done in a background QThread, so the napari UI will not
    freeze when the user clicks Enter One-click Segmentation.
    """
    image_layer = self._get_first_image_layer_for_refine()
    labels_layer = self._get_labels_layer()

    if image_layer is None:
        notifications.show_error('Please load an image layer first.')
        return

    if labels_layer is None:
        notifications.show_error('Please load or create a label layer first.')
        return

    if image_layer.data.ndim != 3 or labels_layer.data.ndim != 3:
        notifications.show_error(
            'One-click segmentation expects 3D image and 3D labels in Z/Y/X order.'
        )
        return

    if tuple(image_layer.data.shape) != tuple(labels_layer.data.shape):
        notifications.show_error(
            f'Image and label shape mismatch: '
            f'image={image_layer.data.shape}, label={labels_layer.data.shape}.'
        )
        return

    checkpoint = self.checkpoint_edit.text().strip()
    config_file = self.seg_config_edit.text().strip()

    if not checkpoint:
        checkpoint = self._default_mask2former_checkpoint_text()
        self.checkpoint_edit.setText(checkpoint)

    if not config_file:
        config_file = self._default_mask2former_config_text()
        self.seg_config_edit.setText(config_file)

    checkpoint_real_path = self._resolve_mask2former_backend_path(checkpoint)
    config_real_path = self._resolve_mask2former_backend_path(config_file)

    if not checkpoint_real_path.exists():
        notifications.show_error(
            f'Checkpoint file not found:\n'
            f'UI path: {checkpoint}\n'
            f'Resolved path: {checkpoint_real_path}'
        )
        return

    if not config_real_path.exists():
        notifications.show_error(
            f'Config file not found:\n'
            f'UI path: {config_file}\n'
            f'Resolved path: {config_real_path}'
        )
        return

    cuda_visible_devices = None
    if hasattr(self, '_selected_cuda_visible_devices'):
        cuda_visible_devices = self._selected_cuda_visible_devices()

    cache_key = _make_one_click_model_cache_key(
        config_file=config_real_path,
        weights_path=checkpoint_real_path,
        cuda_visible_devices=cuda_visible_devices,
    )

    # Case 1: the correct model is already cached.
    cached_model = getattr(self, '_local_refine_model', None)
    cached_key = getattr(self, '_local_refine_model_key', None)

    if cached_model is not None and cached_key == cache_key:
        self._local_refine_config_file = str(config_real_path)
        self._local_refine_weights_path = str(checkpoint_real_path)

        self._local_refine_mode_active = True
        self._local_refine_busy = False

        self._install_local_refinement_callbacks()

        self.btn_enter_local_refinement.setEnabled(False)
        self.btn_exit_local_refinement.setEnabled(True)

        if hasattr(self, 'local_refine_status_label'):
            self.local_refine_status_label.setText(
                'Active: click a cell in the viewer'
            )
            self.local_refine_status_label.setStyleSheet('color: #66cc66;')

        notifications.show_info(
            'One-click segmentation mode activated. Reusing loaded model.'
        )
        return

    # Case 2: model is not loaded or config/checkpoint changed.
    self._local_refine_mode_active = False
    self._local_refine_busy = False
    self._local_refine_load_cancelled = False

    _start_one_click_model_loading(
        self,
        config_file=str(config_real_path),
        weights_path=str(checkpoint_real_path),
        cache_key=cache_key,
        cuda_visible_devices=cuda_visible_devices,
    )


def _exit_local_refinement_mode(self):
    """
    Exit brush-like one-click segmentation mode.
    """
    self._local_refine_mode_active = False
    self._remove_local_refinement_callbacks()

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(True)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(False)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText('Inactive')
        self.local_refine_status_label.setStyleSheet('color: #aaaaaa;')

    notifications.show_info('One-click segmentation mode deactivated.')


def _start_clicked_local_refinement(self, coord_zyx):
    """
    Start clicked one-click segmentation in a background thread.
    """
    image_layer = self._get_first_image_layer_for_refine()
    labels_layer = self._get_labels_layer()

    if image_layer is None or labels_layer is None:
        notifications.show_error(
            'Please load both image and label layers first.'
        )
        return

    # Prefer original image path, because infer_clicked_instance already supports
    # file path and avoids copying the whole volume.
    image_input = None
    try:
        image_input = getattr(image_layer.source, 'path', None)
    except Exception:
        image_input = None

    if image_input is None or str(image_input).strip() == '':
        # Fallback: pass the actual array.
        # click_inference.py should support ndarray input.
        image_input = np.asarray(image_layer.data)
    else:
        image_input = str(image_input)

    self._local_refine_busy = True

    if hasattr(self, 'btn_enter_local_refinement'):
        self.btn_enter_local_refinement.setEnabled(False)

    if hasattr(self, 'btn_exit_local_refinement'):
        self.btn_exit_local_refinement.setEnabled(False)

    if hasattr(self, 'local_refine_status_label'):
        self.local_refine_status_label.setText(
            f'Running at zyx={coord_zyx}...'
        )
        self.local_refine_status_label.setStyleSheet('color: #ffcc66;')

    self.local_refine_progress = QProgressDialog(
        'Starting clicked one-click segmentation...',
        'Cancel',
        0,
        100,
        self,
    )
    self.local_refine_progress.setWindowTitle('one-click segmentation')
    self.local_refine_progress.setAutoClose(True)
    self.local_refine_progress.setAutoReset(True)
    self.local_refine_progress.show()

    patch_size = (
        int(self.seg_patch_size_z_spin.value()),
        int(self.seg_patch_size_y_spin.value()),
        int(self.seg_patch_size_x_spin.value()),
    )

    model = getattr(self, '_local_refine_model', None)
    if model is None:
        self._local_refine_busy = False
        notifications.show_error(
            'One-click segmentation model is not loaded yet. '
            'Please wait until model loading finishes.'
        )
        return

    self.local_refine_thread = QThread()
    self.local_refine_worker = ClickedLocalRefineWorker(
        image_input=image_input,
        coord_zyx=coord_zyx,
        config_file=self._local_refine_config_file,
        weights_path=self._local_refine_weights_path,
        model=model,
        patch_size=patch_size,
        score_thresh=0.02,
        click_prob_thresh=0.05,
        mask_thresh=0.50,
        min_voxels=10,
    )

    self.local_refine_worker.moveToThread(self.local_refine_thread)

    self.local_refine_worker.progress.connect(
        lambda val, msg: self.local_refine_progress.setValue(val)
    )
    self.local_refine_worker.progress.connect(
        lambda val, msg: self.local_refine_progress.setLabelText(msg)
    )

    self.local_refine_worker.finished.connect(
        self._on_clicked_local_refinement_finished
    )
    self.local_refine_worker.error.connect(
        self._on_clicked_local_refinement_error
    )

    self.local_refine_progress.canceled.connect(
        self.local_refine_thread.requestInterruption
    )

    self.local_refine_thread.started.connect(self.local_refine_worker.run)
    self.local_refine_worker.finished.connect(self.local_refine_thread.quit)
    self.local_refine_worker.error.connect(self.local_refine_thread.quit)
    self.local_refine_worker.finished.connect(
        self.local_refine_worker.deleteLater
    )
    self.local_refine_worker.error.connect(
        self.local_refine_worker.deleteLater
    )
    self.local_refine_thread.finished.connect(
        self.local_refine_thread.deleteLater
    )

    self.local_refine_thread.start()


def _bbox_from_mask_3d(mask):
    zz, yy, xx = np.where(mask)
    if len(zz) == 0:
        return None

    return (
        int(zz.min()),
        int(zz.max()) + 1,
        int(yy.min()),
        int(yy.max()) + 1,
        int(xx.min()),
        int(xx.max()) + 1,
    )


def _union_bboxes_3d(bboxes, shape, margin=2):
    bboxes = [b for b in bboxes if b is not None]
    if len(bboxes) == 0:
        return None

    Z, Y, X = shape

    z0 = min(b[0] for b in bboxes)
    z1 = max(b[1] for b in bboxes)
    y0 = min(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    x0 = min(b[4] for b in bboxes)
    x1 = max(b[5] for b in bboxes)

    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)

    z1 = min(Z, z1 + margin)
    y1 = min(Y, y1 + margin)
    x1 = min(X, x1 + margin)

    return z0, z1, y0, y1, x0, x1


def _on_clicked_local_refinement_finished(self, result):
    """
    Write clicked inference result back to the current Labels layer.

    Important:
    - labels_layer.data may be numpy, dask, or zarr.
    - For zarr, do NOT do global operations like label_data > 0.
    - Only read/write the local bbox crop.
    """
    if (
        hasattr(self, 'local_refine_progress')
        and self.local_refine_progress is not None
    ):
        self.local_refine_progress.close()

    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        self._local_refine_busy = False
        notifications.show_error(
            'No labels layer found after one-click segmentation.'
        )
        return

    label_data = labels_layer.data

    # Dask arrays are not a good editable backend.
    # Convert dask to numpy only if it appears here.
    # Zarr should stay zarr because it supports local read/write.
    if isinstance(label_data, da.Array):
        notifications.show_info(
            'Converting Dask labels to NumPy for one-click segmentation...'
        )
        label_data = label_data.compute()
        labels_layer.data = label_data

    instance_mask = np.asarray(result['instance_mask_full']).astype(bool)
    coord_zyx = tuple(int(v) for v in result['coord_zyx'])
    z, y, x = coord_zyx

    if tuple(instance_mask.shape) != tuple(label_data.shape):
        self._local_refine_busy = False
        notifications.show_error(
            f'Mask shape mismatch: mask={instance_mask.shape}, '
            f'label={label_data.shape}.'
        )
        return

    if not np.any(instance_mask):
        self._local_refine_busy = False
        notifications.show_warning('Clicked inference returned an empty mask.')
        return

    # Read clicked voxel safely. For zarr this returns scalar-like value.
    clicked_label = int(np.asarray(label_data[z, y, x]))

    if clicked_label > 0:
        target_label = clicked_label
        mode_note = 'replace existing label'
    else:
        target_label = self._consume_next_label_id(labels_layer)
        mode_note = 'create new label'

    # ------------------------------------------------------------
    # 1. New mask bbox from clicked inference result
    # ------------------------------------------------------------
    new_bbox = _bbox_from_mask_3d(instance_mask)
    if new_bbox is None:
        self._local_refine_busy = False
        notifications.show_warning('Cannot locate refined mask bbox.')
        return

    # ------------------------------------------------------------
    # 2. Estimate old bbox locally if replacing an existing label
    #    Do not use global label_data == target_label on zarr.
    # ------------------------------------------------------------
    old_bbox = None

    if clicked_label > 0:
        old_bbox = self._get_label_bbox_near_click_for_local_refine(
            label_data=label_data,
            label_val=target_label,
            coord_zyx=coord_zyx,
            z_radius=25,
        )

    # Union old and new bbox, then only operate inside this local crop.
    bbox = _union_bboxes_3d(
        [old_bbox, new_bbox],
        shape=label_data.shape,
        margin=2,
    )

    if bbox is None:
        self._local_refine_busy = False
        notifications.show_warning('Cannot locate refinement bbox.')
        return

    z0, z1, y0, y1, x0, x1 = bbox

    # ------------------------------------------------------------
    # 3. Work only on local crop
    # ------------------------------------------------------------
    old_crop = np.asarray(label_data[z0:z1, y0:y1, x0:x1]).copy()
    new_crop = old_crop.copy()

    mask_crop = instance_mask[z0:z1, y0:y1, x0:x1]

    # Prevent overwriting neighboring cells.
    # This is the local-crop replacement of:
    #     other_labels = (label_data > 0) & (label_data != target_label)
    other_labels_crop = (old_crop > 0) & (old_crop != target_label)
    mask_crop = mask_crop.copy()
    mask_crop[other_labels_crop] = False

    if not np.any(mask_crop):
        self._local_refine_busy = False
        notifications.show_warning(
            'Refined mask only overlaps other labels. Nothing was changed.'
        )
        return

    # Remove the old version of this cell only inside the local bbox.
    new_crop[new_crop == target_label] = 0

    # Write refined mask as target label.
    new_crop[mask_crop] = target_label

    # ------------------------------------------------------------
    # 4. Save undo information
    # ------------------------------------------------------------
    if not hasattr(self, '_local_refine_undo_stack'):
        self._local_refine_undo_stack = []

    self._local_refine_undo_stack.append(
        {
            'labels_layer': labels_layer,
            'bbox': bbox,
            'old_crop': old_crop,
            'label_val': target_label,
        }
    )

    max_undo = 20
    if len(self._local_refine_undo_stack) > max_undo:
        self._local_refine_undo_stack.pop(0)

    # ------------------------------------------------------------
    # 5. Write local crop back.
    #    Works for both numpy and zarr.
    # ------------------------------------------------------------
    label_data[z0:z1, y0:y1, x0:x1] = new_crop

    labels_layer.selected_label = target_label
    labels_layer.refresh()

    old_voxels = int(np.sum(old_crop == target_label))
    new_voxels = int(np.sum(mask_crop))

    self._append_log_entry(
        operation='clicked one-click segmentation',
        label_id_or_count=target_label,
        z_index=z,
        layer_name=labels_layer.name,
        note=(
            f'{mode_note}; clicked zyx={coord_zyx}; '
            f'selected_query={result.get("selected_query")}; '
            f'click_prob={result.get("click_prob")}; '
            f'bbox=(z:{z0}-{z1}, y:{y0}-{y1}, x:{x0}-{x1}); '
            f'old_voxels={old_voxels}, new_voxels={new_voxels}; '
            f'time={result.get("clicked_infer_time_sec")} sec.'
        ),
    )

    self._local_refine_busy = False

    if getattr(self, '_local_refine_mode_active', False):
        self.btn_enter_local_refinement.setEnabled(False)
        self.btn_exit_local_refinement.setEnabled(True)
        if hasattr(self, 'local_refine_status_label'):
            self.local_refine_status_label.setText(
                'Active: click another cell'
            )
            self.local_refine_status_label.setStyleSheet('color: #66cc66;')
    else:
        self.btn_enter_local_refinement.setEnabled(True)
        self.btn_exit_local_refinement.setEnabled(False)

    notifications.show_info(
        f'one-click segmentation updated label {target_label}: '
        f'{old_voxels} → {new_voxels} voxels. Ctrl+Z to undo.'
    )


def _get_label_bbox_near_click_for_local_refine(
    self,
    label_data,
    label_val,
    coord_zyx,
    z_radius=25,
):
    """
    Find bbox of one label near clicked z.

    This avoids global operations such as:
        label_data == label_val

    because zarr.Array does not support global comparison operators.
    """
    Z, Y, X = label_data.shape
    cz, cy, cx = [int(v) for v in coord_zyx]
    label_val = int(label_val)

    z_start = max(0, cz - int(z_radius))
    z_end = min(Z, cz + int(z_radius) + 1)

    z_hits = []
    y0_list = []
    y1_list = []
    x0_list = []
    x1_list = []

    for zz in range(z_start, z_end):
        sl = np.asarray(label_data[zz])

        yy, xx = np.where(sl == label_val)
        if len(yy) == 0:
            continue

        z_hits.append(zz)
        y0_list.append(int(yy.min()))
        y1_list.append(int(yy.max()) + 1)
        x0_list.append(int(xx.min()))
        x1_list.append(int(xx.max()) + 1)

    if len(z_hits) == 0:
        return None

    return (
        int(min(z_hits)),
        int(max(z_hits)) + 1,
        int(min(y0_list)),
        int(max(y1_list)),
        int(min(x0_list)),
        int(max(x1_list)),
    )


def _on_clicked_local_refinement_error(self, error_msg):
    """
    Handle clicked one-click segmentation failure.
    """
    if (
        hasattr(self, 'local_refine_progress')
        and self.local_refine_progress is not None
    ):
        self.local_refine_progress.close()

    self._local_refine_busy = False

    if getattr(self, '_local_refine_mode_active', False):
        if hasattr(self, 'btn_enter_local_refinement'):
            self.btn_enter_local_refinement.setEnabled(False)
        if hasattr(self, 'btn_exit_local_refinement'):
            self.btn_exit_local_refinement.setEnabled(True)
        if hasattr(self, 'local_refine_status_label'):
            self.local_refine_status_label.setText(
                'Active: click another cell'
            )
            self.local_refine_status_label.setStyleSheet('color: #66cc66;')
    else:
        if hasattr(self, 'btn_enter_local_refinement'):
            self.btn_enter_local_refinement.setEnabled(True)
        if hasattr(self, 'btn_exit_local_refinement'):
            self.btn_exit_local_refinement.setEnabled(False)

    notifications.show_error(
        f'Clicked one-click segmentation failed: {error_msg}'
    )


def _install_local_refinement_callbacks(self):
    """
    Install mouse callback on current image and label layers.

    Add the callback to both layer types so the click still works whether
    the user has selected the image layer or the labels layer.
    """
    for layer in self.viewer.layers:
        if isinstance(layer, (Image, Labels)):
            if (
                self._on_local_refinement_click
                not in layer.mouse_drag_callbacks
            ):
                layer.mouse_drag_callbacks.append(
                    self._on_local_refinement_click
                )


def _remove_local_refinement_callbacks(self):
    """
    Remove one-click segmentation mouse callbacks from all layers.
    """
    for layer in list(self.viewer.layers):
        if not hasattr(layer, 'mouse_drag_callbacks'):
            continue

        with suppress(ValueError):
            layer.mouse_drag_callbacks.remove(self._on_local_refinement_click)


def _on_local_refinement_click(self, layer, event):
    """
    Mouse callback for one-click segmentation mode.
    """
    if not getattr(self, '_local_refine_mode_active', False):
        return

    if getattr(self, '_local_refine_busy', False):
        notifications.show_warning(
            'one-click segmentation is still running. Please wait for it to finish.'
        )
        return

    try:
        data_pos = layer.world_to_data(event.position)
        data_pos = np.asarray(data_pos, dtype=float)

        if data_pos.size >= 3:
            z, y, x = data_pos[-3:]
        elif data_pos.size == 2:
            z = int(self.viewer.dims.current_step[0])
            y, x = data_pos
        else:
            notifications.show_warning('Cannot parse clicked position.')
            return

        coord_zyx = (
            int(round(z)),
            int(round(y)),
            int(round(x)),
        )

        labels_layer = self._get_labels_layer()
        if labels_layer is None:
            notifications.show_error('No labels layer found.')
            return

        Z, Y, X = labels_layer.data.shape
        zz, yy, xx = coord_zyx

        if not (0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X):
            notifications.show_warning(
                f'Clicked point is outside data: {coord_zyx}'
            )
            return

        self._start_clicked_local_refinement(coord_zyx)

    except Exception as e:
        notifications.show_error(
            f'Failed to handle one-click segmentation click: {e}'
        )

    yield


def _update_curation_controls(self):
    """
    Enable/disable curation-related widgets based on whether curation mode is active.
    Also respects the current ROI/new-label mode.
    """
    # curation mode must be active to use any curation tool
    curation_active = self._pick_mode_active
    in_session = (
        self._current_roi_mode is not None
        or self._new_label_mode_active
        or self._delete_inside_mode_active
        or self._delete_all_active
        or getattr(self, '_cell_refine_active', False)
    )
    # Buttons that are always disabled when curation mode is off
    self.btn_roi_add.setEnabled(curation_active)
    self.btn_roi_subtract.setEnabled(curation_active)
    self.btn_add_new_label.setEnabled(curation_active)
    self.btn_delete_slice.setEnabled(curation_active and not in_session)
    self.btn_delete_all.setEnabled(curation_active and not in_session)
    self.btn_change_label.setEnabled(curation_active and not in_session)
    self.btn_delete_inside.setEnabled(curation_active and not in_session)
    self.btn_export_log.setEnabled(curation_active)
    # Apply/Cancel are only enabled during an active ROI or new-label session,
    # and only if curation mode is active
    if curation_active and in_session:
        # They will be enabled/disabled by the ROI/new-label entry/exit methods
        # (their state is managed separately, so we don't override here)
        pass
    else:
        self.btn_apply_roi.setEnabled(False)
        self.btn_cancel_roi.setEnabled(False)

    # Draw mode radio buttons are only enabled if no ROI/new-label session is active
    # and curation mode is active

    self.roi_polygon_radio.setEnabled(curation_active and not in_session)
    self.roi_brush_radio.setEnabled(curation_active and not in_session)

    if hasattr(self, 'roi_trace_radio'):
        self.roi_trace_radio.setEnabled(curation_active and not in_session)

    # Brush size widget is only visible when brush mode is selected and curation active
    if curation_active:
        self._on_roi_draw_mode_changed()  # this handles visibility of brush_size_widget
    else:
        self.brush_size_widget.setVisible(False)

    active_labels = self._get_labels_layer()
    if active_labels:
        self.chk_contour.blockSignals(False)
        self.chk_contour.setChecked(active_labels.contour)
        self.chk_contour.blockSignals(False)

    # one-click segmentation is in Segmentation tab and does not require curation mode.
    if hasattr(self, 'btn_enter_local_refinement'):
        local_active = getattr(self, '_local_refine_mode_active', False)
        local_busy = getattr(self, '_local_refine_busy', False)
        local_loading = getattr(self, '_local_refine_loading', False)

        self.btn_enter_local_refinement.setEnabled(
            (not local_active) and (not local_busy) and (not local_loading)
        )

    if hasattr(self, 'btn_exit_local_refinement'):
        local_active = getattr(self, '_local_refine_mode_active', False)
        local_busy = getattr(self, '_local_refine_busy', False)
        self.btn_exit_local_refinement.setEnabled(
            local_active and (not local_busy) and (not local_loading)
        )


def _on_roi_draw_mode_changed(self):
    """Show brush size widget only when brush mode is selected."""
    if self._new_label_mode_active:
        return

    use_brush = self.roi_brush_radio.isChecked()
    self.brush_size_widget.setVisible(use_brush)


def _on_roi_brush_size_changed(self, value):
    self._roi_brush_size = int(value)

    # ROI brush layer
    if self._roi_brush_layer is not None:
        self._roi_brush_layer.brush_size = int(value)
        self._roi_brush_layer.refresh()

    # active labels layer
    active_layer = self.viewer.layers.selection.active
    if isinstance(active_layer, Labels):
        mode = str(active_layer.mode).lower()
        mode = mode.split('.')[-1]

        if mode in {'pick', 'paint', 'erase'}:
            active_layer.brush_size = int(value)
            active_layer.refresh()


def _change_selected_label_id(self):
    """
    Change ID of selected label on current Z slice only.
    1. Get selected label from labels layer
    2. Show input dialog for new ID
    3. Replace old ID with new ID on current slice
    """
    # Get active labels layer (reuse existing method)
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    # Get current selected label (source ID) - skip background (0)
    source_label = labels_layer.selected_label
    if source_label == 0:
        notifications.show_warning('Cannot modify background label (0)!')
        return

    # Get current Z slice index (Z is first axis in 3D data)
    current_z = self.viewer.dims.current_step[0]

    # Show input dialog for new label ID
    new_label, ok = QInputDialog.getInt(
        self,
        'Change Label ID',
        f'Current selected label: {source_label}\nCurrent slice: {current_z}\nEnter new label ID:',
        min=1,  # New ID can't be background (0)
        max=65535,  # Max value for uint16 (common label dtype)
        value=source_label + 1,  # Default to next available ID
    )

    # Exit if user cancels dialog
    if not ok:
        return

    # Skip if new ID is same as original
    if new_label == source_label:
        notifications.show_info(
            f'New ID ({new_label}) matches original - no changes made.'
        )
        return

    # Get current slice data (supports numpy/zarr arrays)
    slice_data = labels_layer.data[current_z]

    # Validate data type and replace label IDs
    if isinstance(slice_data, np.ndarray):
        # Create mask for pixels with source label
        mask = slice_data == source_label

        # Check if source label exists on current slice
        if not np.any(mask):
            notifications.show_info(
                f'Label {source_label} not found on slice {current_z}!'
            )
            return

        # Replace source label with new ID
        slice_data[mask] = new_label

        # Update the slice in labels layer (critical for zarr compatibility)
        labels_layer.data[current_z] = slice_data
        labels_layer.refresh()  # Refresh viewer display

        notifications.show_info(
            f'Success! Label {source_label} → {new_label} (slice {current_z}).'
        )
    else:
        notifications.show_error(
            f'Unsupported data type: {type(slice_data)} (only numpy/zarr supported).'
        )


def _delete_selected_label_slice(self):
    """
    Delete the currently selected label on the current Z slice only.

    Optimized:
    - Scan current slice once to find the selected label.
    - Then write back only the tight bounding-box region containing that label.
    """
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    label_val = labels_layer.selected_label
    if label_val == 0:
        notifications.show_warning('Cannot delete background (label 0).')
        return

    z_idx = int(self.viewer.dims.current_step[0])
    data = labels_layer.data

    # Need one scan of the current slice to locate the selected label.
    # This is unavoidable unless we maintain a label_id -> bbox index.
    slice_data = np.asarray(data[z_idx])
    yy, xx = np.where(slice_data == label_val)

    if len(yy) == 0:
        notifications.show_info(
            f'Label {label_val} not found on slice {z_idx}.'
        )
        return

    y0 = int(yy.min())
    y1 = int(yy.max()) + 1
    x0 = int(xx.min())
    x1 = int(xx.max()) + 1

    # Read/write only the bbox region.
    region = np.asarray(data[z_idx, y0:y1, x0:x1]).copy()
    region[region == label_val] = 0
    data[z_idx, y0:y1, x0:x1] = region

    labels_layer.refresh()

    self._append_log_entry(
        operation='delete current Z',
        label_id_or_count=label_val,
        z_index=z_idx,
        note=(
            f'Label {label_val} deleted on current slice. '
            f'bbox=(y:{y0}-{y1}, x:{x0}-{x1}).'
        ),
    )

    notifications.show_info(f'Label {label_val} deleted on slice {z_idx}.')


def _on_delete_error(self, error_msg):
    """Handle error from any delete worker."""
    notifications.show_error(f'Delete failed: {error_msg}')

    # Close progress dialog if it exists
    if hasattr(self, 'delete_progress'):
        self.delete_progress.close()

    # Reset any active delete flags and restore UI
    if self._delete_all_active:
        self._finalize_delete_all()
    if self._delete_inside_mode_active:
        self._finalize_delete_inside()
    self._update_curation_controls()
    if hasattr(self, 'delete_all_thread'):
        self.delete_all_thread.quit()
        self.delete_all_thread.wait()
    if hasattr(self, 'delete_inside_thread'):
        self.delete_inside_thread.quit()
        self.delete_inside_thread.wait()


def _delete_selected_label_all(self):
    """Delete the currently selected label on all Z slices (background)."""
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return
    label_val = labels_layer.selected_label
    if label_val == 0:
        notifications.show_warning('Cannot delete background (label 0).')
        return
    data = labels_layer.data
    current_z = self.viewer.dims.current_step[0]
    self._delete_all_active = True
    self._update_curation_controls()  # This will disable relevant buttons

    self.delete_all_thread = QThread()
    self.delete_all_worker = DeleteAllWorker(data, label_val, current_z)
    self.delete_all_worker.moveToThread(self.delete_all_thread)

    self.delete_all_progress = QProgressDialog(
        'Deleting label from all slices...', 'Cancel', 0, 100, self
    )
    self.delete_all_progress.setWindowTitle('Delete All Progress')
    self.delete_all_progress.canceled.connect(
        self.delete_all_thread.requestInterruption
    )

    self.delete_all_worker.progress.connect(self.delete_all_progress.setValue)
    self.delete_all_worker.progress.connect(
        lambda v, msg: self.delete_all_progress.setLabelText(msg)
    )
    self.delete_all_worker.finished.connect(self._on_delete_all_finished)
    self.delete_all_worker.error.connect(self._on_delete_error)
    self.delete_all_thread.started.connect(self.delete_all_worker.run)

    self.delete_all_thread.finished.connect(
        lambda: self._finalize_delete_all()
    )
    self.delete_all_worker.finished.connect(self.delete_all_thread.quit)
    self.delete_all_worker.finished.connect(self.delete_all_worker.deleteLater)
    self.delete_all_thread.finished.connect(self.delete_all_thread.deleteLater)
    self.delete_all_thread.finished.connect(self.delete_all_progress.close)

    self.delete_all_thread.start()
    self.delete_all_progress.show()


def _on_delete_all_finished(self, label_val):
    labels_layer = self._get_labels_layer()
    if labels_layer:
        labels_layer.refresh()
        current_z = self.viewer.dims.current_step[0]
        self._append_log_entry(
            operation='delete all Z',
            label_id_or_count=label_val,
            z_index=current_z,
            note='Deleted label in nearby slices around current Z.',
        )
        notifications.show_info(
            f'Label {label_val} nearby slices (current ±30).'
        )
    else:
        notifications.show_info(
            f'Label {label_val} not found in nearby slices.'
        )


# ---------- ROI drawing and application ----------
def _enter_roi_mode(self, mode: str, target_label=None):
    """
    Enter ROI drawing mode for adding or subtracting.

    Parameters
    ----------
    mode : 'add' or 'subtract'
    target_label : int, optional
        The label that will be modified. If not provided, the currently
        selected label in the labels layer is used.
    """
    # Check if a labels layer exists
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    # Determine the target label
    if target_label is None:
        self._roi_target_label = None
        if labels_layer.selected_label == 0:
            notifications.show_warning(
                'Please select a non-background label first.'
            )
            return
    else:
        # Ensure target_label is not background
        if target_label == 0:
            notifications.show_error('Target label cannot be background.')
            return
        # Store target label for use in _apply_roi
        self._roi_target_label = target_label

    # Remove any existing ROI layer
    self._cancel_roi()

    # Determine drawing mode and show/hide brush size accordingly
    if self.roi_polygon_radio.isChecked():
        self._roi_layer = self.viewer.add_shapes(
            name='ROI (draw area)',
            shape_type='polygon',
            edge_color='red',
            face_color='transparent',
            opacity=0.8,
            ndim=3,
        )
        self.viewer.layers.selection.active = self._roi_layer
        self._roi_layer.mode = 'add_polygon'

    elif hasattr(self, 'roi_trace_radio') and self.roi_trace_radio.isChecked():
        self._roi_layer = self.viewer.add_shapes(
            name='ROI (trace boundary)',
            shape_type='path',
            edge_color='yellow',
            face_color='transparent',
            opacity=0.9,
            ndim=3,
        )
        self.viewer.layers.selection.active = self._roi_layer

        try:
            self._roi_layer.mode = 'add_path'
        except Exception:
            self._roi_layer.mode = 'add_polygon'

    else:
        z_idx = int(self.viewer.dims.current_step[0])
        slice_shape = labels_layer.data.shape[1:3]

        self._roi_brush_z = z_idx
        self._roi_brush_masks = {}

        self._roi_brush_layer = self.viewer.add_labels(
            np.zeros(slice_shape, dtype=np.uint8),
            name='New Label (brush here)',
            opacity=0.5,
        )
        self._roi_brush_layer.colormap = {1: self._roi_brush_color}
        self._roi_brush_layer.brush_size = self._roi_brush_size
        self._roi_brush_layer.mode = 'paint'
        self._roi_brush_layer.selected_label = 1
        self.viewer.layers.selection.active = self._roi_brush_layer

    self._current_roi_mode = mode
    self.btn_apply_roi.setEnabled(True)
    self.btn_cancel_roi.setEnabled(True)
    self.btn_roi_add.setEnabled(False)
    self.btn_roi_subtract.setEnabled(False)
    self.btn_add_new_label.setEnabled(False)
    self.btn_delete_slice.setEnabled(False)
    self.btn_delete_all.setEnabled(False)
    # Also disable draw mode radios (cannot switch during session)
    self.roi_polygon_radio.setEnabled(False)
    self.roi_brush_radio.setEnabled(False)
    if hasattr(self, 'roi_trace_radio'):
        self.roi_trace_radio.setEnabled(False)
    mode_str = 'ADD to' if mode == 'add' else 'SUBTRACT from'
    notifications.show_info(
        f"{mode_str} label {self._roi_target_label}. Draw. Click 'Apply' to execute, 'Cancel' to abort."
    )


def _apply_roi(self):
    """
    Apply ROI to add/subtract pixels for the target label on the current Z slice.

    Optimized:
    - Only read/write the bounding box of the drawn ROI.
    - Avoid modifying the whole slice when the ROI is small.
    - Works for both polygon mode and brush mode.
    """
    if self._new_label_mode_active:
        self._apply_new_label()
        return

    if self._delete_inside_mode_active:
        self._apply_delete_inside()
        return

    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        self._cancel_roi()
        return

    if self._roi_target_label is None:
        target_label = labels_layer.selected_label
        if target_label == 0:
            notifications.show_error('No valid target label selected.')
            self._cancel_roi()
            return
    else:
        target_label = self._roi_target_label

    if target_label is None or target_label == 0:
        notifications.show_error('No valid target label set. Cancelling.')
        self._cancel_roi()
        return

    data = labels_layer.data
    z_idx = int(self.viewer.dims.current_step[0])
    slice_shape = data.shape[1:3]  # (Y, X)

    use_polygon = self.roi_polygon_radio.isChecked() or (
        hasattr(self, 'roi_trace_radio') and self.roi_trace_radio.isChecked()
    )

    # ------------------------------------------------------------
    # 1. Build ROI mask and bbox
    # ------------------------------------------------------------
    if use_polygon:
        if (
            self._roi_layer is None
            or self._roi_layer not in self.viewer.layers
        ):
            notifications.show_error('No ROI layer found.')
            self._cancel_roi()
            return

        shapes_data = self._roi_layer.data
        if len(shapes_data) == 0:
            notifications.show_warning('No polygons drawn.')
            self._cancel_roi()
            return

        combined_mask = np.zeros(slice_shape, dtype=bool)

        for polygon in shapes_data:
            polygon = np.asarray(polygon)

            if polygon.ndim != 2 or polygon.shape[0] < 3:
                continue

            # Robustly support both 2D polygon coordinates (y, x)
            # and 3D polygon coordinates (z, y, x).
            if polygon.shape[1] >= 3:
                poly_yx = polygon[:, 1:3]
            else:
                poly_yx = polygon[:, :2]

            poly_yx = _smooth_closed_polygon_yx(
                poly_yx,
                samples_per_segment=16,
            )

            rr, cc = draw_polygon(
                poly_yx[:, 0],
                poly_yx[:, 1],
                slice_shape,
            )
            combined_mask[rr, cc] = True

        if not np.any(combined_mask):
            notifications.show_warning('No pixels inside ROI.')
            self._cancel_roi(labels_layer_to_activate=labels_layer)
            return

        yy, xx = np.where(combined_mask)

    else:
        if (
            self._roi_brush_layer is None
            or self._roi_brush_layer not in self.viewer.layers
        ):
            notifications.show_error('No ROI brush layer found.')
            self._cancel_roi()
            return

        brush_data = np.asarray(self._roi_brush_layer.data)
        yy, xx = np.where(brush_data == 1)

        if len(yy) == 0:
            notifications.show_warning('No brush strokes drawn.')
            self._cancel_roi()
            return

        z_idx = int(self._roi_brush_z)

    # ------------------------------------------------------------
    # 2. Convert ROI pixels to a tight bounding box
    # ------------------------------------------------------------
    y0 = int(yy.min())
    y1 = int(yy.max()) + 1
    x0 = int(xx.min())
    x1 = int(xx.max()) + 1

    # Safety clipping.
    y0 = max(0, min(y0, slice_shape[0]))
    y1 = max(0, min(y1, slice_shape[0]))
    x0 = max(0, min(x0, slice_shape[1]))
    x1 = max(0, min(x1, slice_shape[1]))

    if y1 <= y0 or x1 <= x0:
        notifications.show_warning('Invalid ROI bounding box.')
        self._cancel_roi(labels_layer_to_activate=labels_layer)
        return

    # Build only local mask_crop instead of using the whole slice.
    if use_polygon:
        mask_crop = combined_mask[y0:y1, x0:x1]
    else:
        # Shift global brush coordinates into local bbox coordinates.
        mask_crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        mask_crop[yy - y0, xx - x0] = True

    if not np.any(mask_crop):
        notifications.show_warning('No pixels inside ROI crop.')
        self._cancel_roi(labels_layer_to_activate=labels_layer)
        return

    # ------------------------------------------------------------
    # 3. Read/write only this local region
    # ------------------------------------------------------------
    region = np.asarray(data[z_idx, y0:y1, x0:x1]).copy()

    if self._current_roi_mode == 'add':
        region[mask_crop] = target_label
        data[z_idx, y0:y1, x0:x1] = region

        self._append_log_entry(
            operation='add to label',
            label_id_or_count=target_label,
            z_index=z_idx,
            note=(
                f'Added ROI region to label {target_label}. '
                f'bbox=(y:{y0}-{y1}, x:{x0}-{x1}).'
            ),
        )
        notifications.show_info(
            f'Added label {target_label} inside ROI on slice {z_idx}.'
        )

    elif self._current_roi_mode == 'subtract':
        # Only remove target_label pixels inside the ROI.
        target_pixels = mask_crop & (region == target_label)

        if np.any(target_pixels):
            region[target_pixels] = 0
            data[z_idx, y0:y1, x0:x1] = region

        self._append_log_entry(
            operation='subtract from label',
            label_id_or_count=target_label,
            z_index=z_idx,
            note=(
                f'Subtracted ROI region from label {target_label}. '
                f'bbox=(y:{y0}-{y1}, x:{x0}-{x1}).'
            ),
        )
        notifications.show_info(
            f'Subtracted label {target_label} inside ROI on slice {z_idx}.'
        )

    else:
        notifications.show_error('Unknown ROI mode.')
        self._cancel_roi()
        return

    labels_layer.refresh()
    self._cancel_roi(labels_layer_to_activate=labels_layer)


def _apply_new_label(self):
    """
    Apply accumulated drawings to create a new label.

    Optimized:
    - Do not replace the whole labels_layer.data.
    - For brush mode, write only the painted bounding-box region.
    - For polygon mode, still rasterize per slice, but avoid full layer reassignment.
    """
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        self._cancel_roi()
        return

    target_label = self._new_label_target
    if target_label is None:
        notifications.show_error('No target label. Cancelling.')
        self._cancel_roi()
        return

    data = labels_layer.data
    z_max = data.shape[0] - 1
    slice_shape = data.shape[1:3]
    current_z = int(self.viewer.dims.current_step[0])

    modified_slices = set()

    use_shape_roi = self.roi_polygon_radio.isChecked() or (
        hasattr(self, 'roi_trace_radio') and self.roi_trace_radio.isChecked()
    )

    if use_shape_roi:
        if (
            self._roi_layer is None
            or self._roi_layer not in self.viewer.layers
        ):
            notifications.show_error('ROI shape layer missing.')
            self._cancel_roi()
            return

        shapes = self._roi_layer.data
        if len(shapes) == 0:
            notifications.show_warning('No ROI drawn. Nothing to apply.')
            self._cancel_roi()
            return

        slice_masks = self._rasterize_shapes_to_slice_masks(
            shapes_data=shapes,
            slice_shape=slice_shape,
            default_z=current_z,
            z_max=z_max,
        )

        if not slice_masks:
            notifications.show_warning('No valid ROI drawn. Nothing to apply.')
            self._cancel_roi()
            return

        for z, mask in slice_masks.items():
            if not np.any(mask):
                continue

            z = int(z)
            slice_data = np.asarray(data[z]).copy()
            slice_data[mask] = target_label
            data[z] = slice_data
            modified_slices.add(z)
    else:
        if (
            self._roi_brush_layer is None
            or self._roi_brush_layer not in self.viewer.layers
        ):
            notifications.show_error('ROI brush layer missing.')
            self._cancel_roi()
            return

        # Save current visible brush slice before applying.
        self._save_current_brush_slice()

        if not self._roi_brush_masks:
            notifications.show_warning(
                'No brush strokes drawn. Nothing to apply.'
            )
            self._cancel_roi()
            return

        for z, saved in self._roi_brush_masks.items():
            z = int(z)

            if isinstance(saved, dict):
                y0, y1, x0, x1 = saved['bbox']
                mask_crop = saved['mask']

                if mask_crop is None or not np.any(mask_crop):
                    continue

                # Read only the local bbox region.
                region = np.asarray(data[z, y0:y1, x0:x1]).copy()
                region[mask_crop] = target_label
                data[z, y0:y1, x0:x1] = region

            else:
                # Backward compatibility if old full-size masks still exist.
                slice_mask = saved
                if not np.any(slice_mask):
                    continue
                slice_data = np.asarray(data[z]).copy()
                slice_data[slice_mask] = target_label
                data[z] = slice_data

            modified_slices.add(z)

    self._append_log_entry(
        operation='add new label',
        label_id_or_count=target_label,
        z_index=current_z,
        note=f'Created new label {target_label}. Modified slices: {sorted(modified_slices)}.',
    )

    labels_layer.selected_label = target_label

    # Important:
    # Do NOT call self._modify_labels(labels_layer, data), because that resets the whole layer.
    labels_layer.refresh()

    notifications.show_info(
        f'New label {target_label} added on {len(modified_slices)} slice(s).'
    )

    self._exit_new_label_mode(labels_layer)


def _save_current_brush_slice(self):
    """
    Save current 2D brush layer into self._roi_brush_masks[current_z].

    Optimized:
    - Store only the bounding-box crop of the painted mask.
    - Avoid keeping a full-size (Y, X) bool mask for every edited Z slice.
    """
    if self._roi_brush_layer is None:
        return
    if self._roi_brush_z is None:
        return
    if self._roi_brush_layer not in self.viewer.layers:
        return

    mask = np.asarray(self._roi_brush_layer.data) == 1
    z = int(self._roi_brush_z)

    if not np.any(mask):
        self._roi_brush_masks.pop(z, None)
        return

    yy, xx = np.where(mask)
    y0 = int(yy.min())
    y1 = int(yy.max()) + 1
    x0 = int(xx.min())
    x1 = int(xx.max()) + 1

    mask_crop = mask[y0:y1, x0:x1].copy()

    # Store compact representation: bbox + cropped mask.
    self._roi_brush_masks[z] = {
        'bbox': (y0, y1, x0, x1),
        'mask': mask_crop,
    }


def _on_brush_z_changed(self, event=None):
    """
    When changing Z during Add New Label brush mode,
    save the old slice and load the brush mask of the new slice.

    Optimized:
    - Internally stores brush masks as bbox crops.
    - Only expands to full 2D uint8 mask for the temporary napari brush layer.
    """
    if not self._new_label_mode_active:
        return
    if self._roi_brush_layer is None:
        return
    if self._roi_brush_layer not in self.viewer.layers:
        return
    if not self.roi_brush_radio.isChecked():
        return

    # Save old Z mask in compact form.
    self._save_current_brush_slice()

    # Switch to new Z.
    new_z = int(self.viewer.dims.current_step[0])
    self._roi_brush_z = new_z

    slice_shape = self._roi_brush_layer.data.shape
    new_data = np.zeros(slice_shape, dtype=np.uint8)

    saved = self._roi_brush_masks.get(new_z, None)
    if saved is not None:
        y0, y1, x0, x1 = saved['bbox']
        mask_crop = saved['mask']
        new_data[y0:y1, x0:x1][mask_crop] = 1

    self._roi_brush_layer.data = new_data
    self._roi_brush_layer.mode = 'paint'
    self._roi_brush_layer.selected_label = 1
    self._roi_brush_layer.refresh()


def _cancel_roi(self, labels_layer_to_activate=None):
    """Remove the ROI layer(s) and reset buttons."""
    # Remove polygon layer if exists
    if self._delete_inside_mode_active:
        self._delete_inside_mode_active = False

    if self._new_label_mode_active:
        self._exit_new_label_mode(labels_layer_to_activate)
        return
    if self._roi_layer is not None and self._roi_layer in self.viewer.layers:
        self.viewer.layers.remove(self._roi_layer)
    self._roi_layer = None
    # Remove brush layer if exists
    if (
        self._roi_brush_layer is not None
        and self._roi_brush_layer in self.viewer.layers
    ):
        self.viewer.layers.remove(self._roi_brush_layer)
    self._roi_brush_layer = None
    self._roi_brush_z = None
    self._current_roi_mode = None
    self._roi_target_label = None
    if self._delete_all_active:
        self._delete_all_active = False
    self._update_curation_controls()
    # Activate the specified labels layer, or any labels layer if none specified
    # Restore label layer selection after cancelling ROI.
    labels_layer = None

    if labels_layer_to_activate is not None:
        if isinstance(labels_layer_to_activate, Labels):
            labels_layer = labels_layer_to_activate
        else:
            print(
                f'Warning: Attempted to activate non-label layer: {labels_layer_to_activate}'
            )

    if hasattr(self, '_activate_labels_layer'):
        self._activate_labels_layer(labels_layer, show_warning=False)
    else:
        if labels_layer is not None:
            self.viewer.layers.selection.active = labels_layer
        else:
            for layer in self.viewer.layers:
                if isinstance(layer, Labels):
                    self.viewer.layers.selection.active = layer
                    break


def _exit_new_label_mode(self, labels_layer_to_activate=None):
    """Clean up after finishing (apply or cancel) a new‑label drawing session."""
    # Remove temporary layers
    if self._roi_layer is not None and self._roi_layer in self.viewer.layers:
        self.viewer.layers.remove(self._roi_layer)
    self._roi_layer = None
    if (
        self._roi_brush_layer is not None
        and self._roi_brush_layer in self.viewer.layers
    ):
        self.viewer.layers.remove(self._roi_brush_layer)
    self._roi_brush_layer = None

    # Reset state flags
    self._new_label_mode_active = False
    self._new_label_target = None

    # Re‑enable all buttons (but respect curation mode via _update_curation_controls)
    self._update_curation_controls()

    # Re‑enable all buttons that were disabled
    self.btn_roi_add.setEnabled(True)
    self.btn_roi_subtract.setEnabled(True)
    self.btn_add_new_label.setEnabled(True)
    self.btn_delete_slice.setEnabled(True)
    self.btn_delete_all.setEnabled(True)
    self.btn_save.setEnabled(True)
    self.roi_polygon_radio.setEnabled(True)
    self.roi_brush_radio.setEnabled(True)
    if hasattr(self, 'roi_trace_radio'):
        self.roi_trace_radio.setEnabled(True)
    # Disable Apply/Cancel (they are only active during ROI modes)
    self.btn_apply_roi.setEnabled(False)
    self.btn_cancel_roi.setEnabled(False)
    self._roi_brush_z = None
    self._roi_brush_masks = {}
    # Restore label layer selection after exiting new-label mode.
    labels_layer = None

    if labels_layer_to_activate is not None:
        if isinstance(labels_layer_to_activate, Labels):
            labels_layer = labels_layer_to_activate
        else:
            print(
                f'Warning: Attempted to activate non-label layer: {labels_layer_to_activate}'
            )

    if hasattr(self, '_activate_labels_layer'):
        self._activate_labels_layer(labels_layer, show_warning=False)
    else:
        if labels_layer is not None:
            self.viewer.layers.selection.active = labels_layer
        else:
            for layer in self.viewer.layers:
                if isinstance(layer, Labels):
                    self.viewer.layers.selection.active = layer
                    break


def _enter_delete_inside_roi_mode(self):
    """Enter a mode to draw a polygon that will be used to delete all labels inside it across all Z slices."""
    # Check if a labels layer exists
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    # Remove any existing ROI layer
    self._cancel_roi()

    # Create a shapes layer for polygon drawing (force polygon mode)
    self._roi_layer = self.viewer.add_shapes(
        name='Delete ROI (draw polygon)',
        shape_type='polygon',
        edge_color='red',
        face_color='transparent',
        opacity=0.8,
    )
    self.viewer.layers.selection.active = self._roi_layer
    self._roi_layer.mode = 'add_polygon'

    # Set mode flag
    self._delete_inside_mode_active = True

    # Disable other curation buttons
    self.btn_roi_add.setEnabled(False)
    self.btn_roi_subtract.setEnabled(False)
    self.btn_add_new_label.setEnabled(False)
    self.btn_delete_slice.setEnabled(False)
    self.btn_delete_all.setEnabled(False)
    self.btn_delete_inside.setEnabled(False)  # disable itself during session
    self.btn_change_label.setEnabled(False)
    self.roi_polygon_radio.setEnabled(False)
    self.roi_brush_radio.setEnabled(False)

    # Enable Apply/Cancel
    self.btn_apply_roi.setEnabled(True)
    self.btn_cancel_roi.setEnabled(True)

    notifications.show_info(
        'Draw a polygon on the current slice. Click Apply to delete all labels inside this polygon on ALL slices.'
    )


def _apply_delete_inside(self):
    """Delete entire cells intersecting the drawn polygon (background)."""
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        self._cancel_roi()
        return

    if self._roi_layer is None or self._roi_layer not in self.viewer.layers:
        notifications.show_error('No ROI polygon layer found.')
        self._cancel_roi()
        return

    shapes_data = self._roi_layer.data.copy()  # copy to avoid reference issues
    if len(shapes_data) == 0:
        notifications.show_warning('No polygons drawn.')
        self._cancel_roi()
        return

    current_z = self.viewer.dims.current_step[0]
    data = labels_layer.data

    # Manually remove ROI layers WITHOUT resetting the mode flag
    if self._roi_layer is not None and self._roi_layer in self.viewer.layers:
        self.viewer.layers.remove(self._roi_layer)
        self._roi_layer = None
    if (
        self._roi_brush_layer is not None
        and self._roi_brush_layer in self.viewer.layers
    ):
        self.viewer.layers.remove(self._roi_brush_layer)
        self._roi_brush_layer = None

    # Disable Apply/Cancel buttons immediately (they are no longer needed)
    self.btn_apply_roi.setEnabled(False)
    self.btn_cancel_roi.setEnabled(False)

    # Create thread and worker
    self.delete_inside_thread = QThread()
    self.delete_inside_worker = DeleteInsideWorker(
        data, shapes_data, current_z
    )
    self.delete_inside_worker.moveToThread(self.delete_inside_thread)

    # Progress dialog
    self.delete_inside_progress = QProgressDialog(
        'Deleting inside ROI...', 'Cancel', 0, 100, self
    )
    self.delete_inside_progress.setWindowTitle('Delete Inside ROI Progress')
    self.delete_inside_progress.canceled.connect(
        self.delete_inside_thread.requestInterruption
    )

    self.delete_inside_worker.progress.connect(
        self.delete_inside_progress.setValue
    )
    self.delete_inside_worker.progress.connect(
        lambda v, msg: self.delete_inside_progress.setLabelText(msg)
    )
    self.delete_inside_worker.finished.connect(self._on_delete_inside_finished)
    self.delete_inside_worker.error.connect(self._on_delete_error)
    self.delete_inside_thread.started.connect(self.delete_inside_worker.run)

    self.delete_inside_thread.finished.connect(
        lambda: self._finalize_delete_inside()
    )
    self.delete_inside_worker.finished.connect(self.delete_inside_thread.quit)
    self.delete_inside_worker.finished.connect(
        self.delete_inside_worker.deleteLater
    )
    self.delete_inside_thread.finished.connect(
        self.delete_inside_thread.deleteLater
    )
    self.delete_inside_thread.finished.connect(
        self.delete_inside_progress.close
    )

    self.delete_inside_thread.start()
    self.delete_inside_progress.show()


def _on_delete_inside_finished(self, deleted_count):
    labels_layer = self._get_labels_layer()

    if labels_layer:
        labels_layer.refresh()

        if hasattr(self, '_activate_labels_layer'):
            self._activate_labels_layer(labels_layer, show_warning=False)
        else:
            self.viewer.layers.selection.active = labels_layer

        current_z = self.viewer.dims.current_step[0]
        self._append_log_entry(
            operation='delete inside roi + apply roi',
            label_id_or_count=deleted_count,
            z_index=current_z,
            note=f'Deleted {deleted_count} label(s) intersecting ROI.',
        )
        notifications.show_info(
            f'Deleted {deleted_count} label(s) that intersected the polygon.'
        )
    else:
        notifications.show_info(
            'No labels found inside polygon or operation cancelled.'
        )


def _add_new_label(self):
    """Enter a mode to draw a new label on any slice. Drawings accumulate until Apply/Cancel."""
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    try:
        new_label = self._consume_next_label_id(labels_layer)
    except Exception as e:
        notifications.show_error(f'Failed to get next label ID: {e}')
        return

    self._cancel_roi()

    # ------------------------------------------------------------
    # 1. Polygon mode
    # ------------------------------------------------------------
    if self.roi_polygon_radio.isChecked():
        self._roi_layer = self.viewer.add_shapes(
            name='New Label (draw polygons)',
            shape_type='polygon',
            edge_color='red',
            face_color='transparent',
            opacity=0.8,
            ndim=3,
        )
        self.viewer.layers.selection.active = self._roi_layer
        self._roi_layer.mode = 'add_polygon'

    # ------------------------------------------------------------
    # 2. Trace mode
    # ------------------------------------------------------------
    elif hasattr(self, 'roi_trace_radio') and self.roi_trace_radio.isChecked():
        self._roi_layer = self.viewer.add_shapes(
            name='New Label (trace boundary)',
            shape_type='path',
            edge_color='yellow',
            face_color='transparent',
            opacity=0.9,
            ndim=3,
        )
        self.viewer.layers.selection.active = self._roi_layer

        # Different napari versions may support different shape modes.
        # Try path first; fallback to polygon if path mode is unavailable.
        try:
            self._roi_layer.mode = 'add_path'
        except Exception:
            self._roi_layer.mode = 'add_polygon'

    # ------------------------------------------------------------
    # 3. Brush mode
    # ------------------------------------------------------------
    else:
        z_idx = int(self.viewer.dims.current_step[0])
        slice_shape = labels_layer.data.shape[1:3]

        self._roi_brush_z = z_idx
        self._roi_brush_masks = {}

        self._roi_brush_layer = self.viewer.add_labels(
            np.zeros(slice_shape, dtype=np.uint8),
            name='New Label (brush here)',
            opacity=0.5,
        )
        self._roi_brush_layer.colormap = {1: self._roi_brush_color}
        self._roi_brush_layer.brush_size = self._roi_brush_size
        self._roi_brush_layer.mode = 'paint'
        self._roi_brush_layer.selected_label = 1
        self.viewer.layers.selection.active = self._roi_brush_layer

    self._new_label_mode_active = True
    self._new_label_target = new_label

    self.btn_roi_add.setEnabled(False)
    self.btn_roi_subtract.setEnabled(False)
    self.btn_add_new_label.setEnabled(False)
    self.btn_delete_slice.setEnabled(False)
    self.btn_delete_all.setEnabled(False)
    self.btn_save.setEnabled(False)

    self.roi_polygon_radio.setEnabled(False)
    self.roi_brush_radio.setEnabled(False)
    if hasattr(self, 'roi_trace_radio'):
        self.roi_trace_radio.setEnabled(False)

    self.btn_apply_roi.setEnabled(True)
    self.btn_cancel_roi.setEnabled(True)

    notifications.show_info(
        f'Adding new label {new_label}. Draw on any slice. Click Apply to confirm, Cancel to abort.'
    )


def _choose_roi_brush_color(self):
    color = QColorDialog.getColor(parent=self)
    if not color.isValid():
        return

    self._roi_brush_color = color.name()  # e.g. '#ff0000'
    self._update_brush_color_button()

    if self._roi_brush_layer is not None:
        self._roi_brush_layer.colormap = {1: self._roi_brush_color}
        self._roi_brush_layer.refresh()


def _update_brush_color_button(self):
    self.btn_brush_color.setText(' ')
    self.btn_brush_color.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {self._roi_brush_color};
            border: 1px solid #888;
            min-width: 40px;
        }}
        """
    )


def _ensure_next_label_id(self, labels_layer):
    """Cache next available label id on the layer metadata."""
    if labels_layer is None:
        raise ValueError('No labels layer.')

    if 'next_label_id' not in labels_layer.metadata:
        data = labels_layer.data

        if isinstance(data, np.ndarray):
            max_label = int(np.max(data))

        elif isinstance(data, da.Array):
            max_label = int(da.max(data).compute())

        else:
            # zarr.Array or other array-like backend
            max_label = int(da.max(da.asarray(data)).compute())

        labels_layer.metadata['next_label_id'] = max_label + 1

    return int(labels_layer.metadata['next_label_id'])


def _consume_next_label_id(self, labels_layer):
    next_id = self._ensure_next_label_id(labels_layer)
    labels_layer.metadata['next_label_id'] = next_id + 1
    return next_id


def _rasterize_shapes_to_slice_masks(
    self,
    shapes_data,
    slice_shape,
    default_z=None,
    z_max=None,
):
    """Convert napari polygon shapes into {z: mask2d}."""

    if z_max is None:
        z_max = 0

    slice_masks = {}

    for shape in shapes_data:
        shape = np.asarray(shape)
        if shape.ndim != 2 or shape.shape[0] < 3:
            continue

        if shape.shape[1] >= 3:
            # napari 3D shapes: assume (z, y, x)
            z = int(np.clip(np.round(np.median(shape[:, 0])), 0, z_max))
            poly = shape[:, 1:3]
        else:
            if default_z is None:
                continue
            z = int(np.clip(default_z, 0, z_max))
            poly = shape[:, :2]

        poly = _smooth_closed_polygon_yx(
            poly,
            samples_per_segment=16,
        )

        rr, cc = draw_polygon(poly[:, 0], poly[:, 1], slice_shape)
        mask = np.zeros(slice_shape, dtype=bool)
        mask[rr, cc] = True

        if z in slice_masks:
            slice_masks[z] |= mask
        else:
            slice_masks[z] = mask

    return slice_masks


def _undo_local_refinement(self, viewer=None):
    """
    Undo the latest one-click segmentation operation.
    """
    if not hasattr(self, '_local_refine_undo_stack'):
        notifications.show_info('No one-click segmentation history.')
        return

    if len(self._local_refine_undo_stack) == 0:
        notifications.show_info('No one-click segmentation operation to undo.')
        return

    item = self._local_refine_undo_stack.pop()

    labels_layer = item['labels_layer']
    bbox = item['bbox']
    old_crop = item['old_crop']

    if labels_layer is None or labels_layer not in self.viewer.layers:
        notifications.show_warning(
            'The original labels layer no longer exists.'
        )
        return

    z0, z1, y0, y1, x0, x1 = bbox

    labels_layer.data[z0:z1, y0:y1, x0:x1] = old_crop
    labels_layer.refresh()

    notifications.show_info(
        f'Undo one-click segmentation: restored bbox z:{z0}-{z1}, y:{y0}-{y1}, x:{x0}-{x1}.'
    )
