from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import traceback
from contextlib import suppress
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from dask import compute, delayed
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from napari.layers import Image, Labels
from napari.utils import notifications
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage.measure import marching_cubes, regionprops


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


class ReconstructionWorker:
    """
    Worker logic for 3D reconstruction.

    This class is intended to be used as a QObject worker in a QThread.
    It only computes mesh data and never creates any GUI/OpenGL objects.
    """

    def __init__(self, labels_data, label_id, zratio):
        self.labels_data = labels_data
        self.label_id = int(label_id)
        self.zratio = float(zratio)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self._emit_progress(0, 'Locating selected label...')
            crop_result = self._prepare_binary_crop(
                self.labels_data, self.label_id
            )

            if self._is_cancelled:
                self._emit_error('Cancelled')
                return

            if crop_result is None:
                self._emit_finished(None)
                return

            crop_mask, origin = crop_result
            voxel_count = int(crop_mask.sum())

            if voxel_count == 0:
                self._emit_finished(None)
                return

            self._emit_progress(55, 'Extracting surface mesh...')
            result = self._build_mesh_result(
                crop_mask=crop_mask,
                origin=origin,
                voxel_count=voxel_count,
            )

            if self._is_cancelled:
                self._emit_error('Cancelled')
                return

            self._emit_progress(100, 'Done')
            self._emit_finished(result)

        except (ValueError, RuntimeError, OSError) as e:
            self._emit_error(str(e))

    def _emit_progress(self, value, message):
        if self.progress is not None:
            self.progress.emit(int(value), str(message))

    def _emit_finished(self, result):
        if self.finished is not None:
            self.finished.emit(result)

    def _emit_error(self, message):
        if self.error is not None:
            self.error.emit(str(message))

    def _prepare_binary_crop(self, data, label_id):
        """Dispatch crop preparation based on the input array type."""
        if isinstance(data, np.ndarray):
            return self._prepare_binary_crop_numpy(data, label_id)

        if isinstance(data, da.Array):
            return self._prepare_binary_crop_chunked(data, label_id)

        if hasattr(data, 'chunks') and hasattr(data, 'shape'):
            return self._prepare_binary_crop_chunked(data, label_id)

        if hasattr(data, '__array__'):
            return self._prepare_binary_crop_numpy(np.asarray(data), label_id)

        raise ValueError('Unsupported data type for 3D reconstruction')

    def _prepare_binary_crop_numpy(self, data, label_id):
        """Prepare a cropped binary mask for a NumPy array."""
        mask = data == label_id
        if not np.any(mask):
            return None

        objects = ndi.find_objects(mask.astype(np.uint8))
        if not objects or objects[0] is None:
            return None

        sl = objects[0]
        crop_mask = mask[sl]
        origin = (sl[0].start, sl[1].start, sl[2].start)
        return crop_mask, origin

    def _prepare_binary_crop_chunked(self, data, label_id):
        """
        Scan chunk-by-chunk to find the bounding box of the selected label,
        then load only the cropped region into memory.
        """
        shape = tuple(int(v) for v in data.shape)
        chunk_shape = self._normalize_chunk_shape(data, shape)

        z_starts = range(0, shape[0], chunk_shape[0])
        y_starts = range(0, shape[1], chunk_shape[1])
        x_starts = range(0, shape[2], chunk_shape[2])

        total_chunks = (
            len(range(0, shape[0], chunk_shape[0]))
            * len(range(0, shape[1], chunk_shape[1]))
            * len(range(0, shape[2], chunk_shape[2]))
        )

        min_z = min_y = min_x = None
        max_z = max_y = max_x = None
        processed = 0
        last_progress = -1

        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    if self._is_cancelled:
                        return None

                    z1 = min(z0 + chunk_shape[0], shape[0])
                    y1 = min(y0 + chunk_shape[1], shape[1])
                    x1 = min(x0 + chunk_shape[2], shape[2])

                    block = np.asarray(data[z0:z1, y0:y1, x0:x1])
                    local_mask = block == label_id

                    if np.any(local_mask):
                        zz = np.any(local_mask, axis=(1, 2))
                        yy = np.any(local_mask, axis=(0, 2))
                        xx = np.any(local_mask, axis=(0, 1))

                        local_min_z = z0 + int(np.argmax(zz))
                        local_max_z = z0 + int(
                            len(zz) - 1 - np.argmax(zz[::-1])
                        )
                        local_min_y = y0 + int(np.argmax(yy))
                        local_max_y = y0 + int(
                            len(yy) - 1 - np.argmax(yy[::-1])
                        )
                        local_min_x = x0 + int(np.argmax(xx))
                        local_max_x = x0 + int(
                            len(xx) - 1 - np.argmax(xx[::-1])
                        )

                        if min_z is None:
                            min_z, min_y, min_x = (
                                local_min_z,
                                local_min_y,
                                local_min_x,
                            )
                            max_z, max_y, max_x = (
                                local_max_z,
                                local_max_y,
                                local_max_x,
                            )
                        else:
                            min_z = min(min_z, local_min_z)
                            min_y = min(min_y, local_min_y)
                            min_x = min(min_x, local_min_x)
                            max_z = max(max_z, local_max_z)
                            max_y = max(max_y, local_max_y)
                            max_x = max(max_x, local_max_x)

                    processed += 1
                    progress = int(5 + 35 * processed / max(total_chunks, 1))
                    if progress != last_progress:
                        self._emit_progress(
                            progress,
                            f'Scanning chunks {processed}/{total_chunks}...',
                        )
                        last_progress = progress

        if min_z is None:
            return None

        if self._is_cancelled:
            return None

        self._emit_progress(45, 'Loading cropped label region...')

        crop = np.asarray(
            data[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]
        )
        crop_mask = crop == label_id
        origin = (min_z, min_y, min_x)
        return crop_mask, origin

    def _normalize_chunk_shape(self, data, shape):
        """Normalize chunk metadata into a 3-int tuple."""
        chunks = getattr(data, 'chunks', None)

        if chunks is None:
            return tuple(min(s, 64) for s in shape)

        normalized = []
        for dim_chunks, _ in zip(chunks, shape, strict=False):
            if isinstance(dim_chunks, tuple):
                normalized.append(int(dim_chunks[0]))
            else:
                normalized.append(int(dim_chunks))
        return tuple(
            max(1, min(c, s)) for c, s in zip(normalized, shape, strict=False)
        )

    def _build_mesh_result(self, crop_mask, origin, voxel_count):
        """
        Build a surface mesh using marching cubes.

        Returned vertices are in physical XYZ world coordinates:
        X = column
        Y = row
        Z = slice * zratio
        """
        vol = np.pad(
            np.ascontiguousarray(crop_mask.astype(np.uint8)),
            pad_width=1,
            mode='constant',
            constant_values=0,
        )

        spacing_zyx = np.array([self.zratio, 1.0, 1.0], dtype=np.float32)

        verts_zyx, faces, normals_zyx, _ = marching_cubes(
            vol,
            level=0.5,
            spacing=tuple(spacing_zyx),
        )

        # Remove the physical offset introduced by the 1-voxel padding
        verts_zyx = verts_zyx - spacing_zyx[None, :]

        origin_z, origin_y, origin_x = origin

        # Convert Z,Y,X to X,Y,Z world coordinates
        verts_xyz = np.column_stack(
            [
                origin_x + verts_zyx[:, 2],
                origin_y + verts_zyx[:, 1],
                origin_z * self.zratio + verts_zyx[:, 0],
            ]
        ).astype(np.float32)

        faces = faces.astype(np.int32, copy=False)

        return {
            'vertices': verts_xyz,
            'faces': faces,
            'n_vertices': int(verts_xyz.shape[0]),
            'n_faces': int(faces.shape[0]),
            'voxel_count': int(voxel_count),
            'zratio': float(self.zratio),
        }


def _toggle_full_3d_view(self, *args, **kwargs):
    z_res = float(self.full_view_z_res_spin.value())

    if z_res <= 0:
        QMessageBox.warning(
            self, 'Invalid Z Ratio', 'Z Ratio must be larger than 0.'
        )
        return

    current_ndisplay = int(getattr(self.viewer.dims, 'ndisplay', 2))

    if current_ndisplay == 2:
        _prepare_full_3d_navigation(self)
        _apply_full_view_z_scale(self, z_res)

        self.viewer.dims.ndisplay = 3

        try:
            self.viewer.axes.visible = True
            self.viewer.scale_bar.visible = True
        except Exception:
            pass

        self.btn_toggle_full_3d_view.setText('Switch to 2D View')

        with suppress(Exception):
            self.viewer.reset_view()

    else:
        self.viewer.dims.ndisplay = 2
        self.btn_toggle_full_3d_view.setText('Switch to 3D View')

        with suppress(Exception):
            self.viewer.reset_view()


def _prepare_full_3d_navigation(self):
    """
    Prepare main viewer for 3D camera navigation.

    Important:
    Labels pick/paint/erase modes can consume mouse dragging events,
    which prevents 3D camera rotation.
    """
    if getattr(self, '_pick_mode_active', False):
        labels_layer = self._get_labels_layer()
        if labels_layer is not None:
            with suppress(Exception):
                labels_layer.mode = 'pan_zoom'

        self._current_roi_mode = None
        self._new_label_mode_active = False
        self._delete_inside_mode_active = False
        self._delete_all_active = False
        self._pick_mode_active = False

        if hasattr(self, 'btn_pick_mode'):
            self.btn_pick_mode.setText('Enter Curation Mode')

        with suppress(Exception):
            self._update_curation_controls()

    for layer in self.viewer.layers:
        if hasattr(layer, 'mode'):
            with suppress(Exception):
                layer.mode = 'pan_zoom'

    for layer in self.viewer.layers:
        if layer.__class__.__name__ == 'Image':
            with suppress(Exception):
                self.viewer.layers.selection.active = layer
            break


def _apply_full_view_z_scale(self, z_res):
    """
    Apply anisotropic Z scale to all spatial layers in the main viewer.

    For 3D data with axis order (Z, Y, X), layer.scale should be:
        (z_res, 1.0, 1.0)

    For higher-dimensional data, only the last three spatial axes are scaled.
    """
    for layer in self.viewer.layers:
        try:
            ndim = int(layer.ndim)
        except Exception:
            continue

        if ndim < 3:
            continue

        old_scale = tuple(getattr(layer, 'scale', (1.0,) * ndim))

        if len(old_scale) != ndim:
            old_scale = (1.0,) * ndim

        # Save original scale only once
        if 'original_scale_before_full_3d_view' not in layer.metadata:
            layer.metadata['original_scale_before_full_3d_view'] = old_scale

        new_scale = list(old_scale)

        # Spatial axis order is assumed to be (..., Z, Y, X)
        new_scale[-3] = float(z_res)
        new_scale[-2] = 1.0
        new_scale[-1] = 1.0

        layer.scale = tuple(new_scale)

        with suppress(Exception):
            layer.refresh()


def _show_3d_reconstruction(self):
    """
    Build the 3D reconstruction for the selected label.

    Heavy mesh extraction runs in a worker thread.
    The final 3D display is shown in a separate napari viewer window
    using a Surface layer.
    """
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        QMessageBox.warning(
            self,
            'No Label Layer',
            'Please load a label layer first. Use "Load from Zarr" or "Load from TIFF" before reconstructing a selected label.',
        )
        return

    if hasattr(self, '_activate_labels_layer'):
        self._activate_labels_layer(labels_layer, show_warning=False)
    else:
        self.viewer.layers.selection.active = labels_layer

    selected_label = int(labels_layer.selected_label)
    if selected_label == 0:
        QMessageBox.warning(
            self, 'No Label', 'Please select a non-background label first.'
        )
        return

    zratio = float(self.z_res_spin.value())

    self.recon_progress = QProgressDialog(
        'Preparing 3D reconstruction...',
        'Cancel',
        0,
        100,
        self,
    )
    self.recon_progress.setWindowTitle('3D Reconstruction')
    self.recon_progress.setAutoClose(True)
    self.recon_progress.setMinimumDuration(0)
    self.recon_progress.setValue(0)
    self.recon_progress.show()

    self.recon_thread = QThread(self)

    # Wrap ReconstructionWorker with QObject-style signals dynamically
    class _QtReconstructionWorker(ReconstructionWorker, QObject):
        finished = Signal(object)
        error = Signal(str)
        progress = Signal(int, str)

        def __init__(self, labels_data, label_id, zratio):
            QObject.__init__(self)
            ReconstructionWorker.__init__(self, labels_data, label_id, zratio)

    self.recon_worker = _QtReconstructionWorker(
        labels_layer.data,
        selected_label,
        zratio,
    )
    self.recon_worker.moveToThread(self.recon_thread)

    self.recon_thread.started.connect(self.recon_worker.run)
    self.recon_worker.progress.connect(self._on_recon_progress)
    self.recon_worker.finished.connect(self._on_reconstruction_finished)
    self.recon_worker.error.connect(self._on_recon_error)

    self.recon_progress.canceled.connect(self.recon_worker.cancel)

    self.recon_worker.finished.connect(self.recon_thread.quit)
    self.recon_worker.error.connect(self.recon_thread.quit)
    self.recon_worker.finished.connect(self.recon_worker.deleteLater)
    self.recon_worker.error.connect(self.recon_worker.deleteLater)
    self.recon_thread.finished.connect(self.recon_thread.deleteLater)

    self.recon_thread.start()


def _on_recon_progress(self, value, message):
    """Update progress dialog safely."""
    if hasattr(self, 'recon_progress') and self.recon_progress is not None:
        self.recon_progress.setValue(int(value))
        self.recon_progress.setLabelText(str(message))


def _on_reconstruction_finished(self, result):
    """
    Handle successful reconstruction.

    Instead of creating a custom VTK dialog, open a separate napari viewer
    and display the reconstructed mesh as a Surface layer.
    """
    if hasattr(self, 'recon_progress') and self.recon_progress is not None:
        self.recon_progress.close()
        self.recon_progress = None

    if result is None or result.get('n_vertices', 0) == 0:
        QMessageBox.information(
            self, 'Empty', 'No voxels found for this label.'
        )
        return

    self.recon_result = result
    self.btn_save_3d_mesh.setEnabled(True)

    vertices_xyz = result['vertices']
    faces = result['faces']

    if (
        vertices_xyz is None
        or faces is None
        or len(vertices_xyz) == 0
        or len(faces) == 0
    ):
        QMessageBox.information(
            self, 'Empty', 'No valid mesh was generated for this label.'
        )
        return

    # Convert coordinates from X, Y, Z to napari's expected Z, Y, X order.
    # Z is already scaled by zratio in the worker output.
    vertices_zyx = np.column_stack(
        [
            vertices_xyz[:, 2],
            vertices_xyz[:, 1],
            vertices_xyz[:, 0],
        ]
    ).astype(np.float32)

    # Use Z coordinate as vertex values for coloring
    vertex_values = vertices_zyx[:, 0].astype(np.float32)

    # Close previous reconstruction viewer if it exists
    if hasattr(self, 'recon_viewer') and self.recon_viewer is not None:
        with suppress(Exception):
            self.recon_viewer.close()

    labels_layer = self._get_labels_layer()
    label_name = 'label'
    label_id = '?'
    if labels_layer is not None:
        label_name = labels_layer.name
        label_id = labels_layer.selected_label

    # Open a separate napari viewer in 3D mode
    self.recon_viewer = napari.Viewer(
        title=f'3D Reconstruction - {label_name} - Label {label_id}',
        ndisplay=3,
    )

    self.recon_surface_layer = self.recon_viewer.add_surface(
        (vertices_zyx, faces, vertex_values),
        name=f'Label {label_id}',
        opacity=1.0,
        shading='smooth',
        colormap='viridis',
    )

    self.recon_viewer.axes.visible = True
    self.recon_viewer.scale_bar.visible = True

    with suppress(Exception):
        self.recon_viewer.reset_view()


def _on_recon_error(self, error_msg):
    """Handle reconstruction failure."""
    if hasattr(self, 'recon_progress') and self.recon_progress is not None:
        self.recon_progress.close()
        self.recon_progress = None

    if error_msg == 'Cancelled':
        return

    QMessageBox.critical(
        self, 'Error', f'Failed to build 3D model: {error_msg}'
    )


def _save_3d_mesh(self):
    """Save the last reconstructed mesh to an NPZ file."""
    if not hasattr(self, 'recon_result') or self.recon_result is None:
        QMessageBox.information(
            self, 'No Mesh', 'Please reconstruct a 3D mesh first.'
        )
        return

    labels_layer = self._get_labels_layer()
    label_id = None
    if labels_layer is not None:
        label_id = int(labels_layer.selected_label)

    default_name = (
        'reconstruction_mesh.npz'
        if label_id is None
        else f'label_{label_id}_mesh.npz'
    )

    file_path, _ = QFileDialog.getSaveFileName(
        self, 'Save 3D Mesh', default_name, 'NumPy mesh (*.npz)'
    )
    if not file_path:
        return

    result = self.recon_result

    np.savez_compressed(
        file_path,
        vertices=result['vertices'],
        faces=result['faces'],
        n_vertices=np.int32(result['n_vertices']),
        n_faces=np.int32(result['n_faces']),
        voxel_count=np.int32(result['voxel_count']),
        zratio=np.float32(result['zratio']),
        label_id=np.int32(-1 if label_id is None else label_id),
    )

    QMessageBox.information(self, 'Saved', f'3D mesh saved to:\n{file_path}')


def _load_3d_mesh(self):
    """Load a saved NPZ mesh file and display it in a 3D napari viewer."""
    file_path, _ = QFileDialog.getOpenFileName(
        self, 'Load 3D Mesh', '', 'NumPy mesh (*.npz)'
    )
    if not file_path:
        return

    try:
        data = np.load(file_path)

        result = {
            'vertices': data['vertices'],
            'faces': data['faces'],
            'n_vertices': int(data['n_vertices']),
            'n_faces': int(data['n_faces']),
            'voxel_count': int(data['voxel_count']),
            'zratio': float(data['zratio']),
        }

        self.recon_result = result
        self.btn_save_3d_mesh.setEnabled(True)

        vertices_xyz = result['vertices']
        faces = result['faces']

        if (
            vertices_xyz is None
            or faces is None
            or len(vertices_xyz) == 0
            or len(faces) == 0
        ):
            QMessageBox.information(
                self, 'Empty', 'The loaded mesh file is empty.'
            )
            return

        # Convert X, Y, Z back to napari Surface input order: Z, Y, X
        vertices_zyx = np.column_stack(
            [
                vertices_xyz[:, 2],
                vertices_xyz[:, 1],
                vertices_xyz[:, 0],
            ]
        ).astype(np.float32)

        vertex_values = vertices_zyx[:, 0].astype(np.float32)

        if hasattr(self, 'recon_viewer') and self.recon_viewer is not None:
            with suppress(Exception):
                self.recon_viewer.close()
            self.recon_viewer = None

        loaded_label_id = '?'
        if 'label_id' in data:
            loaded_label_id = int(data['label_id'])
            if loaded_label_id < 0:
                loaded_label_id = '?'

        self.recon_viewer = napari.Viewer(
            title=f'3D Reconstruction - Loaded Mesh - Label {loaded_label_id}',
            ndisplay=3,
        )

        self.recon_surface_layer = self.recon_viewer.add_surface(
            (vertices_zyx, faces, vertex_values),
            name=f'Loaded Label {loaded_label_id}',
            opacity=1.0,
            shading='smooth',
            colormap='viridis',
        )

        self.recon_viewer.axes.visible = True
        self.recon_viewer.scale_bar.visible = True

        with suppress(Exception):
            self.recon_viewer.reset_view()

    except (ValueError, OSError, KeyError) as e:
        QMessageBox.critical(
            self, 'Load Error', f'Failed to load 3D mesh:\n{e}'
        )


def _show_quantitative_stats(self):
    """Compute statistics in background, then open dialog."""
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        QMessageBox.warning(
            self,
            'No Label Layer',
            'Please load or create a label layer first.',
        )
        return

    image_layer = self._get_active_image()
    labels_data = labels_layer.data
    image_data = image_layer.data if image_layer else None

    # Progress dialog
    self.stats_progress = QProgressDialog(
        'Computing statistics...', 'Cancel', 0, 100, self
    )
    self.stats_progress.setWindowTitle('Statistics Progress')
    self.stats_progress.setAutoClose(True)
    self.stats_progress.show()

    # Worker
    self.stats_thread = QThread()
    self.stats_worker = QuantitativeWorker(labels_data, image_data)
    self.stats_worker.moveToThread(self.stats_thread)

    self.stats_worker.progress.connect(
        lambda val, msg: self.stats_progress.setLabelText(msg)
    )
    self.stats_worker.progress.connect(
        lambda val, msg: self.stats_progress.setValue(val)
    )
    self.stats_worker.finished.connect(self._on_stats_finished)
    self.stats_progress.canceled.connect(self.stats_worker.cancel)
    self.stats_progress.canceled.connect(self.stats_thread.quit)
    self.stats_thread.started.connect(self.stats_worker.run)
    self.stats_worker.finished.connect(self.stats_thread.quit)
    self.stats_worker.finished.connect(self.stats_worker.deleteLater)
    self.stats_thread.finished.connect(self.stats_thread.deleteLater)

    self.stats_thread.start()


def _on_stats_finished(self, stats, error_msg):
    """Called when statistics are ready; display dialog in main thread."""
    self.stats_progress.close()
    if error_msg:
        QMessageBox.critical(self, 'Statistics Error', error_msg)
        return
    if not stats or stats['num_cells'] == 0:
        QMessageBox.information(
            self, 'No Cells', 'No non‑background labels found.'
        )
        return

    # Now build and show the statistics dialog (same UI as before)
    self._display_stats_dialog(
        stats
    )  # extract the dialog creation code to a separate method


def _display_stats_dialog(self, stats):
    """
    Display the quantitative statistics dialog using pre-computed stats.
    stats: dict with keys: 'labels', 'areas', 'num_cells', 'max_area', 'min_area'
        (intensity stats can be added later)
    """
    dlg = QDialog(self)
    dlg.setWindowTitle('Quantitative Statistics')
    dlg.resize(800, 600)
    main_layout = QVBoxLayout(dlg)

    # --- Size Statistics Group ---
    size_group = QGroupBox('Cell Size Statistics')
    size_layout = QVBoxLayout(size_group)

    num_cells = stats['num_cells']
    max_area = stats['max_area']
    min_area = stats['min_area']
    areas_list = stats['areas']

    # Summary labels
    summary_layout = QHBoxLayout()
    summary_layout.addWidget(QLabel(f'Number of cells: {num_cells}'))
    summary_layout.addWidget(QLabel(f'Max size: {int(max_area)}'))
    summary_layout.addWidget(QLabel(f'Min size: {int(min_area)}'))
    summary_layout.addStretch()
    size_layout.addLayout(summary_layout)

    # Histogram of cell sizes
    fig_size = Figure(figsize=(5, 3))
    canvas_size = FigureCanvas(fig_size)
    ax_size = fig_size.add_subplot(111)
    ax_size.hist(areas_list, bins=50, color='skyblue', edgecolor='black')
    ax_size.set_xlabel('Size (voxels)')
    ax_size.set_ylabel('Frequency')
    ax_size.set_title('Cell Size Distribution')
    fig_size.tight_layout()
    size_layout.addWidget(canvas_size)

    main_layout.addWidget(size_group)

    # --- Intensity Statistics Group (if available) ---
    # For now, intensity stats are not computed in the background.
    # You can extend the QuantitativeWorker to also compute and include them in the stats dict.
    intensity_group = QGroupBox('Intensity Statistics (within cells)')
    intensity_layout = QVBoxLayout(intensity_group)
    no_intensity_label = QLabel('Intensity statistics not computed.')
    no_intensity_label.setStyleSheet('color: gray; font-style: italic;')
    intensity_layout.addWidget(no_intensity_label)
    main_layout.addWidget(intensity_group)

    # Close button
    btn_close = QPushButton('Close')
    btn_close.clicked.connect(dlg.accept)
    main_layout.addWidget(btn_close, alignment=Qt.AlignCenter)

    dlg.exec_()


# ---------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------
class MorphometryTaskWorker(QObject):
    """
    Run one selected morphometry task in a background QThread.

    This worker is intentionally task-based:
        morphology
        intensity
        neighborhood
        contact
        region
        clustering
        anomaly

    It does not run a coupled all-in-one morphometry pipeline.
    """

    progress = Signal(int, str)
    finished = Signal(dict, str)

    def __init__(
        self,
        analysis_name,
        raw_data,
        label_data,
        out_dir,
        common_config,
        task_config,
    ):
        super().__init__()
        self.analysis_name = str(analysis_name)
        self.raw_data = raw_data
        self.label_data = label_data
        self.out_dir = out_dir
        self.common_config = common_config or {}
        self.task_config = task_config or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _progress_callback(self, value, message):
        self.progress.emit(int(value), str(message))

    def _cancel_callback(self):
        return bool(self._cancelled)

    def run(self):
        try:
            # Import only when the user actually runs a task.
            # This prevents optional morphometry dependencies from breaking
            # plugin startup.
            from focus3d.analysis import morphometry_engine as engine

            name = self.analysis_name

            if name == 'basic_info':
                result = engine.run_basic_info_from_arrays(
                    raw_data=self.raw_data,
                    label_data=self.label_data,
                    out_dir=self.out_dir,
                    common_config=self.common_config,
                    basic_config=self.task_config,
                    progress_callback=self._progress_callback,
                    cancel_callback=self._cancel_callback,
                )

            elif name == 'neighborhood':
                result = engine.run_neighborhood_from_arrays(
                    label_data=self.label_data,
                    out_dir=self.out_dir,
                    common_config=self.common_config,
                    neighborhood_config=self.task_config,
                    progress_callback=self._progress_callback,
                    cancel_callback=self._cancel_callback,
                )

            elif name == 'local_comparison':
                result = engine.run_local_comparison_from_arrays(
                    raw_data=self.raw_data,
                    label_data=self.label_data,
                    out_dir=self.out_dir,
                    common_config=self.common_config,
                    local_config=self.task_config,
                    progress_callback=self._progress_callback,
                    cancel_callback=self._cancel_callback,
                )

            elif name == 'contact':
                result = engine.run_contact_from_arrays(
                    label_data=self.label_data,
                    out_dir=self.out_dir,
                    common_config=self.common_config,
                    contact_config=self.task_config,
                    progress_callback=self._progress_callback,
                    cancel_callback=self._cancel_callback,
                )

            elif name == 'clustering':
                result = engine.run_clustering_from_arrays(
                    raw_data=self.raw_data,
                    label_data=self.label_data,
                    out_dir=self.out_dir,
                    common_config=self.common_config,
                    clustering_config=self.task_config,
                    progress_callback=self._progress_callback,
                    cancel_callback=self._cancel_callback,
                )

            else:
                raise ValueError(f'Unknown analysis task: {name}')

            if result.get('cancelled', False):
                self.finished.emit({}, 'Cancelled')
                return

            self.finished.emit(result, '')

        except ModuleNotFoundError as e:
            if getattr(e, 'name', '') == 'sklearn':
                self.finished.emit(
                    {},
                    'This task requires scikit-learn.\n\n'
                    'Install it with:\n\n'
                    '    pip install scikit-learn\n\n'
                    'Note: the import name is sklearn, but the package name is scikit-learn.',
                )
            else:
                self.finished.emit({}, f'Missing dependency: {e}')

        except Exception as e:
            self.finished.emit({}, str(e))


def _parse_int_list(text: str, default=(10,)):
    vals = []
    for part in str(text).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(float(part)))
    vals = tuple(v for v in vals if v > 0)
    return vals if vals else tuple(default)


def _parse_float_list(text: str, default=(20.0,)):
    vals = []
    for part in str(text).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    vals = tuple(v for v in vals if v > 0)
    return vals if vals else tuple(default)


def _make_small_spin(value, minimum=0.0, maximum=9999.0, decimals=2):
    spin = QDoubleSpinBox()
    spin.setRange(float(minimum), float(maximum))
    spin.setDecimals(int(decimals))
    spin.setSingleStep(0.1)
    spin.setValue(float(value))
    spin.setFixedWidth(58)
    return spin


def _make_small_int_spin(value, minimum=1, maximum=9999):
    spin = QSpinBox()
    spin.setRange(int(minimum), int(maximum))
    spin.setValue(int(value))
    spin.setFixedWidth(58)
    return spin


def _default_morphometry_output_dir(self) -> str:
    """Default output directory for morphometry analysis."""
    labels_layer = None
    if hasattr(self, '_get_segmentation_labels_layer'):
        labels_layer = self._get_segmentation_labels_layer()
    if labels_layer is None and hasattr(self, '_get_labels_layer'):
        labels_layer = self._get_labels_layer()

    source_path = None
    if labels_layer is not None:
        try:
            source_path = getattr(labels_layer.source, 'path', None)
        except Exception:
            source_path = None

    if source_path:
        p = Path(str(source_path))
        base = p.parent if p.suffix else p
        return str(base / 'morphometry')

    # Fall back to current label path if it exists.
    if hasattr(self, 'current_label_path') and self.current_label_path:
        p = Path(str(self.current_label_path))
        base = p.parent if p.suffix else p
        return str(base / 'morphometry')

    return str(Path.cwd() / 'morphometry')


def _make_analysis_section_box(title: str) -> QGroupBox:
    """
    Create one analysis section box.

    The explicit stylesheet prevents Qt / parent stylesheet from making
    the 1-7 analysis titles bold.
    """
    box = QGroupBox(title)
    box.setMinimumWidth(0)
    box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    box.setStyleSheet(
        """
        QGroupBox {
            font-weight: normal;
        }
        QGroupBox::title {
            font-weight: normal;
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 3px;
        }
        """
    )
    return box


def _make_shrinkable(widget):
    """
    Allow line edits / combo boxes to shrink inside a narrow napari dock.
    This avoids forcing the whole Analysis panel wider than the plugin.
    """
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return widget


def _make_compact_combo(combo: QComboBox, min_chars: int = 10):
    """
    Prevent long layer names from expanding the whole panel width.
    """
    combo.setMinimumWidth(0)
    combo.setMinimumContentsLength(int(min_chars))
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return combo


BASIC_INFO_FEATURES = [
    ('Volume', 'volume_um3'),
    ('Equivalent diameter', 'equivalent_diameter_um'),
    ('Surface area', 'surface_area_um2'),
    ('Centroid', 'centroid'),
    ('Sphericity', 'sphericity'),
    ('Compactness', 'compactness'),
    ('Axis major', 'axis_major_um'),
    ('Elongation', 'elongation'),
    ('Flatness', 'flatness'),
    ('Min intensity', 'min_intensity'),
    ('Max intensity', 'max_intensity'),
    ('Mean intensity', 'mean_intensity'),
    ('Std intensity', 'std_intensity'),
]

# These are computed into CSV but should not be shown as mapped 3D feature layers.
NON_VISUAL_BASIC_FEATURES = {
    'centroid',
    'axis_major_um',
    'min_intensity',
    'max_intensity',
}

LOCAL_COMPARISON_FEATURES = [
    ('Mean intensity', 'mean_intensity'),
    ('Sphericity', 'sphericity'),
    ('Volume', 'volume_um3'),
    ('Surface area', 'surface_area_um2'),
    ('Compactness', 'compactness'),
    ('Elongation', 'elongation'),
    ('Flatness', 'flatness'),
]

CONTACT_FEATURES = [
    ('Neighbor count', 'contact_neighbor_count'),
    ('Total contact area', 'contact_area_total_um2'),
    ('Mean contact area', 'contact_area_mean_um2'),
    ('Max contact area', 'contact_area_max_um2'),
    ('Contact fraction', 'contact_fraction'),
]

CLUSTERING_FEATURES = [
    ('Volume', 'volume_um3'),
    ('Equivalent diameter', 'equivalent_diameter_um'),
    ('Surface area', 'surface_area_um2'),
    ('Sphericity', 'sphericity'),
    ('Compactness', 'compactness'),
    ('Elongation', 'elongation'),
    ('Flatness', 'flatness'),
    ('Mean intensity', 'mean_intensity'),
    ('Std intensity', 'std_intensity'),
    ('Contact neighbor count', 'contact_neighbor_count'),
    ('Total contact area', 'contact_area_total_um2'),
    ('Mean contact area', 'contact_area_mean_um2'),
    ('Contact fraction', 'contact_fraction'),
]


def _feature_requires_raw(feature: str) -> bool:
    return str(feature) in {
        'min_intensity',
        'max_intensity',
        'mean_intensity',
        'std_intensity',
    }


def _add_feature_combo_items(combo: QComboBox, items):
    combo.clear()
    for text, feature in items:
        combo.addItem(text, feature)


def _checked_features_from_checkboxes(self, attr_prefix: str):
    features = []
    for _, feature in BASIC_INFO_FEATURES:
        attr = f'{attr_prefix}_{feature}_chk'.replace('.', '_')
        if hasattr(self, attr) and getattr(self, attr).isChecked():
            features.append(feature)
    return features


def _selected_contact_features(self):
    out = []
    for _, feature in CONTACT_FEATURES:
        attr = f'contact_{feature}_chk'
        if hasattr(self, attr) and getattr(self, attr).isChecked():
            out.append(feature)
    return out


def _make_feature_checkbox(text: str, checked: bool = True) -> QCheckBox:
    chk = QCheckBox(text)
    chk.setChecked(bool(checked))
    chk.setMinimumWidth(0)
    chk.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return chk


def _make_row_widget(label_text: str, editor_widget: QWidget):
    """
    Create a row widget so the whole row can be shown/hidden.
    """
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)

    row_layout.addWidget(QLabel(label_text))
    row_layout.addWidget(editor_widget)
    row_layout.addStretch()

    return row_widget


def _add_scrollable_feature_checkboxes(
    self,
    parent_layout,
    features,
    attr_prefix: str,
    checked_features,
    height: int = 150,
):
    """
    Add many feature checkboxes into a compact scroll area.

    This prevents the Basic Information section from becoming too tall.
    """
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(0)
    scroll.setFixedHeight(int(height))
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(4, 4, 4, 4)
    content_layout.setSpacing(3)

    checked_features = set(checked_features)

    for text, feature in features:
        chk = _make_feature_checkbox(
            text,
            checked=feature in checked_features,
        )
        attr = f'{attr_prefix}_{feature}_chk'.replace('.', '_')
        setattr(self, attr, chk)
        content_layout.addWidget(chk)

    content_layout.addStretch()
    scroll.setWidget(content)

    parent_layout.addWidget(scroll)
    return scroll


# ---------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------
def _init_morphometry_analysis_group(self, parent_layout, group_style=None):
    """
    Rebuilt morphometry analysis panel.

    New logic:
        1. Basic information
        2. Neighborhood analysis
        3. Contact graph analysis
        4. Clustering
    """
    if group_style is None:
        group_style = self._section_group_style()

    group = QGroupBox('Morphometry Analysis')
    group.setMinimumWidth(0)
    group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    group.setStyleSheet(group_style)

    layout = QVBoxLayout(group)
    layout.setContentsMargins(4, 8, 4, 6)
    layout.setSpacing(6)

    _init_morphometry_common_group(self, layout)

    _init_basic_info_group(self, layout)
    _init_neighborhood_group(self, layout)
    _init_contact_group(self, layout)
    _init_clustering_group(self, layout)

    parent_layout.addWidget(group)


def _init_morphometry_common_group(self, layout):
    """Common settings shared by all morphometry tasks."""
    common_box = QGroupBox('Common Settings')
    common_layout = QVBoxLayout(common_box)

    voxel_title = QLabel('Voxel size (Z / Y / X)')
    common_layout.addWidget(voxel_title)

    voxel_row = QHBoxLayout()
    self.morph_voxel_z_spin = _make_small_spin(1.0, 0.0001, 10000.0, 3)
    self.morph_voxel_y_spin = _make_small_spin(1.0, 0.0001, 10000.0, 3)
    self.morph_voxel_x_spin = _make_small_spin(1.0, 0.0001, 10000.0, 3)
    voxel_row.addWidget(self.morph_voxel_z_spin)
    voxel_row.addWidget(self.morph_voxel_y_spin)
    voxel_row.addWidget(self.morph_voxel_x_spin)
    voxel_row.addStretch()
    common_layout.addLayout(voxel_row)

    output_row = QHBoxLayout()
    self.morph_output_edit = QLineEdit(_default_morphometry_output_dir(self))
    _make_shrinkable(self.morph_output_edit)
    output_row.addWidget(self.morph_output_edit)

    self.btn_browse_morph_output = QPushButton('Browse')
    self.btn_browse_morph_output.setFixedWidth(70)
    self.btn_browse_morph_output.clicked.connect(
        lambda: _browse_morphometry_output_dir(self)
    )
    output_row.addWidget(self.btn_browse_morph_output)
    common_layout.addLayout(output_row)

    self.btn_open_morphometry_output = QPushButton('Open Output Folder')
    self.btn_open_morphometry_output.clicked.connect(
        lambda: _open_morphometry_output_folder(self)
    )
    common_layout.addWidget(self.btn_open_morphometry_output)

    layout.addWidget(common_box)


def _init_basic_info_group(self, layout):
    box = _make_analysis_section_box('1. Basic Information')
    box_layout = QVBoxLayout(box)
    box_layout.setContentsMargins(8, 8, 8, 8)
    box_layout.setSpacing(6)

    info = QLabel(
        'Compute selected per-cell measurements and save them into one CSV table.'
    )
    info.setWordWrap(True)
    info.setMinimumWidth(0)
    box_layout.addWidget(info)

    raw_row = QHBoxLayout()
    raw_row.addWidget(QLabel('Raw layer:'))
    self.basic_raw_layer_combo = QComboBox()
    _make_compact_combo(self.basic_raw_layer_combo, min_chars=10)
    raw_row.addWidget(self.basic_raw_layer_combo)
    raw_row.addStretch()
    box_layout.addLayout(raw_row)

    self.btn_refresh_basic_raw_layers = QPushButton('Refresh Raw Layers')
    self.btn_refresh_basic_raw_layers.clicked.connect(
        lambda: _refresh_basic_raw_layers(self)
    )
    box_layout.addWidget(self.btn_refresh_basic_raw_layers)

    feature_label = QLabel('Features to compute')
    box_layout.addWidget(feature_label)

    _add_scrollable_feature_checkboxes(
        self,
        parent_layout=box_layout,
        features=BASIC_INFO_FEATURES,
        attr_prefix='basic',
        checked_features={
            'volume_um3',
            'equivalent_diameter_um',
            'surface_area_um2',
            'centroid',
            'sphericity',
            'compactness',
            'mean_intensity',
            'std_intensity',
        },
        height=150,
    )

    self.btn_run_basic_info = QPushButton('Run Basic Information')
    self.btn_run_basic_info.clicked.connect(
        lambda: _run_single_morphometry_task(
            self,
            analysis_name='basic_info',
            task_config=_collect_basic_info_config(self),
            require_raw=_basic_info_requires_raw(self),
        )
    )
    box_layout.addWidget(self.btn_run_basic_info)

    vis_row = QHBoxLayout()
    vis_row.addWidget(QLabel('Show feature:'))
    self.basic_visual_feature_combo = QComboBox()
    _make_compact_combo(self.basic_visual_feature_combo, min_chars=12)
    self.basic_visual_feature_combo.setEnabled(False)
    vis_row.addWidget(self.basic_visual_feature_combo)

    self.btn_show_basic_feature = QPushButton('Show')
    self.btn_show_basic_feature.setEnabled(False)
    self.btn_show_basic_feature.clicked.connect(
        lambda: _show_feature_from_result(
            self,
            result_attr='basic_info_result',
            combo=self.basic_visual_feature_combo,
            layer_prefix='basic',
        )
    )
    vis_row.addWidget(self.btn_show_basic_feature)
    box_layout.addLayout(vis_row)

    _refresh_basic_raw_layers(self)

    layout.addWidget(box)


def _init_neighborhood_group(self, layout):
    box = _make_analysis_section_box('2. Neighborhood Analysis')
    box_layout = QVBoxLayout(box)
    box_layout.setContentsMargins(8, 8, 8, 8)
    box_layout.setSpacing(6)

    info = QLabel(
        'First compute neighborhood features using either kNN or radius. '
        'After that, run local comparison for one selected cell feature.'
    )
    info.setWordWrap(True)
    info.setMinimumWidth(0)
    box_layout.addWidget(info)

    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel('Mode:'))
    self.neigh_mode_combo = QComboBox()
    self.neigh_mode_combo.addItems(['kNN distance', 'Radius count / density'])
    mode_row.addWidget(self.neigh_mode_combo)
    mode_row.addStretch()
    box_layout.addLayout(mode_row)

    self.neigh_k_spin = _make_small_int_spin(10, 1, 999)
    self.neigh_k_row_widget = _make_row_widget('k:', self.neigh_k_spin)
    box_layout.addWidget(self.neigh_k_row_widget)

    self.neigh_radius_spin = _make_small_spin(20.0, 0.0001, 1000000.0, 2)
    self.neigh_radius_row_widget = _make_row_widget(
        'Radius pixels:',
        self.neigh_radius_spin,
    )
    box_layout.addWidget(self.neigh_radius_row_widget)

    self.neigh_mode_combo.currentTextChanged.connect(
        lambda _: _update_neighborhood_parameter_visibility(self)
    )
    _update_neighborhood_parameter_visibility(self)

    self.btn_run_neighborhood = QPushButton('Run Neighborhood')
    self.btn_run_neighborhood.clicked.connect(
        lambda: _run_single_morphometry_task(
            self,
            analysis_name='neighborhood',
            task_config=_collect_neighborhood_config(self),
            require_raw=False,
        )
    )
    box_layout.addWidget(self.btn_run_neighborhood)

    neigh_vis_row = QHBoxLayout()
    neigh_vis_row.addWidget(QLabel('Show feature:'))
    self.neigh_visual_feature_combo = QComboBox()
    _make_compact_combo(self.neigh_visual_feature_combo, min_chars=12)
    self.neigh_visual_feature_combo.setEnabled(False)
    neigh_vis_row.addWidget(self.neigh_visual_feature_combo)

    self.btn_show_neigh_feature = QPushButton('Show')
    self.btn_show_neigh_feature.setEnabled(False)
    self.btn_show_neigh_feature.clicked.connect(
        lambda: _show_feature_from_result(
            self,
            result_attr='neighborhood_result',
            combo=self.neigh_visual_feature_combo,
            layer_prefix='neighborhood',
        )
    )
    neigh_vis_row.addWidget(self.btn_show_neigh_feature)
    box_layout.addLayout(neigh_vis_row)

    local_title = QLabel('Local comparison')
    local_title.setStyleSheet('font-weight: normal;')
    box_layout.addWidget(local_title)

    local_row = QHBoxLayout()
    local_row.addWidget(QLabel('Feature:'))
    self.local_compare_feature_combo = QComboBox()
    _make_compact_combo(self.local_compare_feature_combo, min_chars=12)
    _add_feature_combo_items(
        self.local_compare_feature_combo, LOCAL_COMPARISON_FEATURES
    )
    local_row.addWidget(self.local_compare_feature_combo)
    local_row.addStretch()
    box_layout.addLayout(local_row)

    self.btn_run_local_comparison = QPushButton('Run Local Comparison')
    self.btn_run_local_comparison.setEnabled(False)
    self.btn_run_local_comparison.clicked.connect(
        lambda: _run_single_morphometry_task(
            self,
            analysis_name='local_comparison',
            task_config=_collect_local_comparison_config(self),
            require_raw=_feature_requires_raw(
                self.local_compare_feature_combo.currentData()
            ),
        )
    )
    box_layout.addWidget(self.btn_run_local_comparison)

    layout.addWidget(box)


def _update_neighborhood_parameter_visibility(self):
    """
    Show only the parameter relevant to the selected neighborhood mode.
    """
    if not hasattr(self, 'neigh_mode_combo'):
        return

    mode_text = self.neigh_mode_combo.currentText()
    is_knn = mode_text == 'kNN distance'

    if hasattr(self, 'neigh_k_row_widget'):
        self.neigh_k_row_widget.setVisible(is_knn)

    if hasattr(self, 'neigh_radius_row_widget'):
        self.neigh_radius_row_widget.setVisible(not is_knn)


def _init_contact_group(self, layout):
    box = _make_analysis_section_box('3. Contact Graph Analysis')
    box_layout = QVBoxLayout(box)
    box_layout.setContentsMargins(8, 8, 8, 8)
    box_layout.setSpacing(6)

    info = QLabel(
        'Compute face-touching contact graph features for each cell.'
    )
    info.setWordWrap(True)
    info.setMinimumWidth(0)
    box_layout.addWidget(info)

    box_layout.addWidget(QLabel('Features to compute'))

    for text, feature in CONTACT_FEATURES:
        chk = _make_feature_checkbox(text, checked=True)
        setattr(self, f'contact_{feature}_chk', chk)
        box_layout.addWidget(chk)

    self.btn_run_contact = QPushButton('Run Contact Graph')
    self.btn_run_contact.clicked.connect(
        lambda: _run_single_morphometry_task(
            self,
            analysis_name='contact',
            task_config=_collect_contact_config(self),
            require_raw=False,
        )
    )
    box_layout.addWidget(self.btn_run_contact)

    vis_row = QHBoxLayout()
    vis_row.addWidget(QLabel('Show feature:'))
    self.contact_visual_feature_combo = QComboBox()
    _make_compact_combo(self.contact_visual_feature_combo, min_chars=12)
    self.contact_visual_feature_combo.setEnabled(False)
    vis_row.addWidget(self.contact_visual_feature_combo)

    self.btn_show_contact_feature = QPushButton('Show')
    self.btn_show_contact_feature.setEnabled(False)
    self.btn_show_contact_feature.clicked.connect(
        lambda: _show_feature_from_result(
            self,
            result_attr='contact_result',
            combo=self.contact_visual_feature_combo,
            layer_prefix='contact',
        )
    )
    vis_row.addWidget(self.btn_show_contact_feature)
    box_layout.addLayout(vis_row)

    layout.addWidget(box)


def _init_clustering_group(self, layout):
    box = _make_analysis_section_box('4. Clustering')
    box_layout = QVBoxLayout(box)
    box_layout.setContentsMargins(8, 8, 8, 8)
    box_layout.setSpacing(6)

    info = QLabel(
        'Cluster cells using one selected feature. '
        'The cluster_id result will be saved and displayed automatically.'
    )
    info.setWordWrap(True)
    info.setMinimumWidth(0)
    box_layout.addWidget(info)

    feature_row = QHBoxLayout()
    feature_row.addWidget(QLabel('Feature:'))
    self.cluster_feature_combo = QComboBox()
    _make_compact_combo(self.cluster_feature_combo, min_chars=12)
    _add_feature_combo_items(self.cluster_feature_combo, CLUSTERING_FEATURES)
    feature_row.addWidget(self.cluster_feature_combo)
    feature_row.addStretch()
    box_layout.addLayout(feature_row)

    cluster_row = QHBoxLayout()
    cluster_row.addWidget(QLabel('Clusters:'))
    self.cluster_n_clusters_spin = _make_small_int_spin(6, 2, 100)
    cluster_row.addWidget(self.cluster_n_clusters_spin)
    cluster_row.addStretch()
    box_layout.addLayout(cluster_row)

    self.btn_run_clustering = QPushButton('Run Clustering')
    self.btn_run_clustering.clicked.connect(
        lambda: _run_single_morphometry_task(
            self,
            analysis_name='clustering',
            task_config=_collect_clustering_config(self),
            require_raw=_feature_requires_raw(
                self.cluster_feature_combo.currentData()
            ),
        )
    )
    box_layout.addWidget(self.btn_run_clustering)
    layout.addWidget(box)


def _collect_common_morphometry_config(self):
    return {
        'voxel_size_zyx': (
            float(self.morph_voxel_z_spin.value()),
            float(self.morph_voxel_y_spin.value()),
            float(self.morph_voxel_x_spin.value()),
        ),
        # New UI logic:
        # analysis only saves CSV;
        # feature visualization is created on demand in napari.
        'save_feature_tifs': False,
        'load_maps': False,
    }


def _neighbor_mode_from_text(text):
    if text == 'kNN':
        return 'knn'
    if text == 'Radius':
        return 'radius'
    return 'knn_radius'


def _collect_neighborhood_config(self):
    mode_text = self.neigh_mode_combo.currentText()

    mode = 'radius' if mode_text == 'Radius count / density' else 'knn'

    return {
        'neighbor_mode': mode,
        'k': int(self.neigh_k_spin.value()),
        'radius_vox': float(self.neigh_radius_spin.value()),
    }


def _collect_contact_config(self):
    selected = []

    for _, feature in CONTACT_FEATURES:
        attr = f'contact_{feature}_chk'
        if hasattr(self, attr) and getattr(self, attr).isChecked():
            selected.append(feature)

    if not selected:
        selected = ['contact_neighbor_count']

    return {
        'selected_features': selected,
    }


def _collect_local_comparison_config(self):
    mode_text = self.neigh_mode_combo.currentText()

    mode = 'radius' if mode_text == 'Radius count / density' else 'knn'

    feature = self.local_compare_feature_combo.currentData()

    return {
        'raw_layer_name': self.basic_raw_layer_combo.currentText()
        if hasattr(self, 'basic_raw_layer_combo')
        else None,
        'neighbor_mode': mode,
        'k': int(self.neigh_k_spin.value()),
        'radius_vox': float(self.neigh_radius_spin.value()),
        'feature': str(feature),
    }


def _collect_basic_info_config(self):
    selected = []

    for _, feature in BASIC_INFO_FEATURES:
        attr = f'basic_{feature}_chk'.replace('.', '_')
        if hasattr(self, attr) and getattr(self, attr).isChecked():
            selected.append(feature)

    if not selected:
        selected = ['volume_um3']

    return {
        'raw_layer_name': self.basic_raw_layer_combo.currentText(),
        'selected_features': selected,
    }


def _collect_clustering_config(self):
    feature = self.cluster_feature_combo.currentData()

    return {
        'raw_layer_name': self.basic_raw_layer_combo.currentText()
        if hasattr(self, 'basic_raw_layer_combo')
        else None,
        'feature': str(feature),
        'n_clusters': int(self.cluster_n_clusters_spin.value()),
    }


def _browse_morphometry_output_dir(self):
    default = (
        self.morph_output_edit.text().strip()
        or _default_morphometry_output_dir(self)
    )
    folder = QFileDialog.getExistingDirectory(
        self, 'Select Morphometry Output Folder', default
    )
    if folder:
        self.morph_output_edit.setText(folder)


def _open_morphometry_output_folder(self):
    folder = self.morph_output_edit.text().strip()
    if not folder:
        return
    folder = str(Path(folder).expanduser())
    if not Path(folder).exists():
        notifications.show_warning('Output folder does not exist yet.')
        return
    try:
        os.startfile(folder)  # Windows
    except Exception:
        try:
            import subprocess

            subprocess.Popen(['xdg-open', folder])
        except Exception:
            notifications.show_info(f'Output folder: {folder}')


def _get_morphometry_layers(self, raw_layer_name=None):
    """
    Return image layer and label layer for morphometry.

    If raw_layer_name is provided, use that exact Image layer.
    Otherwise, use active Image layer first, then the first Image layer.
    """
    labels_layer = None
    if hasattr(self, '_get_segmentation_labels_layer'):
        labels_layer = self._get_segmentation_labels_layer()
    if labels_layer is None and hasattr(self, '_get_labels_layer'):
        labels_layer = self._get_labels_layer()

    image_layer = None

    if raw_layer_name:
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.name == raw_layer_name:
                image_layer = layer
                break

    if image_layer is None:
        try:
            active = self.viewer.layers.selection.active
            if isinstance(active, Image):
                image_layer = active
        except Exception:
            pass

    if image_layer is None:
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                image_layer = layer
                break

    return image_layer, labels_layer


def _run_single_morphometry_task(
    self, analysis_name, task_config, require_raw=False
):
    """
    Run one independent morphometry task.

    Each task writes to:
        <output_root>/<analysis_name>/
    """
    raw_layer_name = task_config.get('raw_layer_name', None)

    image_layer, labels_layer = _get_morphometry_layers(
        self,
        raw_layer_name=raw_layer_name,
    )

    if labels_layer is None:
        QMessageBox.warning(
            self,
            'No label layer',
            'Please load or create a label layer first.',
        )
        return

    if not isinstance(labels_layer, Labels):
        QMessageBox.warning(
            self,
            'Invalid label layer',
            'The selected label layer is not a napari Labels layer.',
        )
        return

    try:
        if int(getattr(labels_layer, 'ndim', 0)) != 3:
            QMessageBox.warning(
                self,
                'Invalid labels',
                'Morphometry analysis currently supports 3D labels only.',
            )
            return
    except Exception:
        pass

    if require_raw and image_layer is None:
        QMessageBox.warning(
            self,
            'No raw image',
            f'{analysis_name} analysis requires a raw image layer.',
        )
        return

    raw_data = image_layer.data if image_layer is not None else None
    label_data = labels_layer.data

    common_config = _collect_common_morphometry_config(self)

    output_root = (
        self.morph_output_edit.text().strip()
        or _default_morphometry_output_dir(self)
    )
    output_root = Path(output_root).expanduser()

    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    task_out_dir = output_root / analysis_name
    task_out_dir.mkdir(parents=True, exist_ok=True)

    self.morph_output_edit.setText(str(output_root))

    self.morph_progress = QProgressDialog(
        f'Running {analysis_name} analysis...',
        'Cancel',
        0,
        100,
        self,
    )
    self.morph_progress.setWindowTitle(
        f'{analysis_name.capitalize()} Analysis'
    )
    self.morph_progress.setAutoClose(True)
    self.morph_progress.setAutoReset(True)
    self.morph_progress.setValue(0)
    self.morph_progress.show()

    self.morph_thread = QThread()
    self.morph_worker = MorphometryTaskWorker(
        analysis_name=analysis_name,
        raw_data=raw_data,
        label_data=label_data,
        out_dir=str(task_out_dir),
        common_config=common_config,
        task_config=task_config,
    )

    self.morph_worker.moveToThread(self.morph_thread)

    self.morph_thread.started.connect(self.morph_worker.run)

    self.morph_worker.progress.connect(
        lambda v, m: self.morph_progress.setValue(int(v))
    )
    self.morph_worker.progress.connect(
        lambda v, m: self.morph_progress.setLabelText(str(m))
    )

    self.morph_worker.finished.connect(
        lambda result, err: _on_single_morphometry_finished(
            self,
            analysis_name,
            result,
            err,
        )
    )

    self.morph_progress.canceled.connect(self.morph_worker.cancel)
    self.morph_progress.canceled.connect(self.morph_thread.quit)

    self.morph_worker.finished.connect(self.morph_thread.quit)
    self.morph_worker.finished.connect(self.morph_worker.deleteLater)
    self.morph_thread.finished.connect(self.morph_thread.deleteLater)

    self.morph_thread.start()


def _on_single_morphometry_finished(self, analysis_name, result, error_msg):
    if hasattr(self, 'morph_progress') and self.morph_progress is not None:
        self.morph_progress.close()

    if error_msg:
        if error_msg == 'Cancelled':
            notifications.show_info(f'{analysis_name} cancelled.')
            return

        QMessageBox.critical(
            self,
            f'{analysis_name.replace("_", " ").title()} Error',
            str(error_msg),
        )
        return

    if not result:
        QMessageBox.warning(
            self,
            analysis_name.replace('_', ' ').title(),
            'No result was returned.',
        )
        return

    self.last_morphometry_result = result
    setattr(self, f'{analysis_name}_result', result)

    if analysis_name == 'basic_info':
        _populate_visual_feature_combo(
            self.basic_visual_feature_combo,
            result,
        )
        self.basic_visual_feature_combo.setEnabled(True)
        self.btn_show_basic_feature.setEnabled(True)

    elif analysis_name == 'neighborhood':
        _populate_visual_feature_combo(
            self.neigh_visual_feature_combo,
            result,
        )
        self.neigh_visual_feature_combo.setEnabled(True)
        self.btn_show_neigh_feature.setEnabled(True)

        # Unlock local comparison only after neighborhood is computed.
        self.btn_run_local_comparison.setEnabled(True)

    elif analysis_name == 'local_comparison':
        visualizable = result.get('visualizable_features', []) or []
        if visualizable:
            feature = visualizable[0].get('feature', '')
            _show_feature_name_from_result(
                self,
                result_attr='local_comparison_result',
                feature=feature,
                layer_prefix='local',
            )

    elif analysis_name == 'contact':
        _populate_visual_feature_combo(
            self.contact_visual_feature_combo,
            result,
        )
        self.contact_visual_feature_combo.setEnabled(True)
        self.btn_show_contact_feature.setEnabled(True)

    elif analysis_name == 'clustering':
        visualizable = result.get('visualizable_features', []) or []
        if visualizable:
            feature = visualizable[0].get('feature', 'cluster_id')
        else:
            feature = 'cluster_id'

        _show_feature_name_from_result(
            self,
            result_attr='clustering_result',
            feature=feature,
            layer_prefix='cluster',
        )

    _show_single_morphometry_result_dialog(self, analysis_name, result)


def _show_single_morphometry_result_dialog(self, analysis_name, result):
    dlg = QDialog(self)
    dlg.setWindowTitle(f'{analysis_name.capitalize()} Results')
    dlg.resize(760, 520)

    layout = QVBoxLayout(dlg)

    n_cells = result.get('n_cells', 0)
    out_dir = result.get('out_dir', '')

    summary = QLabel(
        f'Analysis: {analysis_name}\n'
        f'Cells analyzed: {n_cells}\n'
        f'Output folder:\n{out_dir}'
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    text = QTextEdit()
    text.setReadOnly(True)

    lines = []
    lines.append('Saved files:')

    for key in [
        'cell_csv',
        'region_csv',
        'summary_csv',
        'contact_csv',
        'feature_manifest',
        'parameter_json',
    ]:
        val = result.get(key, '')
        if val:
            lines.append(f'  {key}: {val}')

    lines.append('')
    lines.append('Summary:')
    for row in result.get('summary', [])[:30]:
        lines.append(f'  {row.get("metric")}: {row.get("value")}')

    text.setPlainText('\n'.join(lines))
    layout.addWidget(text)

    btn_row = QHBoxLayout()

    btn_open = QPushButton('Open Output Folder')
    btn_open.clicked.connect(lambda: _open_morphometry_output_folder(self))
    btn_row.addWidget(btn_open)

    btn_close = QPushButton('Close')
    btn_close.clicked.connect(dlg.accept)
    btn_row.addWidget(btn_close)

    layout.addLayout(btn_row)

    dlg.exec_()


def _populate_visual_feature_combo(combo: QComboBox, result: dict):
    combo.clear()

    visualizable = result.get('visualizable_features', []) or []
    if not visualizable:
        combo.setEnabled(False)
        return

    for row in visualizable:
        label = row.get('label', row.get('feature', 'feature'))
        feature = row.get('feature', '')
        if feature:
            combo.addItem(label, feature)


def _normalize_values_to_uint16(values):
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=np.uint16)

    valid = np.isfinite(values)
    if not np.any(valid):
        return out

    lo, hi = np.nanpercentile(values[valid], [1, 99])
    scaled = np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    out[valid] = (scaled[valid] * 65535).astype(np.uint16)
    return out


def _load_labels_as_numpy_for_visualization(labels_data):
    if hasattr(labels_data, 'compute'):
        return np.asarray(labels_data.compute())
    return np.asarray(labels_data)


def _show_feature_name_from_result(
    self,
    result_attr: str,
    feature: str,
    layer_prefix: str,
):
    result = getattr(self, result_attr, None)
    if not result:
        QMessageBox.information(
            self,
            'No result',
            'Please run the analysis first.',
        )
        return

    if not feature:
        QMessageBox.information(
            self,
            'No feature',
            'No visualizable feature was returned.',
        )
        return

    csv_path = result.get('cell_csv', '')
    if not csv_path or not Path(csv_path).exists():
        QMessageBox.warning(
            self,
            'Missing CSV',
            'The result CSV file cannot be found.',
        )
        return

    labels_layer = None
    if hasattr(self, '_get_segmentation_labels_layer'):
        labels_layer = self._get_segmentation_labels_layer()
    if labels_layer is None and hasattr(self, '_get_labels_layer'):
        labels_layer = self._get_labels_layer()

    if labels_layer is None:
        QMessageBox.warning(
            self,
            'No label layer',
            'Please load a label layer first.',
        )
        return

    try:
        df = pd.read_csv(csv_path)

        if 'label_id' not in df.columns:
            QMessageBox.warning(
                self,
                'Invalid CSV',
                'The result CSV does not contain label_id.',
            )
            return

        if feature not in df.columns:
            QMessageBox.warning(
                self,
                'Feature not found',
                f"The feature '{feature}' is not in the result CSV.",
            )
            return

        label_arr = _load_labels_as_numpy_for_visualization(labels_layer.data)

        if not np.issubdtype(label_arr.dtype, np.integer):
            label_arr = label_arr.astype(np.int64)

        max_label = int(np.max(label_arr))
        lut = np.zeros(max_label + 1, dtype=np.uint16)

        label_ids = df['label_id'].to_numpy(int)
        values = df[feature].to_numpy(float)

        mapped_values = _normalize_values_to_uint16(values)

        valid = (label_ids >= 0) & (label_ids <= max_label)
        lut[label_ids[valid]] = mapped_values[valid]

        mapped = lut[label_arr]

        layer_name = f'{layer_prefix}_{feature}'
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

        layer = self.viewer.add_image(
            mapped,
            name=layer_name,
            visible=True,
            blending='translucent',
        )

        with suppress(Exception):
            layer.colormap = 'viridis'

        notifications.show_info(f'Displayed feature: {feature}')

    except Exception as e:
        QMessageBox.critical(
            self,
            'Visualization Error',
            f'Failed to visualize feature:\n{e}',
        )


def _show_feature_from_result(
    self, result_attr: str, combo: QComboBox, layer_prefix: str
):
    feature = combo.currentData()
    _show_feature_name_from_result(
        self,
        result_attr=result_attr,
        feature=feature,
        layer_prefix=layer_prefix,
    )


def _refresh_morphometry_raw_layers(self):
    """
    Refresh raw image layer combo box used by intensity analysis.
    """
    if not hasattr(self, 'intensity_raw_layer_combo'):
        return

    current = self.intensity_raw_layer_combo.currentText()

    self.intensity_raw_layer_combo.clear()

    image_layers = [
        layer for layer in self.viewer.layers if isinstance(layer, Image)
    ]

    for layer in image_layers:
        self.intensity_raw_layer_combo.addItem(layer.name)

    if current:
        idx = self.intensity_raw_layer_combo.findText(current)
        if idx >= 0:
            self.intensity_raw_layer_combo.setCurrentIndex(idx)


def _refresh_basic_raw_layers(self):
    """
    Refresh raw image layer combo box used by Basic Information.
    """
    if not hasattr(self, 'basic_raw_layer_combo'):
        return

    current = self.basic_raw_layer_combo.currentText()
    self.basic_raw_layer_combo.clear()

    image_layers = [
        layer for layer in self.viewer.layers if isinstance(layer, Image)
    ]

    for layer in image_layers:
        self.basic_raw_layer_combo.addItem(layer.name)

    if current:
        idx = self.basic_raw_layer_combo.findText(current)
        if idx >= 0:
            self.basic_raw_layer_combo.setCurrentIndex(idx)


def _basic_info_requires_raw(self):
    features = _collect_basic_info_config(self).get('selected_features', [])
    return any(_feature_requires_raw(f) for f in features)


def _load_morphometry_maps_to_viewer(self, result):
    maps = result.get('feature_maps', []) or []
    if not maps:
        return

    loaded = 0
    for row in maps:
        path = row.get('path', '')
        feature = row.get('feature', 'feature')
        if not path or not Path(path).exists():
            continue

        try:
            arr = tifffile.imread(path)
            group = row.get('group', 'morph')
            layer_name = f'morph_{group}_{feature}'
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

            layer = self.viewer.add_image(arr, name=layer_name, visible=False)
            try:
                layer.colormap = 'viridis'
                layer.blending = 'translucent'
            except Exception:
                pass
            loaded += 1
        except Exception as e:
            notifications.show_warning(
                f'Failed to load feature map {feature}: {e}'
            )

    if loaded > 0:
        notifications.show_info(f'Loaded {loaded} morphometry feature map(s).')
