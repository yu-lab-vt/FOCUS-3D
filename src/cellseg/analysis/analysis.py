from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import traceback
from contextlib import suppress

import dask.array as da
import napari
import numpy as np
from dask import compute, delayed
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
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

        try:
            self.viewer.reset_view()
        except Exception:
            pass

    else:
        self.viewer.dims.ndisplay = 2
        self.btn_toggle_full_3d_view.setText('Switch to 3D View')

        try:
            self.viewer.reset_view()
        except Exception:
            pass


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
            try:
                labels_layer.mode = 'pan_zoom'
            except Exception:
                pass

        self._current_roi_mode = None
        self._new_label_mode_active = False
        self._delete_inside_mode_active = False
        self._delete_all_active = False
        self._pick_mode_active = False

        if hasattr(self, 'btn_pick_mode'):
            self.btn_pick_mode.setText('Enter Curation Mode')

        try:
            self._update_curation_controls()
        except Exception:
            pass

    for layer in self.viewer.layers:
        if hasattr(layer, 'mode'):
            try:
                layer.mode = 'pan_zoom'
            except Exception:
                pass

    for layer in self.viewer.layers:
        if layer.__class__.__name__ == 'Image':
            try:
                self.viewer.layers.selection.active = layer
            except Exception:
                pass
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

        try:
            layer.refresh()
        except Exception:
            pass


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
