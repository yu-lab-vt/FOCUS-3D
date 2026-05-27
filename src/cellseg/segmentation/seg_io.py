from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import csv
import traceback
from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
import zarr
from napari.utils import notifications
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QLineEdit, QMessageBox, QProgressDialog
from skimage import io
from zarr.convenience import copy_store


class SaveWorker(QObject):
    """Worker for saving labels in the background."""

    progress = Signal(int, str)
    finished = Signal(str)  # success message or empty if error
    error = Signal(str)

    def __init__(self, labels_data, save_dir, save_format, renumber):
        super().__init__()
        self.labels_data = labels_data
        self.save_dir = save_dir
        self.save_format = save_format  # 'TIFF' or 'Zarr', 2 formats
        self.renumber = renumber
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(0, 'Preparing data...')
            if self.renumber:
                # Renumbering (needs to be done in worker)
                data = self._renumber_labels(self.labels_data)
            else:
                data = self.labels_data

            if self._is_cancelled:
                self.finished.emit('')
                return

            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = os.path.join(self.save_dir, f'labels_{timestamp}')

            if self.save_format.startswith('TIFF'):
                self.progress.emit(30, 'Saving as TIFF...')
                filename = base + '.tif'
                if isinstance(data, da.Array):
                    data = data.compute()
                with tifffile.TiffWriter(filename, bigtiff=True) as tif:
                    tif.write(
                        data, photometric='minisblack', compression='zlib'
                    )
                self.progress.emit(100, 'TIFF saved.')
                self.finished.emit(f'Saved TIFF to {filename}')
            else:  # Zarr
                self.progress.emit(30, 'Saving as Zarr...')
                store_path = base + '.zarr'
                if isinstance(data, da.Array):
                    data.to_zarr(store_path, compute=True)
                elif isinstance(data, zarr.Array):
                    dest_store = zarr.DirectoryStore(store_path)
                    copy_store(data.store, dest_store)
                elif isinstance(data, np.ndarray):
                    da.from_array(data, chunks='auto').to_zarr(
                        store_path, compute=True
                    )
                else:
                    raise TypeError(
                        f'Unsupported data type for Zarr: {type(data)}'
                    )
                self.progress.emit(100, 'Zarr saved.')
                self.finished.emit(f'Saved Zarr to {store_path}')
        except (ValueError, OSError) as e:
            self.error.emit(f'Save error: {e}\n{traceback.format_exc()}')

    def _renumber_labels(self, data):
        """
        Renumber labels in the given data to consecutive integers starting from 1.
        Returns a new array (numpy, dask, or zarr) with renumbered labels.
        """
        # If data is a dask array, we can compute unique labels in a lazy way
        if isinstance(data, da.Array):
            # Get unique labels (excluding 0) using dask
            unique_labels = da.unique(data)
            # Filter out background (0) – note: unique_labels is a dask array
            # We need to compute it to build the mapping, but this may be heavy for large data.
            # For large data, renumbering is not trivial without loading all labels.
            # Here we compute unique labels; for very large arrays this might still be slow.
            unique_labels = unique_labels[unique_labels != 0].compute()
            if len(unique_labels) == 0:
                return data  # nothing to renumber

            # Build mapping old -> new (1..N)
            mapping = {
                old: i + 1 for i, old in enumerate(sorted(unique_labels))
            }

            # Apply mapping: we need a function to remap each chunk
            def remap_chunk(block):
                # block is a numpy array chunk
                new_block = np.zeros_like(block)
                for old, new in mapping.items():
                    new_block[block == old] = new
                return new_block

            # Use map_blocks to apply remapping chunkwise
            new_data = data.map_blocks(remap_chunk, dtype=data.dtype)
            return new_data

        else:  # numpy array (or other in-memory)
            # Convert to numpy if not already
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels != 0]
            if len(unique_labels) == 0:
                return data

            mapping = {
                old: i + 1 for i, old in enumerate(sorted(unique_labels))
            }
            new_data = np.zeros_like(data)
            for old, new in mapping.items():
                new_data[data == old] = new
            return new_data


# ---------- Helper methods ----------
def _browse_folder(self):
    folder = QFileDialog.getExistingDirectory(self, 'Select Save Directory')
    if folder:
        self.path_edit.setText(folder)


def _browse_folder_for_lineedit(self, line_edit: QLineEdit):
    """Open a directory dialog and write the selected path into the given line edit."""
    folder = QFileDialog.getExistingDirectory(self, 'Select Save Directory')
    if folder:
        line_edit.setText(folder)


def _browse_checkpoint(self):
    """Open file dialog to select a checkpoint file."""
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        'Select Checkpoint File',
        '',
        'Checkpoint files (*.pth *.pt *.ckpt);;All files (*)',
    )
    if file_path:
        self.checkpoint_edit.setText(file_path)


def _init_session_log_file(self, label_path: str):
    """Create a new CSV log file for the currently loaded label."""
    label_path = Path(label_path)
    self.current_label_path = label_path

    # Special handling for .zarr: save log beside the .zarr folder
    if label_path.suffix.lower() == '.zarr':
        self.log_dir = label_path.parent
        base_name = label_path.stem
    elif label_path.is_dir():
        self.log_dir = label_path
        base_name = label_path.name
    else:
        self.log_dir = label_path.parent
        base_name = label_path.stem

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    self.log_file_path = (
        self.log_dir / f'{base_name}_curation_log_{timestamp}.csv'
    )

    with open(
        self.log_file_path, mode='w', newline='', encoding='utf-8-sig'
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'timestamp',
                'operation',
                'label_id_or_count',
                'z_index',
                'layer_name',
                'note',
            ]
        )


def _append_log_entry(
    self,
    operation: str,
    label_id_or_count=None,
    z_index=None,
    layer_name=None,
    note: str = '',
):
    """Append one operation record directly to the session CSV log."""
    try:
        if self.log_file_path is None:
            notifications.show_warning(
                'No active curation log file. Please load a label first.'
            )
            return

        if layer_name is None:
            labels_layer = self._get_labels_layer()
            layer_name = labels_layer.name if labels_layer is not None else ''

        with open(
            self.log_file_path, mode='a', newline='', encoding='utf-8-sig'
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    operation,
                    '' if label_id_or_count is None else label_id_or_count,
                    '' if z_index is None else z_index,
                    layer_name,
                    note,
                ]
            )

    except Exception as e:
        notifications.show_warning(f'Failed to write log entry: {e}')


def _show_log_file_location(self):
    """Show the current log file location."""
    if self.log_file_path is None:
        QMessageBox.information(
            self,
            'Curation Log',
            'No active log file yet. Please load a label first.',
        )
        return

    QMessageBox.information(
        self,
        'Log File',
        f'Log file:\n{self.log_file_path}',
    )


def _show_segmentation_tif(self):
    filepath, _ = QFileDialog.getOpenFileName(
        self,
        'Select Segmentation File',
        self.path_edit.text(),
        'Supported files (*.tif *.tiff *.zarr)',
    )
    if filepath:
        if filepath.endswith('.zarr'):
            data = da.from_zarr(filepath)
            layer = self.viewer.add_labels(
                data, name=os.path.basename(filepath)
            )
            self._ensure_next_label_id(layer)
            layer.contour = False
            layer.edge_color = 'white'
            layer.edge_width = 1
            layer.opacity = 0.8
        else:
            data = io.imread(filepath)
            layer = self.viewer.add_labels(
                data, name=os.path.basename(filepath)
            )
            self._ensure_next_label_id(layer)
            layer.contour = False
            layer.edge_color = 'white'
            layer.edge_width = 1
            layer.opacity = 0.8
        self._init_session_log_file(filepath)
        self._append_log_entry(
            operation='load label',
            label_id_or_count='',
            z_index='',
            layer_name=layer.name,
            note=f'Loaded label from {filepath}',
        )


def _load_zarr_folder(self):
    """Open a directory dialog to select a Zarr folder and load it."""
    folder = QFileDialog.getExistingDirectory(
        self,
        'Select Segmentation File',
        self.path_edit.text(),
        QFileDialog.ShowDirsOnly,
    )
    if folder:
        try:
            # data = da.from_zarr(folder)
            if os.access(folder, os.W_OK):
                data = zarr.open(folder, mode='r+')
            else:
                data = zarr.open(folder, mode='r')
                notifications.show_warning(
                    'Zarr is read-only. Modifications may not be saved.'
                )
            layer = self.viewer.add_labels(data, name=os.path.basename(folder))
            self._ensure_next_label_id(layer)
            layer.contour = False
            layer.edge_color = 'white'
            layer.edge_width = 1
            layer.opacity = 0.8

            # Create a new log file for this loaded label
            self._init_session_log_file(folder)

            # Optional: write one record indicating this label was loaded
            self._append_log_entry(
                operation='load label',
                label_id_or_count='',
                z_index='',
                layer_name=layer.name,
                note=f'Loaded Zarr label from {folder}',
            )

            notifications.show_info(
                f'Loaded Zarr from {folder}\nCuration log: {self.log_file_path}'
            )

        except (ValueError, OSError) as e:
            notifications.show_error(f'Failed to load Zarr: {e}')


def _save_current_labels(self):
    """Save labels in background."""
    labels_layer = self._get_labels_layer()
    if labels_layer is None:
        return

    data = labels_layer.data
    renumber = self.chk_renumber.isChecked()
    save_format = self.save_format_combo.currentText()
    save_dir = self.path_edit.text()

    # Progress dialog
    self.save_progress = QProgressDialog('Saving...', 'Cancel', 0, 100, self)
    self.save_progress.setWindowTitle('Save Progress')
    self.save_progress.setAutoClose(True)
    self.save_progress.show()

    # Worker
    self.save_thread = QThread()
    self.save_worker = SaveWorker(data, save_dir, save_format, renumber)
    self.save_worker.moveToThread(self.save_thread)

    self.save_worker.progress.connect(
        lambda val, msg: self.save_progress.setLabelText(msg)
    )
    self.save_worker.progress.connect(
        lambda val, msg: self.save_progress.setValue(val)
    )
    self.save_worker.finished.connect(self._on_save_finished)
    self.save_worker.error.connect(self._on_save_error)
    self.save_progress.canceled.connect(self.save_worker.cancel)
    self.save_progress.canceled.connect(self.save_thread.quit)
    self.save_thread.started.connect(self.save_worker.run)
    self.save_worker.finished.connect(self.save_thread.quit)
    self.save_worker.finished.connect(self.save_worker.deleteLater)
    self.save_thread.finished.connect(self.save_thread.deleteLater)

    self.save_thread.start()


def _on_save_finished(self, message):
    self.save_progress.close()
    if message:
        notifications.show_info(message)


def _on_save_error(self, error_msg):
    self.save_progress.close()
    notifications.show_error(error_msg)


def _save_labels(self, labels: np.ndarray, base_filename: str):
    save_dir = self.path_edit.text()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, base_filename)
    io.imsave(filepath, labels.astype(np.uint16))
    notifications.show_info(f'Saved to {filepath}')


def _save_labels_tiff(self, data):
    save_dir = self.path_edit.text()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(save_dir, f'labels_{timestamp}.tif')
    if isinstance(data, da.Array):
        data = data.compute()
    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        tif.write(data, photometric='minisblack', compression='zlib')
    notifications.show_info(f'Saved TIFF to {filename}')


def _save_labels_zarr(self, data):
    save_dir = self.path_edit.text()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    store_path = os.path.join(save_dir, f'labels_{timestamp}.zarr')
    if isinstance(data, da.Array):
        data.to_zarr(store_path, compute=True)
        notifications.show_info(f'Saved dask array to Zarr: {store_path}')

    elif isinstance(data, zarr.Array):
        dest_store = zarr.DirectoryStore(store_path)
        copy_store(data.store, dest_store)
        notifications.show_info(f'Copied Zarr store to: {store_path}')

    elif isinstance(data, np.ndarray):
        da.from_array(data, chunks='auto').to_zarr(store_path, compute=True)
        notifications.show_info(f'Saved numpy array to Zarr: {store_path}')

        try:
            arr = np.asarray(data)
            da.from_array(arr, chunks='auto').to_zarr(store_path, compute=True)
            notifications.show_info(
                f'Converted and saved to Zarr: {store_path}'
            )
        except (ValueError, OSError) as e:
            notifications.show_error(
                f'Unsupported data type for Zarr save: {type(data)} - {e}'
            )
            return


def _export_log(self):
    """Show the current session log file location."""
    self._show_log_file_location()
