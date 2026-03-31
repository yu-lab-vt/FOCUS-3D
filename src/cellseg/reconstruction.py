import dask.array as da
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes


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
