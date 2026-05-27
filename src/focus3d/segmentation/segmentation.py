import traceback

from qtpy.QtCore import QObject, Signal


class SegmentationWorker(QObject):
    """Worker for running 3D auto-segmentation in a background thread."""

    progress = Signal(int, str)  # progress value, message
    finished = Signal(object, str)  # result, error message
    cancelled = Signal()

    def __init__(
        self,
        image_data,
        checkpoint=None,
        seg_params=None,
    ):
        super().__init__()
        self.image_data = image_data
        self.checkpoint = checkpoint
        self.seg_params = seg_params or {}
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _progress_callback(self, value, message):
        self.progress.emit(int(value), str(message))

    def _is_cancelled_callback(self):
        return self._is_cancelled

    def run(self):
        try:
            self.progress.emit(-1, 'Preparing segmentation...')

            config_file = self.seg_params.get('config_file', '').strip()
            if not config_file:
                raise ValueError('config_file is empty.')

            if not self.checkpoint:
                raise ValueError('checkpoint is empty.')

            output_dir = self.seg_params.get('save_path', '').strip()
            if not output_dir:
                raise ValueError('save_path is empty.')

            if self._is_cancelled:
                self.cancelled.emit()
                return

            try:
                run_mask2former_inference = _load_mask2former_inference()
            except Exception as e:
                self.finished.emit(None, str(e))
                return

            size_filter_max_size = self.seg_params.get(
                'size_filter_max_size',
                100000,
            )

            if size_filter_max_size in [None, '']:
                size_filter_max_size = None
            else:
                size_filter_max_size = int(size_filter_max_size)

            result = run_mask2former_inference(
                image_data=self.image_data,
                image_path=self.seg_params.get('image_path', None),
                input_name=self.seg_params.get('input_name', 'napari_layer'),
                config_file=config_file,
                weights_path=self.checkpoint,
                output_dir=output_dir,
                cuda_visible_devices=self.seg_params.get(
                    'cuda_visible_devices', None
                ),
                device=self.seg_params.get('device', None),
                z_ratio=float(self.seg_params.get('z_ratio', 1.0)),
                lower_percentile=float(
                    self.seg_params.get('lower_percentile', 1.0)
                ),
                upper_percentile=float(
                    self.seg_params.get('upper_percentile', 99.0)
                ),
                background_threshold=float(
                    self.seg_params.get('background_threshold', 20.0)
                ),
                batch_size=int(self.seg_params.get('batch_size', 1)),
                data_loader_num_workers=int(
                    self.seg_params.get('data_loader_num_workers', 0)
                ),
                save_intermediate=bool(
                    self.seg_params.get('save_intermediate', False)
                ),
                score_thresh=float(self.seg_params.get('score_thresh', 0.7)),
                mask_thresh=float(self.seg_params.get('mask_thresh', 0.5)),
                topk_postprocess=int(
                    self.seg_params.get('topk_postprocess', 300)
                ),
                min_edge_area=int(self.seg_params.get('min_edge_area', 20)),
                size_filter_min_size=int(
                    self.seg_params.get('size_filter_min_size', 0)
                ),
                size_filter_max_size=size_filter_max_size,
                use_amp=bool(self.seg_params.get('use_amp', True)),
                amp_dtype=str(self.seg_params.get('amp_dtype', 'float16')),
                patch_size=tuple(
                    self.seg_params.get('patch_size', [32, 96, 96])
                ),
                stride=tuple(self.seg_params.get('stride', [24, 64, 64])),
                # New: send inference progress back to Qt worker.
                progress_callback=self._progress_callback,
                cancel_callback=self._is_cancelled_callback,
            )

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress.emit(-1, 'Segmentation completed.')
            self.finished.emit(result, '')

        except RuntimeError as e:
            if str(e) == '__SEG_CANCELLED__':
                self.cancelled.emit()
                return
            self.finished.emit(
                None, f'Segmentation error: {e}\n{traceback.format_exc()}'
            )

        except Exception as e:
            self.finished.emit(
                None, f'Segmentation error: {e}\n{traceback.format_exc()}'
            )


def _load_mask2former_inference():
    """
    Lazily import the optional Mask2Former backend.

    This keeps napari UI and non-Mask2Former functions usable even when
    the Mask2Former code/package is not installed.
    """
    try:
        from focus3d.segmentation.mask2former_adapter import (
            run_mask2former_inference,
        )

        return run_mask2former_inference

    except ModuleNotFoundError as e:
        raise RuntimeError(
            'Mask2Former backend is not available.\n\n'
            'This only affects Auto-segmentation.\n'
            'Other functions such as loading labels, manual curation, display settings, '
            'patch selection, and saving should still work.\n\n'
            'Missing module:\n'
            f'{e}'
        ) from e

    except Exception as e:
        raise RuntimeError(
            'Failed to initialize Mask2Former backend.\n\n'
            'This usually means that some optional deep-learning dependencies '
            'such as torch/detectron2/custom Mask2Former code are missing or incompatible.\n\n'
            f'Original error:\n{e}'
        ) from e
