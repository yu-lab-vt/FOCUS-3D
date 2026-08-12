import json
import math
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import find_objects
from skimage.measure import label as sk_label
from tqdm.auto import tqdm

from .mask2former.config_win import get_cfg
from .mask2former.maskformer_model_win import (
    build_maskformer_model_from_cfg,
)

def patch_collate_fn(samples):
    images = torch.stack(
        [
            sample["image"]
            for sample in samples
        ],
        dim=0,
    )

    coords = [
        sample["coord"]
        for sample in samples
    ]

    return images, coords

def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return 'estimating...'

    seconds = int(seconds)

    if seconds < 60:
        return f'{seconds}s'

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m {sec}s'

    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes}m'


def _emit_status(
    progress_callback: Callable[[int, str], None] | None,
    message: str,
) -> None:
    """
    Emit status-only message.

    Convention:
        value = -1 means: update text only, do not update progress bar.
    """
    if progress_callback is None:
        return

    progress_callback(-1, str(message))


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    value: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    value = max(0, min(100, int(value)))
    progress_callback(value, str(message))


def _check_cancelled(cancel_callback: Callable[[], bool] | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError('__SEG_CANCELLED__')


# ============================================================
# Windows no-Detectron2 model loading
# ============================================================


def _extract_model_state_dict(checkpoint):
    """
    Extract state_dict from common Detectron2 / PyTorch checkpoint formats.
    """
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            return checkpoint['model']
        if 'state_dict' in checkpoint and isinstance(
            checkpoint['state_dict'], dict
        ):
            return checkpoint['state_dict']
        if 'model_state_dict' in checkpoint and isinstance(
            checkpoint['model_state_dict'], dict
        ):
            return checkpoint['model_state_dict']

    return checkpoint


def _clean_state_dict_keys(state_dict):
    """
    Remove common wrappers from checkpoint keys.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.') :]
        if k.startswith('model.'):
            k = k[len('model.') :]
        cleaned[k] = v
    return cleaned


def build_predictor(cfg):
    model = build_maskformer_model_from_cfg(cfg)

    print(f'Loading weights from: {cfg.MODEL.WEIGHTS}')
    checkpoint = torch.load(
        cfg.MODEL.WEIGHTS, map_location='cpu', weights_only=False
    )
    state_dict = _extract_model_state_dict(checkpoint)
    state_dict = _clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # print(f'[Windows no-D2] Missing keys: {len(missing)}')
    # if missing:
    #     print('  First missing keys:', missing[:20])

    # print(f'[Windows no-D2] Unexpected keys: {len(unexpected)}')
    # if unexpected:
    #     print('  First unexpected keys:', unexpected[:20])

    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model

def resolve_inference_device(device: str | None = None) -> str:
    """
    Resolve inference device to an explicit torch device string.

    Returns:
        cuda:0 / cuda:1 / ... / cpu
    """
    if not torch.cuda.is_available():
        return 'cpu'

    if device is None or str(device).strip() == '':
        return 'cuda:0'

    device = str(device).strip()

    if device == 'cuda':
        return 'cuda:0'

    torch_device = torch.device(device)

    if torch_device.type != 'cuda':
        return str(torch_device)

    dev_index = torch_device.index
    if dev_index is None:
        dev_index = 0

    n_cuda = torch.cuda.device_count()
    if dev_index < 0 or dev_index >= n_cuda:
        raise RuntimeError(
            f'Requested device cuda:{dev_index}, but PyTorch only sees '
            f'{n_cuda} CUDA device(s). Please check the GPU selector.'
        )

    return f'cuda:{dev_index}'


def activate_inference_device(device: str | None = None) -> str:
    """
    Set selected CUDA device before model construction.
    """
    device = resolve_inference_device(device)

    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch_device = torch.device(device)
        torch.cuda.set_device(torch_device)

        print(f'[FOCUS3D inference] requested device: {device}')
        print(
            f'[FOCUS3D inference] current cuda device: '
            f'cuda:{torch.cuda.current_device()}'
        )
        print(
            f'[FOCUS3D inference] GPU name: '
            f'{torch.cuda.get_device_name(torch_device.index)}'
        )
    else:
        print('[FOCUS3D inference] using CPU')

    return device

def setup_cfg(config_file, weights_path, device: str | None = None):
    """
    Windows no-Detectron2 config setup.

    Equivalent to Linux:
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.set_new_allowed(True)
        cfg.INPUT.IMAGE_SIZE = [32, 96, 96]
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_path

    The only intentional difference is that Windows uses config_win and
    maskformer_model_win to avoid Detectron2.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)

    cfg.INPUT.IMAGE_SIZE = [32, 96, 96]
    cfg.merge_from_file(config_file)

    device = resolve_inference_device(device)

    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device

    # inference_win.py is inference-only.
    # Future Windows fine-tuning should use train_win.py and set this True.
    cfg.MODEL.MASK_FORMER.BUILD_CRITERION = False

    if not hasattr(cfg, 'TEST'):
        cfg.TEST = type(cfg)()

    if not hasattr(cfg.TEST, 'DETECTIONS_PER_IMAGE'):
        cfg.TEST.DETECTIONS_PER_IMAGE = (
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        )

    return cfg


# ============================================================
# Main inference function
# ============================================================


def infer_volume(
    image_path: str | Path,
    config_file: str | Path,
    weights_path: str | Path,
    device: str | None = None,
    output_dir: str | Path | None = None,
    z_ratio: float = 1.0,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    background_threshold: float | None = None,
    batch_size: int = 1,
    data_loader_num_workers: int = 4,
    save_intermediate: bool = False,
    # Optional default patch-level thresholds
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    topk_postprocess: int = 300,
    # Exposed patch_postprocess parameters
    min_edge_area: int = 20,
    size_filter_min_size: int = 0,
    size_filter_max_size: int | None = None,
    use_amp: bool = True,
    amp_dtype: str = 'float16',
    patch_size: tuple[int, int, int] = (32, 96, 96),  # (Z, Y, X)
    stride: tuple[int, int, int] | None = None,
    # Inference-time cell-radius normalization
    cell_radius: float = 15.0,
    restore_resampled_mask: bool = True,
    save_resampled_image: bool = False,
    # Stitch parameters, kept aligned with Linux inference.py
    stitch_face_iom_thresh: float = 0.2,
    stitch_min_core_voxels: int = 64,
    stitch_min_contact_cc_pixels: int = 10,
    stitch_min_contact_cc_ratio: float = 0.5,
    stitch_contact_connectivity: int = 2,
    # Windows plugin callbacks
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict:
    """
    Inference on an arbitrarily-sized 3D volume:
        1) Read image
        2) Resample the image according to cell radius and z_ratio
        3) Normalize to [0, 1]
        4) Pad and generate sliding windows
        5) Skip pure-background patches
        6) Run batch inference
        7) Postprocess patch predictions and stitch them
        8) Build final instance/confidence maps
        9) Optionally restore predictions to the original raw shape
        10) Save outputs

    Algorithmic behavior is aligned with the Linux inference.py. Windows-only
    differences are model construction/loading and plugin callbacks.
    """
    t_total_start = time.time()

    if output_dir is None:
        raise ValueError('output_dir must not be None')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit_status(progress_callback, 'Loading segmentation model...')
    _check_cancelled(cancel_callback)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    print('\n[Step 1] "Building config and loading model..."')
    _emit_progress(progress_callback, 0, 'Building config and loading model...')
    _check_cancelled(cancel_callback)

    device = activate_inference_device(device)

    _check_cancelled(cancel_callback)

    cfg = setup_cfg(config_file, weights_path, device=device)

    _check_cancelled(cancel_callback)

    model = build_predictor(cfg)
    # Enable mathematically safe pre-upsampling filtering
    model.fast_score_thresh = float(score_thresh)
    model.fast_mask_thresh = float(mask_thresh)

    _check_cancelled(cancel_callback)

    device = cfg.MODEL.DEVICE

    amp_enabled = bool(use_amp and str(device).startswith('cuda'))
    if amp_dtype == 'float16':
        autocast_dtype = torch.float16
    elif amp_dtype == 'bfloat16':
        autocast_dtype = torch.bfloat16
    else:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")

    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')

    if cell_radius <= 0:
        raise ValueError(
            f'cell_radius must be positive, got {cell_radius}'
        )

    if z_ratio <= 0:
        raise ValueError(
            f'z_ratio must be positive, got {z_ratio}'
        )

    log_info = {
        'image_path': str(image_path),
        'cell_radius': float(cell_radius),
        'reference_cell_radius': 15.0,
        'z_ratio': float(z_ratio),
        'lower_percentile': float(lower_percentile),
        'upper_percentile': float(upper_percentile),
        'background_threshold_user': (
            None
            if background_threshold is None
            else float(background_threshold)
        ),
        'batch_size': int(batch_size),
        'save_intermediate': bool(save_intermediate),
        'amp_enabled_from_cfg': bool(amp_enabled),
        'patch_postprocess_params': {
            'score_thresh': float(score_thresh),
            'mask_thresh': float(mask_thresh),
            'min_edge_area': int(min_edge_area),
        },
        'final_size_filter_params': {
            'size_filter_min_size': int(size_filter_min_size),
            'size_filter_max_size': None
            if size_filter_max_size is None
            else int(size_filter_max_size),
        },
        'stitch_params': {
            'stitch_face_iom_thresh': float(stitch_face_iom_thresh),
            'stitch_min_core_voxels': int(stitch_min_core_voxels),
            'stitch_min_contact_cc_pixels': int(
                stitch_min_contact_cc_pixels
            ),
            'stitch_min_contact_cc_ratio': float(
                stitch_min_contact_cc_ratio
            ),
            'stitch_contact_connectivity': int(
                stitch_contact_connectivity
            ),
        },
    }

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------
    print('\n[Step 2] "Reading image..."')
    _emit_status(progress_callback, 'Reading image...')
    _check_cancelled(cancel_callback)

    raw_volume = read_volume(image_path)
    raw_shape_original = tuple(raw_volume.shape)

    _emit_status(progress_callback, 'Resampling image...')
    _check_cancelled(cancel_callback)

    infer_raw_volume, resample_info = resample_volume_for_inference(
        volume=raw_volume,
        cell_radius=cell_radius,
        z_ratio=z_ratio,
        reference_cell_radius=15.0,
        order=1,
    )

    infer_shape = tuple(infer_raw_volume.shape)
    log_info['resampling'] = resample_info

    if save_resampled_image and resample_info['enabled']:
        save_volume(
            output_dir / f'{Path(image_path).stem}_resampled_image.tif',
            infer_raw_volume.astype(np.float32),
        )
    _emit_status(progress_callback, 'Normalizing image...')
    _check_cancelled(cancel_callback)

    # ============================================================
    # Resolve stride FIRST
    # ============================================================

    if stride is None:
        stride = tuple(s // 2 for s in patch_size)
    else:
        stride = tuple(stride)

    patch_size = tuple(patch_size)


    # ============================================================
    # Compute global percentiles ONCE
    # ============================================================

    percentile_list = [
        float(lower_percentile),
        float(upper_percentile),
    ]

    if background_threshold is None:
        percentile_list.append(5.0)

    percentile_values = compute_percentiles_fast(
        infer_raw_volume,
        percentile_list,
        chunk_z=8,
    )

    p_low = float(percentile_values[0])
    p_high = float(percentile_values[1])

    norm_stats = {
        "p_low": p_low,
        "p_high": p_high,
    }


    # ============================================================
    # Resolve background threshold
    # ============================================================

    if background_threshold is None:
        bg_raw = float(percentile_values[2])
        background_threshold_source = "global_5th_percentile"
    else:
        bg_raw = float(background_threshold)
        background_threshold_source = "user"

    if p_high <= p_low:
        background_threshold_norm = 0.0
    else:
        background_threshold_norm = (
            (bg_raw - p_low)
            / (p_high - p_low)
        )

        background_threshold_norm = float(
            np.clip(
                background_threshold_norm,
                0.0,
                1.0,
            )
        )


    # ============================================================
    # Normalize + pad
    # ============================================================

    padded_volume, pad_info = normalize_and_pad_volume(
        volume=infer_raw_volume,
        p_low=p_low,
        p_high=p_high,
        patch_size=patch_size,
        stride=stride,
        chunk_z=8,
    )
    log_info['pad_info'] = pad_info

    if save_intermediate:
        save_volume(
            output_dir / 'padded_normalized_image.tif',
            padded_volume.astype(np.float32),
        )

    all_coords = generate_patch_coords(padded_volume.shape, patch_size, stride)
    max_z = max(c[0] for c in all_coords)
    max_y = max(c[1] for c in all_coords)
    max_x = max(c[2] for c in all_coords)
    max_coords = (max_z, max_y, max_x)

    infer_coords = []
    skipped_coords = []

    pz, py, px = patch_size
    _emit_status(progress_callback, 'Generating valid inference patches...')
    _check_cancelled(cancel_callback)

    if background_threshold_norm <= 0.0:
        infer_coords = all_coords
        skipped_coords = []

    else:
        infer_coords = []
        skipped_coords = []

        for idx, (z0, y0, x0) in enumerate(all_coords):

            if idx % 16 == 0:
                _check_cancelled(cancel_callback)

            patch = padded_volume[
                z0:z0+pz,
                y0:y0+py,
                x0:x0+px,
            ]

            # Cheap exact early-out
            if patch.max() < background_threshold_norm:
                skipped_coords.append(
                    (z0, y0, x0)
                )
                continue

            if (
                np.percentile(
                    patch,
                    99.9,
                )
                < background_threshold_norm
            ):
                skipped_coords.append(
                    (z0, y0, x0)
                )
            else:
                infer_coords.append(
                    (z0, y0, x0)
                )
    _check_cancelled(cancel_callback)

    log_info['num_infer_patches'] = int(len(infer_coords))
    log_info['num_skipped_background_patches'] = int(len(skipped_coords))

    # --------------------------------------------------------
    # Batch inference
    # --------------------------------------------------------
    print('\n[Step 4] "Batch inference..."')
    patch_dataset = PatchDataset(
        padded_volume=padded_volume,
        infer_coords=infer_coords,
        patch_size=patch_size,
    )
    t_infer_start = time.time()

    padded_shape = padded_volume.shape
    global_instance_map = np.zeros(padded_shape, dtype=np.uint32)
    confidence_sum = np.zeros(padded_shape, dtype=np.float32)
    confidence_count = np.zeros(padded_shape, dtype=np.uint16)
    next_instance_id = 1

    if save_intermediate:
        patch_output_dir = output_dir / 'patch_outputs'
        patch_output_dir.mkdir(parents=True, exist_ok=True)

    t_stitch_start = time.time()
    if data_loader_num_workers < 0:
        raise ValueError('data_loader_num_workers must be >= 0')
    if topk_postprocess < 1:
        raise ValueError('topk_postprocess must be >= 1')

    patch_loader = torch.utils.data.DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=True,
        persistent_workers=(data_loader_num_workers > 0),
        prefetch_factor=2 if data_loader_num_workers > 0 else None,
        collate_fn=patch_collate_fn,
    )

    patch_pbar = tqdm(
        patch_loader,
        total=len(patch_loader),
        desc='Inferencing patches',
        unit='batch',
        disable=(progress_callback is not None),
    )

    total_patches = len(infer_coords)
    total_batches = len(patch_loader)
    processed_patches = 0
    progress_update_every = max(1, total_patches // 100)
    t_patch_start = time.time()

    if total_patches == 0:
        _emit_progress(
            progress_callback,
            100,
            'No foreground patches found. Patch inference skipped.',
        )
    else:
        _emit_progress(
            progress_callback,
            0,
            f'Patch inference: 0/{total_patches} patches',
        )

    for batch_idx, (
        batch_images,
        batch_coords,
    ) in enumerate(patch_pbar):

        _check_cancelled(cancel_callback)

        batch_images = batch_images.to(
            device,
            non_blocking=True,
        )

        inputs = [
            {
                "image": batch_images[i],
                "coord": batch_coords[i],
            }
            for i in range(batch_images.shape[0])
        ]

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ),
        ):
            batch_outputs = model(inputs)

        _check_cancelled(cancel_callback)

        for sample, model_output in zip(inputs, batch_outputs, strict=False):
            _check_cancelled(cancel_callback)

            z0, y0, x0 = sample['coord']

            post = patch_postprocess_argmax(
                model_output=model_output,
                patch_shape=patch_size,
                score_thresh=score_thresh,
                mask_thresh=mask_thresh,
                min_voxels=min_edge_area,
                topk_postprocess=topk_postprocess,
            )

            patch_instance_map = post['patch_instance_map']
            patch_confidence = post['patch_confidence']

            next_instance_id = stitch_patch_instance_results(
                global_instance_map=global_instance_map,
                confidence_sum=confidence_sum,
                confidence_count=confidence_count,
                patch_instance_map=patch_instance_map,
                patch_confidence=patch_confidence,
                z0=z0,
                y0=y0,
                x0=x0,
                next_instance_id=next_instance_id,
                max_coords=max_coords,
                patch_size=patch_size,
                stride=stride,
                face_iom_thresh=stitch_face_iom_thresh,
                stitch_min_core_voxels=stitch_min_core_voxels,
                min_contact_cc_pixels=stitch_min_contact_cc_pixels,
                min_contact_cc_ratio=stitch_min_contact_cc_ratio,
                contact_connectivity=stitch_contact_connectivity,
            )

            processed_patches += 1

            if (
                processed_patches == 1
                or processed_patches % progress_update_every == 0
                or processed_patches == total_patches
            ):
                elapsed = time.time() - t_patch_start
                eta = (
                    elapsed
                    / max(1, processed_patches)
                    * max(0, total_patches - processed_patches)
                )

                percent = int(100 * processed_patches / max(1, total_patches))

                _emit_progress(
                    progress_callback,
                    percent,
                    (
                        f'Patch inference: '
                        f'{processed_patches}/{total_patches} patches '
                        f'| batch {batch_idx}/{total_batches} '
                        f'| ETA {_format_eta(eta)}'
                    ),
                )

                _check_cancelled(cancel_callback)

            if save_intermediate:
                patch_tag = f'z{z0:04d}_y{y0:04d}_x{x0:04d}'
                save_volume(
                    patch_output_dir
                    / f'{patch_tag}_patch_postprocess_instance.tif',
                    patch_instance_map.astype(np.uint16),
                )

    # ========================================================
    # Post-processing and saving
    # ========================================================
    print('\n[Step 5] "Post-processing and save results..."')
    _emit_status(progress_callback, 'Post-processing...')
    _check_cancelled(cancel_callback)

    infer_time_sec = time.time() - t_infer_start
    log_info['inference_time_sec'] = infer_time_sec

    if save_intermediate:
        save_volume(
            output_dir / 'merged_instance_padded.tif',
            global_instance_map.astype(np.uint32),
        )
        save_volume(
            output_dir / 'confidence_sum_padded.tif',
            confidence_sum.astype(np.float32),
        )
        save_volume(
            output_dir / 'confidence_count_padded.tif',
            confidence_count.astype(np.uint16),
        )

    log_info['online_postprocess_stitch_time_sec'] = (
        time.time() - t_stitch_start
    )

    instance_map_padded, confidence_map_padded = (
        finalize_instance_and_confidence(
            global_instance_map=global_instance_map,
            confidence_sum=confidence_sum,
            confidence_count=confidence_count,
        )
    )

    # Save result before final post-processing.
    z_inf, y_inf, x_inf = infer_shape
    if save_intermediate:
        pre_post_instance = (
            instance_map_padded[
                :z_inf,
                :y_inf,
                :x_inf,
            ]
        )

        pre_post_confidence = (
            confidence_map_padded[
                :z_inf,
                :y_inf,
                :x_inf,
            ]
        )
        save_volume(
            output_dir
            / f'{Path(image_path).stem}_instance_map_before_postprocess.tif',
            pre_post_instance.astype(np.uint32),
        )
        save_volume(
            output_dir
            / f'{Path(image_path).stem}_confidence_map_before_postprocess.tif',
            pre_post_confidence.astype(np.float32),
        )

    instance_map_padded, confidence_map_padded = filter_instances_by_size(
        instance_map=instance_map_padded,
        confidence_map=confidence_map_padded,
        min_area=min_edge_area,
        intensity_volume=padded_volume,
        min_size=size_filter_min_size,
        max_size=size_filter_max_size,
        background_threshold=background_threshold_norm,
        verbose=True,
    )

    # Crop back to inference image shape first.
    z_inf, y_inf, x_inf = infer_shape
    instance_map_infer = instance_map_padded[:z_inf, :y_inf, :x_inf]
    confidence_map_infer = confidence_map_padded[:z_inf, :y_inf, :x_inf]

    # Restore predictions to the original raw image shape after
    # inference-time resampling.
    if resample_info['enabled'] and restore_resampled_mask:
        instance_map = resize_volume_to_shape(
            instance_map_infer.astype(np.uint32, copy=False),
            target_shape=raw_shape_original,
            order=0,  # nearest neighbor for instance labels
        ).astype(np.uint32, copy=False)

        confidence_map = resize_volume_to_shape(
            confidence_map_infer.astype(np.float32, copy=False),
            target_shape=raw_shape_original,
            order=1,  # linear interpolation for confidence
        ).astype(np.float32, copy=False)
    else:
        instance_map = instance_map_infer.astype(
            np.uint32,
            copy=False,
        )
        confidence_map = confidence_map_infer.astype(
            np.float32,
            copy=False,
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    stem = Path(image_path).stem

    instance_path = output_dir / f'{stem}_instance_map.tif'
    if save_intermediate:
        confidence_path = output_dir / f'{stem}_confidence_map.tif'
        log_path = output_dir / f'{stem}_log.json'

    _check_cancelled(cancel_callback)
    save_volume(instance_path, instance_map.astype(np.uint32))
    if save_intermediate:
        save_volume(confidence_path, confidence_map.astype(np.float32))

        log_info['total_time_sec'] = time.time() - t_total_start

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_info, f, indent=2)

    result = {
        'instance_map_path': str(instance_path),
        'num_infer_patches': len(infer_coords),
        'num_skipped_background_patches': len(skipped_coords),
        'instance_map': instance_map,
        'confidence_map': confidence_map,
        'log_info': log_info,
    }

    _emit_progress(progress_callback, 100, 'Segmentation finished.')
    return result


# ============================================================
# Basic I/O
# ============================================================


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        padded_volume: np.ndarray,
        infer_coords: list[tuple[int, int, int]],
        patch_size: tuple[int, int, int],
    ):
        self.padded_volume = padded_volume
        self.infer_coords = infer_coords
        self.patch_size = patch_size

    def __len__(self):
        return len(self.infer_coords)

    def __getitem__(self, idx):
        z0, y0, x0 = self.infer_coords[idx]
        pz, py, px = self.patch_size

        patch = self.padded_volume[
            z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px
        ].astype(np.float32, copy=False)

        return {
            'image': torch.from_numpy(patch).unsqueeze(0),  # (1, Z, Y, X)
            'coord': (z0, y0, x0),
        }


def read_volume(image_path):
    image_path = str(image_path)
    suffix = Path(image_path).suffix.lower()

    if suffix in [".tif", ".tiff"]:
        vol = tifffile.imread(
            image_path,
            out="memmap",
        )

    elif suffix == ".zarr":
        vol = zarr.open(
            image_path,
            mode="r",
        )

    else:
        raise ValueError(
            f"Unsupported file format: {image_path}"
        )

    if isinstance(vol, np.ndarray):
        vol = np.squeeze(vol)

    if len(vol.shape) != 3:
        raise ValueError(
            f"Expected a 3D volume after squeeze, got shape {vol.shape}"
        )

    return vol


def save_volume(output_path: str | Path, volume: np.ndarray) -> None:
    """
    Save a volume either as .tif/.tiff or .zarr based on file extension.
    TIFF is saved with Z/Y/X axes metadata.
    """
    output_path = str(output_path)
    suffix = Path(output_path).suffix.lower()

    volume = np.asarray(volume)
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f'Expected 3D volume in Z/Y/X order, got shape={volume.shape}.'
        )

    if suffix in ['.tif', '.tiff']:
        tifffile.imwrite(
            output_path,
            volume,
            bigtiff=True,
            ome=True,
            metadata={'axes': 'ZYX'},
            photometric='minisblack',
            compression='zlib',
        )
    elif suffix == '.zarr':
        z = zarr.open(
            output_path,
            mode='w',
            shape=volume.shape,
            dtype=volume.dtype,
        )
        z[:] = volume
    else:
        raise ValueError(f'Unsupported output format: {output_path}')


def resize_volume_to_shape(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int,
) -> np.ndarray:
    """
    Resize 3D volume to exact target_shape.

    order:
        0: nearest neighbor, for instance labels
        1: linear interpolation, for raw/confidence
    """
    target_shape = tuple(int(s) for s in target_shape)
    if tuple(volume.shape) == target_shape:
        return volume

    zoom_factors = tuple(
        target_shape[i] / float(volume.shape[i]) for i in range(3)
    )

    resized = ndi.zoom(
        volume,
        zoom=zoom_factors,
        order=order,
        mode='nearest',
        prefilter=(order > 1),
    )

    # Safety correction: ndi.zoom usually gives exact shape, but protect
    # against rounding.
    out = np.zeros(target_shape, dtype=resized.dtype)
    z = min(target_shape[0], resized.shape[0])
    y = min(target_shape[1], resized.shape[1])
    x = min(target_shape[2], resized.shape[2])
    out[:z, :y, :x] = resized[:z, :y, :x]

    return out

def resample_volume_for_inference(
    volume: np.ndarray,
    cell_radius: float,
    z_ratio: float,
    reference_cell_radius: float = 15.0,
    order: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Resample a 3D volume so that the cell radius in the inference image
    is approximately equal to reference_cell_radius pixels.

    Parameters
    ----------
    volume:
        Input volume in (Z, Y, X).

    cell_radius:
        Estimated cell radius in pixels in the original XY image.

    z_ratio:
        Z-to-XY sampling ratio.

        If z_ratio > 1:
            Keep the Z dimension unchanged and resample only Y/X.

        If z_ratio <= 1:
            Apply the same resampling factor to Z/Y/X.

    reference_cell_radius:
        Cell radius expected by the model. Default is 15 pixels.

    order:
        Interpolation order. Use 1 for raw intensity images.

    Notes
    -----
    The factor is defined as:

        factor_xy = cell_radius / reference_cell_radius

    Target size is calculated as:

        target_size = original_size / factor

    Therefore:
        factor > 1 -> downsampling
        factor < 1 -> upsampling
        factor = 1 -> no resampling
    """
    if volume.ndim != 3:
        raise ValueError(
            f'volume must be a 3D array in (Z, Y, X), got {volume.shape}'
        )

    cell_radius = float(cell_radius)
    z_ratio = float(z_ratio)
    reference_cell_radius = float(reference_cell_radius)

    if cell_radius <= 0:
        raise ValueError(
            f'cell_radius must be positive, got {cell_radius}'
        )

    if z_ratio <= 0:
        raise ValueError(
            f'z_ratio must be positive, got {z_ratio}'
        )

    if reference_cell_radius <= 0:
        raise ValueError(
            'reference_cell_radius must be positive, '
            f'got {reference_cell_radius}'
        )

    # The original cell radius becomes approximately 15 pixels after:
    #
    #     new_radius = cell_radius / factor_xy
    #
    # Therefore:
    #
    #     factor_xy = cell_radius / 15
    factor_xy = cell_radius / reference_cell_radius

    # If Z sampling is coarser than XY, keep the Z dimension unchanged.
    if z_ratio > 1.0:
        factor_z = 1.0
        z_resampling_mode = 'keep_z'
    else:
        factor_z = factor_xy
        z_resampling_mode = 'same_as_xy'

    factors = (
        float(factor_z),
        float(factor_xy),
        float(factor_xy),
    )

    original_shape = tuple(int(s) for s in volume.shape)

    target_shape = tuple(
        max(
            1,
            int(round(original_shape[i] / factors[i])),
        )
        for i in range(3)
    )

    enabled = target_shape != original_shape

    info = {
        'enabled': bool(enabled),
        'cell_radius': float(cell_radius),
        'reference_cell_radius': float(reference_cell_radius),
        'z_ratio': float(z_ratio),
        'z_resampling_mode': z_resampling_mode,
        'factor_zyx': [float(f) for f in factors],
        'zoom_zyx': [
            float(target_shape[i]) / float(original_shape[i])
            for i in range(3)
        ],
        'original_shape': [int(s) for s in original_shape],
        'inference_shape': [int(s) for s in target_shape],
    }

    if not enabled:
        return volume, info

    resampled = resize_volume_to_shape(
        volume.astype(np.float32, copy=False),
        target_shape=target_shape,
        order=order,
    )

    info['inference_shape'] = [
        int(s) for s in resampled.shape
    ]

    return resampled.astype(np.float32, copy=False), info

# ============================================================
# Sliding-window planning and padding
# ============================================================


def compute_padded_length(length: int, patch: int, stride: int) -> int:
    """
    Compute the padded size so that the sliding windows fully cover the axis.

    Cases:
        - If length <= patch: pad up to patch
        - Otherwise: patch + n * stride, with the smallest n covering the full
          axis
    """
    if length <= patch:
        return patch

    n_steps = math.ceil((length - patch) / stride)
    return patch + n_steps * stride


def generate_patch_coords(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """
    Generate all patch start coordinates for a padded volume.
    """
    z, y, x = volume_shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    coords = []
    for z0 in range(0, z - pz + 1, sz):
        for y0 in range(0, y - py + 1, sy):
            for x0 in range(0, x - px + 1, sx):
                coords.append((z0, y0, x0))
    return coords


# ============================================================
# Patch postprocessing
# ============================================================

def patch_postprocess_argmax(
    model_output,
    patch_shape,
    score_thresh=0.5,
    mask_thresh=0.5,
    min_voxels=20,
    topk_postprocess=300,
 ):
    D, H, W = patch_shape

    patch_instance_map = np.zeros((D, H, W), dtype=np.uint16)
    patch_confidence = np.zeros((D, H, W), dtype=np.float32)

    if "pred_scores" not in model_output or "pred_masks" not in model_output:

        return {
            "patch_instance_map": patch_instance_map,
            "patch_confidence": patch_confidence,
        }

    # ------------------------------------------------------------
    # 0. Cheap query filtering
    # ------------------------------------------------------------
    scores = model_output["pred_scores"].detach().float()
    masks = model_output["pred_masks"].detach()

    if masks.numel() == 0 or scores.numel() == 0:
        return {
            "patch_instance_map": patch_instance_map,
            "patch_confidence": patch_confidence,
        }

    # ------------------------------------------------------------
    # Exact classification-score upper-bound filtering
    # ------------------------------------------------------------
    keep_cls = scores > score_thresh

    if not keep_cls.any().item():
        return {
            "patch_instance_map": patch_instance_map,
            "patch_confidence": patch_confidence,
        }

    scores = scores[keep_cls]
    masks = masks[keep_cls].float()

    # ------------------------------------------------------------
    # Mask logits/probability
    # ------------------------------------------------------------
    # if masks.min() < 0 or masks.max() > 1:
    #     mask_prob = masks.sigmoid()
    # else:
    mask_prob = masks

    # ------------------------------------------------------------
    # Exact empty-mask filtering
    # ------------------------------------------------------------
    mask_bin_tmp = mask_prob > mask_thresh

    has_foreground = mask_bin_tmp.flatten(1).any(dim=1)

    if not has_foreground.any().item():
        return {
            "patch_instance_map": patch_instance_map,
            "patch_confidence": patch_confidence,
        }

    scores = scores[has_foreground]
    mask_prob = mask_prob[has_foreground]
    mask_bin_tmp = mask_bin_tmp[has_foreground]

    # ------------------------------------------------------------
    # 1. Compute mask quality score
    # ------------------------------------------------------------
    mask_prob_flat = mask_prob.flatten(1)
    mask_bin_flat = mask_bin_tmp.flatten(1)

    mask_scores = (
        mask_prob_flat * mask_bin_flat
    ).sum(dim=1) / (
        mask_bin_flat.sum(dim=1) + 1e-6
    )

    final_scores = scores * mask_scores

    keep = (
        (final_scores > score_thresh)
        & (mask_scores > mask_thresh)
    )

    if not keep.any().item():
        return {
            "patch_instance_map": patch_instance_map,
            "patch_confidence": patch_confidence,
        }

    final_scores = final_scores[keep]
    mask_scores = mask_scores[keep]
    mask_prob = mask_prob[keep]
    mask_bin = mask_bin_tmp[keep]

    # ------------------------------------------------------------
    # 2. Sort queries by final score
    # ------------------------------------------------------------
    order = torch.argsort(
        final_scores,
        descending=True,
    )

    final_scores = final_scores[order]
    mask_scores = mask_scores[order]
    mask_prob = mask_prob[order]
    mask_bin = mask_bin[order]

    if (
        topk_postprocess is not None
        and mask_prob.shape[0] > topk_postprocess
    ):
        final_scores = final_scores[:topk_postprocess]
        mask_scores = mask_scores[:topk_postprocess]
        mask_prob = mask_prob[:topk_postprocess]
        mask_bin = mask_bin[:topk_postprocess]

    num_queries = mask_prob.shape[0]

    # ------------------------------------------------------------
    # 3. Conflict resolving before voxel-level argmax
    # ------------------------------------------------------------
    mask_prob_resolved = mask_prob.clone()

    conflict_min_pixels = 20
    merge_iom_thresh = 0.80
    merge_iou_thresh = 0.50

    def _safe_iou_bool(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = torch.logical_and(a, b).sum().item()
        union = torch.logical_or(a, b).sum().item()
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    if num_queries >= 2:
        overlap_count = mask_bin.sum(dim=0)
        conflict_voxel = overlap_count > 1
        conflict_voxel_np = conflict_voxel.cpu().numpy()

        for z in range(D):
            conflict_z = conflict_voxel_np[z]
            if not conflict_z.any():
                continue

            cc_map, num_cc = ndi.label(conflict_z)
            if num_cc == 0:
                continue

            for cc_id in range(1, num_cc + 1):
                cc_mask_np = (cc_map == cc_id)
                if int(cc_mask_np.sum()) < conflict_min_pixels:
                    continue

                cc_mask = torch.from_numpy(cc_mask_np).to(
                    device=mask_prob.device,
                    dtype=torch.bool,
                )

                participate = (
                    mask_bin[:, z] & cc_mask[None, :, :]
                ).flatten(1).any(dim=1)

                if participate.sum().item() <= 1:
                    continue

                participant_ids = participate.nonzero(as_tuple=False).flatten()
                pids = [int(q.item()) for q in participant_ids]
                n = len(pids)

                # ----- Build merge groups among participant queries -----
                parent = list(range(n))

                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

                for i in range(n):
                    qi = pids[i]
                    mi = mask_bin[qi, z]
                    area_i = int(mi.sum().item())
                    if area_i <= 0:
                        continue

                    for j in range(i + 1, n):
                        qj = pids[j]
                        mj = mask_bin[qj, z]
                        area_j = int(mj.sum().item())
                        if area_j <= 0:
                            continue

                        inter = int(torch.logical_and(mi, mj).sum().item())
                        if inter < conflict_min_pixels:
                            continue

                        union_area = area_i + area_j - inter
                        iom = float(inter) / float(min(area_i, area_j) + 1e-6)
                        iou = float(inter) / float(union_area + 1e-6)

                        should_merge = (iom > merge_iom_thresh) or (iou > merge_iou_thresh)
                        if should_merge:
                            union(i, j)

                groups = {}
                for i, q in enumerate(pids):
                    r = find(i)
                    groups.setdefault(r, []).append(q)

                # ----- Resolve each merge group -----
                for group_qs in groups.values():
                    if len(group_qs) <= 1:
                        continue

                    merged_mask = torch.zeros(
                        (H, W),
                        device=mask_prob.device,
                        dtype=torch.bool,
                    )

                    for q in group_qs:
                        merged_mask |= mask_bin[q, z]

                    if int(merged_mask.sum().item()) < min_voxels:
                        continue

                    # Compare merged z-slice structure with each query's
                    # adjacent structures at z-1 and z+1.
                    best_q = None
                    best_adj_iou = -1.0
                    best_score_tie = -1.0

                    for q in group_qs:
                        adj_iou = 0.0

                        if z > 0:
                            adj_iou = max(
                                adj_iou,
                                _safe_iou_bool(merged_mask, mask_bin[q, z - 1]),
                            )

                        if z < D - 1:
                            adj_iou = max(
                                adj_iou,
                                _safe_iou_bool(merged_mask, mask_bin[q, z + 1]),
                            )

                        score_tie = float(final_scores[q].item())

                        if (
                            adj_iou > best_adj_iou
                            or (
                                abs(adj_iou - best_adj_iou) < 1e-6
                                and score_tie > best_score_tie
                            )
                        ):
                            best_adj_iou = adj_iou
                            best_score_tie = score_tie
                            best_q = q

                    if best_q is None:
                        continue

                    # Assign merged structure to best_q.
                    # Use the maximum probability among merged queries as
                    # the winner probability on the merged region.
                    group_probs = mask_prob_resolved[group_qs, z]
                    merged_prob = group_probs.max(dim=0).values

                    mask_prob_resolved[best_q, z][merged_mask] = merged_prob[merged_mask]
                    mask_bin[best_q, z][merged_mask] = True

                    # Suppress other queries only on this merged structure.
                    # Other non-overlapping parts of those queries remain unchanged.
                    for q in group_qs:
                        if q == best_q:
                            continue
                        mask_prob_resolved[q, z][merged_mask] = 0.0
                        mask_bin[q, z][merged_mask] = False

    # ------------------------------------------------------------
    # 4. Voxel-level argmax after conflict resolving
    # ------------------------------------------------------------
    voxel_scores = mask_prob_resolved * final_scores[:, None, None, None]

    best_score, best_idx = voxel_scores.max(dim=0)  # (D, H, W)

    # Foreground is determined by the selected query's resolved mask probability
    selected_mask_prob = torch.gather(
        mask_prob_resolved,
        dim=0,
        index=best_idx.unsqueeze(0)
    ).squeeze(0)

    fg = selected_mask_prob > mask_thresh

    best_idx_np = best_idx.cpu().numpy()
    best_score_np = best_score.cpu().numpy()
    fg_np = fg.cpu().numpy()

    # ------------------------------------------------------------
    # 5. Build final instance map
    # ------------------------------------------------------------
    new_label_id = 1

    for local_q in range(num_queries):
        inst_mask = fg_np & (best_idx_np == local_q)

        if inst_mask.sum() < min_voxels:
            continue

        patch_instance_map[inst_mask] = new_label_id
        patch_confidence[inst_mask] = best_score_np[inst_mask]

        new_label_id += 1

    # ------------------------------------------------------------
    # 6. Remove abnormal full-foreground patch
    # ------------------------------------------------------------
    foreground_ratio = float(np.count_nonzero(patch_instance_map)) / float(patch_instance_map.size)

    nonzero_ids = np.unique(patch_instance_map)
    nonzero_ids = nonzero_ids[nonzero_ids > 0]

    if nonzero_ids.size == 1 and foreground_ratio > 0.8:
        patch_instance_map[...] = 0
        patch_confidence[...] = 0.0

    return {
        "patch_instance_map": patch_instance_map,
        "patch_confidence": patch_confidence,
    }


# ============================================================
# Stitching
# ============================================================
def compute_percentiles_fast(
    volume,
    percentiles,
    chunk_z: int = 8,
):
    qs = np.asarray(
        percentiles,
        dtype=np.float64,
    )

    dtype = np.dtype(volume.dtype)

    if dtype == np.uint8:
        num_bins = 256
    elif dtype == np.uint16:
        num_bins = 65536
    else:
        # Fallback for float / unusual dtype.
        return np.percentile(volume, qs)

    hist = np.zeros(
        num_bins,
        dtype=np.int64,
    )

    z_size = volume.shape[0]

    for z0 in range(0, z_size, chunk_z):
        z1 = min(z0 + chunk_z, z_size)

        chunk = np.asarray(
            volume[z0:z1]
        )

        hist += np.bincount(
            chunk.reshape(-1),
            minlength=num_bins,
        )

    cdf = np.cumsum(hist)
    n = int(cdf[-1])

    if n <= 0:
        return np.zeros_like(qs)

    results = []

    for q in qs:
        rank = (
            (n - 1)
            * float(q)
            / 100.0
        )

        rank_low = int(math.floor(rank))
        rank_high = int(math.ceil(rank))
        alpha = rank - rank_low

        value_low = int(
            np.searchsorted(
                cdf,
                rank_low + 1,
                side="left",
            )
        )

        value_high = int(
            np.searchsorted(
                cdf,
                rank_high + 1,
                side="left",
            )
        )

        value = (
            value_low
            + alpha
            * (value_high - value_low)
        )

        results.append(float(value))

    return np.asarray(
        results,
        dtype=np.float64,
    )


def normalize_and_pad_volume(
    volume,
    p_low: float,
    p_high: float,
    patch_size,
    stride,
    chunk_z: int = 8,
):
    z, y, x = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    z_target = compute_padded_length(
        z, pz, sz
    )
    y_target = compute_padded_length(
        y, py, sy
    )
    x_target = compute_padded_length(
        x, px, sx
    )

    pad_z = z_target - z
    pad_y = y_target - y
    pad_x = x_target - x

    padded = np.empty(
        (z_target, y_target, x_target),
        dtype=np.float32,
    )

    if p_high <= p_low:
        padded.fill(0.0)
    else:
        inv_range = 1.0 / (
            p_high - p_low
        )

        for z0 in range(0, z, chunk_z):
            z1 = min(
                z0 + chunk_z,
                z,
            )

            chunk = np.asarray(
                volume[z0:z1],
                dtype=np.float32,
            )

            np.clip(
                chunk,
                p_low,
                p_high,
                out=chunk,
            )

            chunk -= p_low
            chunk *= inv_range

            padded[
                z0:z1,
                :y,
                :x,
            ] = chunk

    # --------------------------------------------------------
    # Reproduce np.pad(..., mode="reflect")
    # without copying the whole volume.
    # --------------------------------------------------------

    if pad_x > 0:
        x_idx = np.pad(
            np.arange(x),
            (0, pad_x),
            mode="reflect",
        )[x:]

        padded[
            :z,
            :y,
            x:
        ] = np.take(
            padded[:z, :y, :x],
            x_idx,
            axis=2,
        )

    if pad_y > 0:
        y_idx = np.pad(
            np.arange(y),
            (0, pad_y),
            mode="reflect",
        )[y:]

        padded[
            :z,
            y:,
            :
        ] = np.take(
            padded[:z, :y, :],
            y_idx,
            axis=1,
        )

    if pad_z > 0:
        z_idx = np.pad(
            np.arange(z),
            (0, pad_z),
            mode="reflect",
        )[z:]

        padded[
            z:,
            :,
            :
        ] = np.take(
            padded[:z, :, :],
            z_idx,
            axis=0,
        )

    pad_info = {
        "original_shape": [
            int(z),
            int(y),
            int(x),
        ],
        "padded_shape": [
            int(z_target),
            int(y_target),
            int(x_target),
        ],
        "pad_width": [
            [0, int(pad_z)],
            [0, int(pad_y)],
            [0, int(pad_x)],
        ],
    }

    return padded, pad_info

def stitch_patch_instance_results(
    global_instance_map: np.ndarray,
    confidence_sum: np.ndarray,
    confidence_count: np.ndarray,
    patch_instance_map: np.ndarray,
    patch_confidence: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    next_instance_id: int,
    max_coords: tuple[int, int, int],
    patch_size: tuple[int, int, int] = (32, 256, 256),
    stride: tuple[int, int, int] = (16, 224, 224),
    face_iom_thresh: float = 0.2,
    stitch_min_core_voxels: int = 64,
    min_contact_cc_pixels: int = 10,
    min_contact_cc_ratio: float = 0.5,
    contact_connectivity: int = 2,
) -> int:
    """
    Stitch: contiguous-contact based stitching.

    Main idea:
    For each local/global candidate pair on a stitch face, build a 2D contact
    mask:

        contact_mask = (local_face == local_id) & (global_face == global_id)

    Merge only when:
        1. IoM >= face_iom_thresh
        2. largest connected contact component >= min_contact_cc_pixels
        3. largest connected contact component / total contact pixels >=
           min_contact_cc_ratio

    This distinguishes:
        - true straight/continuous stitch seams -> merge
        - sparse dotted contact between densely packed cells -> do not merge
    """
    patch_instance_map = patch_instance_map.copy()

    # =========================================================================
    # 1. Split disconnected components inside each local instance
    # =========================================================================
    max_lid = int(patch_instance_map.max())
    if max_lid > 0:
        instance_slices = find_objects(patch_instance_map, max_lid)
        current_max_local_id = max_lid

        for lid_minus_1, sub_slice in enumerate(instance_slices):
            if sub_slice is None:
                continue

            lid = lid_minus_1 + 1
            sub_map = patch_instance_map[sub_slice]
            mask = sub_map == lid

            labeled_mask, num_features = sk_label(
                mask,
                return_num=True,
                connectivity=1,
            )

            if num_features > 1:
                for feat_id in range(2, num_features + 1):
                    current_max_local_id += 1
                    sub_map[labeled_mask == feat_id] = current_max_local_id

    max_local_id = int(patch_instance_map.max())

    # =========================================================================
    # 2. Dynamic core region
    # =========================================================================
    dz, dy, dx = patch_size
    sz, sy, sx = stride
    mz, my, mx = (dz - sz) // 2, (dy - sy) // 2, (dx - sx) // 2

    max_z, max_y, max_x = max_coords
    limit_z, limit_y, limit_x = global_instance_map.shape

    z_start = 0 if z0 == 0 else mz
    z_end = dz if z0 == max_z else mz + sz

    y_start = 0 if y0 == 0 else my
    y_end = dy if y0 == max_y else my + sy

    x_start = 0 if x0 == 0 else mx
    x_end = dx if x0 == max_x else mx + sx

    core_instance = patch_instance_map[
        z_start:z_end,
        y_start:y_end,
        x_start:x_end,
    ]

    gz0, gy0, gx0 = z0 + z_start, y0 + y_start, x0 + x_start
    gz1, gy1, gx1 = z0 + z_end, y0 + y_end, x0 + x_end

    # =========================================================================
    # 3. Local IDs in core
    # =========================================================================
    core_counts_all = np.bincount(core_instance.ravel())

    if core_counts_all.size > 1:
        core_ids = np.nonzero(core_counts_all)[0]
        core_ids = core_ids[core_ids > 0]
        local_ids_in_core = core_ids[
            core_counts_all[core_ids] >= int(stitch_min_core_voxels)
        ]
    else:
        core_ids = np.array([], dtype=np.int64)
        local_ids_in_core = np.array([], dtype=np.int64)

    # =========================================================================
    # 4. Helper: measure contact continuity
    # =========================================================================
    def _contact_continuity_stats(contact_mask_2d: np.ndarray):
        """
        Return:
            total_contact: total contact pixels
            largest_cc: largest connected contact component size
            cc_ratio: largest_cc / total_contact
        """
        total_contact = int(contact_mask_2d.sum())
        if total_contact <= 0:
            return 0, 0, 0.0

        cc_map = sk_label(
            contact_mask_2d.astype(np.uint8),
            background=0,
            connectivity=int(contact_connectivity),
        )

        if cc_map.max() <= 0:
            return total_contact, 0, 0.0

        cc_sizes = np.bincount(cc_map.ravel())
        if cc_sizes.size <= 1:
            return total_contact, 0, 0.0

        largest_cc = int(cc_sizes[1:].max())
        cc_ratio = float(largest_cc) / float(total_contact + 1e-6)

        return total_contact, largest_cc, cc_ratio

    # =========================================================================
    # 5. Build seam faces separately
    # =========================================================================
    # Each face stores 2D arrays, not flattened arrays. This is necessary
    # because we need connected components of contact pixels.
    faces = []

    if gz0 > 0:
        faces.append(
            (
                'Z',
                core_instance[0, :, :],
                global_instance_map[gz0 - 1, gy0:gy1, gx0:gx1],
            )
        )

    if gy0 > 0:
        faces.append(
            (
                'Y',
                core_instance[:, 0, :],
                global_instance_map[gz0:gz1, gy0 - 1, gx0:gx1],
            )
        )

    if gx0 > 0:
        faces.append(
            (
                'X',
                core_instance[:, :, 0],
                global_instance_map[gz0:gz1, gy0:gy1, gx0 - 1],
            )
        )

    # =========================================================================
    # 6. Candidate matching with contiguous-contact filtering
    # =========================================================================
    id_remap = {}

    for lid in local_ids_in_core:
        candidates = []

        for face_name, local_face, global_face in faces:
            local_mask = local_face == lid
            area_l = int(local_mask.sum())

            if area_l <= 0:
                continue

            touched_global_ids = np.unique(global_face[local_mask])
            touched_global_ids = touched_global_ids[touched_global_ids > 0]

            if touched_global_ids.size == 0:
                continue

            for gid in touched_global_ids:
                gid = int(gid)

                global_mask = global_face == gid
                area_g = int(global_mask.sum())

                if area_g <= 0:
                    continue

                contact_mask = local_mask & global_mask

                total_contact, largest_cc, cc_ratio = (
                    _contact_continuity_stats(contact_mask)
                )

                if total_contact <= 0:
                    continue

                iom = float(total_contact) / float(min(area_l, area_g) + 1e-6)

                if (
                    iom >= float(face_iom_thresh)
                    and largest_cc >= int(min_contact_cc_pixels)
                    and cc_ratio >= float(min_contact_cc_ratio)
                ):
                    candidates.append(
                        {
                            'gid': gid,
                            'face': face_name,
                            'iom': iom,
                            'total_contact': total_contact,
                            'largest_cc': largest_cc,
                            'cc_ratio': cc_ratio,
                        }
                    )

        if candidates:
            # Prefer the candidate with the strongest continuous contact.
            candidates.sort(
                key=lambda c: (
                    c['largest_cc'],
                    c['cc_ratio'],
                    c['iom'],
                    c['total_contact'],
                ),
                reverse=True,
            )

            best_gid = int(candidates[0]['gid'])
            id_remap[int(lid)] = best_gid

        else:
            id_remap[int(lid)] = int(next_instance_id)
            next_instance_id += 1

    # =========================================================================
    # 7. Handle empty remap
    # =========================================================================
    num_small = (
        int(
            np.sum(
                (core_ids > 0)
                & (core_counts_all[core_ids] < int(stitch_min_core_voxels))
            )
        )
        if core_counts_all.size > 1
        else 0
    )

    if len(id_remap) == 0 and num_small > 0:
        remapped_patch = np.zeros_like(patch_instance_map, dtype=np.uint32)
    elif len(id_remap) == 0:
        return next_instance_id
    else:
        lut = np.zeros(max_local_id + 1, dtype=np.uint32)
        for lid, gid in id_remap.items():
            lut[int(lid)] = int(gid)

        remapped_patch = lut[patch_instance_map]

    # =========================================================================
    # 8. Write back
    # =========================================================================
    gz1_full = min(z0 + patch_instance_map.shape[0], limit_z)
    gy1_full = min(y0 + patch_instance_map.shape[1], limit_y)
    gx1_full = min(x0 + patch_instance_map.shape[2], limit_x)

    local_z1 = gz1_full - z0
    local_y1 = gy1_full - y0
    local_x1 = gx1_full - x0

    remapped_crop = remapped_patch[:local_z1, :local_y1, :local_x1]
    conf_crop = patch_confidence[:local_z1, :local_y1, :local_x1]

    valid_mask = remapped_crop > 0

    if valid_mask.any():
        global_crop = global_instance_map[
            z0:gz1_full,
            y0:gy1_full,
            x0:gx1_full,
        ]
        conf_sum_crop = confidence_sum[
            z0:gz1_full,
            y0:gy1_full,
            x0:gx1_full,
        ]
        conf_cnt_crop = confidence_count[
            z0:gz1_full,
            y0:gy1_full,
            x0:gx1_full,
        ]

        global_crop[valid_mask] = remapped_crop[valid_mask]
        conf_sum_crop[valid_mask] += conf_crop[valid_mask]
        conf_cnt_crop[valid_mask] += 1

    return next_instance_id


# ============================================================
# Final filtering / finalization
# ============================================================


def filter_instances_by_size(
    instance_map: np.ndarray,
    confidence_map: np.ndarray,
    min_area: int,
    intensity_volume: np.ndarray | None = None,
    min_size: int = 0,
    max_size: int | None = None,
    background_threshold: float | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast postprocessing.

    Step 1:
        For each z-slice, run connected component analysis directly on the
        label image. Components are connected only when neighboring pixels
        have the same non-zero instance ID.

        This is equivalent to:
            for each instance ID:
                run 2D CC and remove components smaller than min_area
        but avoids looping over every ID.

    Step 2:
        Filter by 3D original instance IDs using vectorized bincount:
            - total voxel count per ID
            - mean intensity per ID
        Then remap valid IDs to consecutive IDs.
    """
    if instance_map.shape != confidence_map.shape:
        raise ValueError(
            'instance_map and confidence_map must have the same shape, '
            f'got {instance_map.shape} and {confidence_map.shape}'
        )

    if (
        intensity_volume is not None
        and intensity_volume.shape != instance_map.shape
    ):
        raise ValueError(
            'intensity_volume must have the same shape as instance_map, '
            f'got {intensity_volume.shape} and {instance_map.shape}'
        )

    if min_area < 0:
        raise ValueError('min_area must be >= 0')

    if min_size < 0:
        raise ValueError('min_size must be >= 0')

    if max_size is not None and max_size < min_size:
        raise ValueError('max_size must be None or >= min_size')

    out_instance = instance_map.copy().astype(np.uint32, copy=False)
    out_confidence = confidence_map.copy().astype(np.float32, copy=False)

    # ============================================================
    # Step 1: fast per-slice same-ID connected component filtering
    # ============================================================
    removed_2d_cc = 0
    removed_2d_voxels = 0

    if min_area > 0:
        for z in range(out_instance.shape[0]):
            sl = out_instance[z]
            conf_sl = out_confidence[z]

            if not np.any(sl > 0):
                continue

            # sk_label on an integer label image labels connected regions of
            # equal value. background=0 ignores background.
            #
            # connectivity=1 means 2D 4-connectivity.
            cc_map = sk_label(
                sl,
                background=0,
                connectivity=1,
            )

            if cc_map.max() == 0:
                continue

            cc_sizes = np.bincount(cc_map.ravel())

            small_cc = cc_sizes < int(min_area)
            small_cc[0] = False

            remove_mask = small_cc[cc_map]

            if remove_mask.any():
                removed_2d_cc += int(np.sum(small_cc))
                removed_2d_voxels += int(remove_mask.sum())

                sl[remove_mask] = 0
                conf_sl[remove_mask] = 0.0

    if verbose:
        print(
            f'[2D filtering] Removed {removed_2d_cc} small components, '
            f'{removed_2d_voxels} voxels in total.'
        )

    # ============================================================
    # Step 2: vectorized ID-wise 3D size / intensity filtering
    # ============================================================
    flat_ids = out_instance.ravel()
    max_old_id = int(flat_ids.max())

    if max_old_id == 0:
        return (
            np.zeros_like(out_instance, dtype=np.uint32),
            np.zeros_like(out_confidence, dtype=np.float32),
        )

    counts = np.bincount(flat_ids, minlength=max_old_id + 1)
    ids_arr = np.arange(max_old_id + 1)

    valid = np.zeros(max_old_id + 1, dtype=bool)
    valid[1:] = counts[1:] > 0

    if min_size > 0:
        valid &= counts >= int(min_size)

    if max_size is not None:
        valid &= counts <= int(max_size)

    valid[0] = False

    removed_by_size = int(np.sum((counts > 0) & (~valid) & (ids_arr > 0)))

    # Intensity filtering
    removed_by_intensity = 0

    if intensity_volume is not None and background_threshold is not None:
        flat_intensity = intensity_volume.astype(
            np.float32, copy=False
        ).ravel()

        intensity_sum = np.bincount(
            flat_ids,
            weights=flat_intensity,
            minlength=max_old_id + 1,
        )

        mean_intensity = np.zeros(max_old_id + 1, dtype=np.float32)
        nonzero = counts > 0
        mean_intensity[nonzero] = intensity_sum[nonzero] / counts[nonzero]

        before_intensity_valid = valid.copy()

        valid &= mean_intensity >= float(background_threshold)
        valid[0] = False

        removed_by_intensity = int(
            np.sum(before_intensity_valid & (~valid) & (ids_arr > 0))
        )

    valid_ids = np.flatnonzero(valid)

    if valid_ids.size == 0:
        new_instance = np.zeros_like(out_instance, dtype=np.uint32)
        new_confidence = np.zeros_like(out_confidence, dtype=np.float32)

        if verbose:
            print(
                '[ID filtering] No valid instances remain after '
                'size/intensity filtering.'
            )

        return new_instance, new_confidence

    # Consecutive remapping: old ID -> new ID
    lut = np.zeros(max_old_id + 1, dtype=np.uint32)
    lut[valid_ids] = np.arange(1, valid_ids.size + 1, dtype=np.uint32)

    new_instance = lut[flat_ids].reshape(out_instance.shape)

    new_confidence = out_confidence.copy()
    new_confidence[new_instance == 0] = 0.0

    if verbose:
        print(
            f'[ID filtering] Kept {valid_ids.size} instances. '
            f'Removed by size: {removed_by_size}. '
            f'Removed by intensity: {removed_by_intensity}. '
        )

    return new_instance.astype(np.uint32), new_confidence.astype(np.float32)


def finalize_instance_and_confidence(
    global_instance_map: np.ndarray,
    confidence_sum: np.ndarray,
    confidence_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finalize stitching result.

    Notes:
        - instance_map is returned from global_instance_map as-is.
        - No connected-component relabeling is performed here.
        - No union-find / global-bridge remapping is performed here.
        - confidence_map is the average confidence over written voxels.
    """
    instance_map = global_instance_map.astype(np.uint32, copy=False)

    valid = confidence_count > 0
    confidence_map = np.zeros_like(confidence_sum, dtype=np.float32)
    confidence_map[valid] = confidence_sum[valid] / confidence_count[valid]
    confidence_map[instance_map == 0] = 0.0

    return instance_map, confidence_map


# ============================================================
# Folder-level inference
# ============================================================


def infer_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config_file: str | Path,
    weights_path: str | Path,
    recursive: bool = False,
    skip_existing: bool = True,
    keep_failed_tmp: bool = False,
    **infer_kwargs,
) -> dict:
    """
    Batch inference for all tif/tiff/zarr files in a folder.
    Only keep final mask result: <stem>_instance_map.tif

    Parameters
    ----------
    input_dir:
        Folder containing tif/tiff/zarr files.
    output_dir:
        Folder to save mask results.
    config_file:
        Config yaml path.
    weights_path:
        Model checkpoint path.
    recursive:
        If True, search tif/tiff/zarr recursively.
    skip_existing:
        If True, skip files whose output mask already exists.
    keep_failed_tmp:
        If True, keep temporary output folder when an image fails.
    infer_kwargs:
        Other parameters passed to infer_volume, e.g. batch_size, patch_size.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f'input_dir does not exist: {input_dir}')

    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.zarr', '*.ZARR']
    image_files = []
    for pat in patterns:
        if recursive:
            image_files.extend(input_dir.rglob(pat))
        else:
            image_files.extend(input_dir.glob(pat))

    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        raise FileNotFoundError(f'No tif/tiff/zarr files found in: {input_dir}')

    summary = {
        'total': len(image_files),
        'done': [],
        'skipped': [],
        'failed': [],
    }

    print(f'[Batch inference] Found {len(image_files)} tif/tiff/zarr files.')
    print(f'[Batch inference] Input : {input_dir}')
    print(f'[Batch inference] Output: {output_dir}')

    for image_path in tqdm(image_files, desc='Batch infer', unit='vol'):
        image_path = Path(image_path)

        if recursive:
            rel_parent = image_path.relative_to(input_dir).parent
            final_out_dir = output_dir / rel_parent
        else:
            final_out_dir = output_dir

        final_out_dir.mkdir(parents=True, exist_ok=True)

        final_mask_path = final_out_dir / f'{image_path.stem}_instance_map.tif'

        if skip_existing and final_mask_path.exists():
            print(f'[Skip] {final_mask_path}')
            summary['skipped'].append(str(final_mask_path))
            continue

        tmp_dir = final_out_dir / f'.tmp_{image_path.stem}'

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = infer_volume(
                image_path=image_path,
                config_file=config_file,
                weights_path=weights_path,
                output_dir=tmp_dir,
                save_intermediate=False,
                **infer_kwargs,
            )

            src_mask_path = Path(result['instance_map_path'])

            if final_mask_path.exists():
                final_mask_path.unlink()

            shutil.move(str(src_mask_path), str(final_mask_path))
            shutil.rmtree(tmp_dir, ignore_errors=True)

            print(f'[Done] {image_path.name} -> {final_mask_path}')
            summary['done'].append(str(final_mask_path))

        except Exception as e:
            err_msg = traceback.format_exc()
            print(f'[Failed] {image_path}')
            print(err_msg)

            summary['failed'].append(
                {
                    'image_path': str(image_path),
                    'error': repr(e),
                }
            )

            if not keep_failed_tmp:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    print('\n[Batch inference finished]')
    print(f"  Total  : {summary['total']}")
    print(f"  Done   : {len(summary['done'])}")
    print(f"  Skipped: {len(summary['skipped'])}")
    print(f"  Failed : {len(summary['failed'])}")

    return summary
