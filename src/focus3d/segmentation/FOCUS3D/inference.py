import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tifffile
import torch
import zarr
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from tqdm.auto import tqdm


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


def _emit_status(progress_callback, message: str) -> None:
    """
    Emit status-only message.

    Convention:
        value = -1 means: update text only, do not update progress bar.
    """
    if progress_callback is None:
        return

    progress_callback(-1, str(message))


def _emit_progress(progress_callback, value: int, message: str) -> None:
    """
    Emit real patch-level progress.

    Only use this during patch inference.
    """
    if progress_callback is None:
        return

    value = max(0, min(100, int(value)))
    progress_callback(value, str(message))


def _check_cancelled(cancel_callback) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError('__SEG_CANCELLED__')


# ============================================================
# Main inference function
# ============================================================
def setup_cfg(config_file, weights_path):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True)
    cfg.INPUT.IMAGE_SIZE = [32, 96, 96]
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.freeze()
    return cfg


def build_predictor(cfg):
    model = build_model(cfg)
    if model.sem_seg_postprocess_before_inference:
        model.sem_seg_postprocess_before_inference = False
    print(f'Loading weights from: {cfg.MODEL.WEIGHTS}')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model


def infer_volume(
    image_path: Union[str, Path],
    config_file: Union[str, Path],
    weights_path: Union[str, Path],
    device: str | None = None,
    z_ratio: float = 1.0,
    output_dir: Union[str, Path] = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    background_threshold: float = 5.0,
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
    patch_size: Tuple[int, int, int] = (32, 96, 96),  # (Z, Y, X)
    stride: Tuple[int, int, int] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> Dict:
    """
    Inference on an arbitrarily-sized 3D volume:
        1) Read image
        2) Normalize to [0, 1]
        3) Pad and generate sliding windows
        4) Skip pure-background patches
        5) Run batch inference
        6) Postprocess patch predictions and stitch them
        7) Build final instance/confidence maps
        8) Save outputs
    """
    t_total_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_status(progress_callback, 'Loading segmentation model...')
    _check_cancelled(cancel_callback)
    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    print('\n[Step 1] "Building config and loading model..."')
    cfg = setup_cfg(config_file, weights_path)
    model = build_predictor(cfg)
    device = cfg.MODEL.DEVICE if device is None else device
    amp_enabled = bool(use_amp and str(device).startswith('cuda'))
    if amp_dtype == 'bfloat16':
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')

    log_info = {
        'image_path': str(image_path),
        'z_ratio': float(z_ratio),
        'lower_percentile': float(lower_percentile),
        'upper_percentile': float(upper_percentile),
        'background_threshold': float(background_threshold),
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
    }

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------
    print('\n[Step 2] "Reading image..."')
    _emit_status(progress_callback, 'Reading and normalizing image...')
    _check_cancelled(cancel_callback)
    raw_volume = read_volume(image_path)
    norm_volume, norm_stats = normalize_img(
        raw_volume,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    bg_raw = float(background_threshold)
    p_low = norm_stats['p_low']
    p_high = norm_stats['p_high']
    if p_high <= p_low:
        background_threshold_norm = 0.0
    else:
        background_threshold_norm = (bg_raw - p_low) / (p_high - p_low)
        background_threshold_norm = float(
            np.clip(background_threshold_norm, 0.0, 1.0)
        )
    log_info['normalization'] = norm_stats

    # --------------------------------------------------------
    # Patch / stride planning and padding
    # --------------------------------------------------------
    _emit_status(progress_callback, 'Preparing sliding-window patches...')
    _check_cancelled(cancel_callback)
    print('\n[Step 3] "Generating patches..."')
    if stride is None:
        stride = tuple(s // 2 for s in patch_size)
    else:
        stride = tuple(stride)
    padded_volume, pad_info = pad_volume_for_sliding_window(
        norm_volume, patch_size, stride
    )

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
    for z0, y0, x0 in all_coords:
        patch = padded_volume[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
        if np.percentile(patch, 99.9) < background_threshold_norm:
            skipped_coords.append((z0, y0, x0))
        else:
            infer_coords.append((z0, y0, x0))

    log_info['num_infer_patches'] = int(len(infer_coords))
    log_info['num_skipped_background_patches'] = int(len(skipped_coords))

    _emit_status(
        progress_callback,
        (
            f'Prepared {len(infer_coords)} inference patches '
            f'and skipped {len(skipped_coords)} background patches.'
        ),
    )
    _check_cancelled(cancel_callback)

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
        collate_fn=lambda x: x,
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

    for batch_idx, inputs in enumerate(patch_pbar, start=1):
        _check_cancelled(cancel_callback)

        for sample in inputs:
            sample['image'] = sample['image'].to(device, non_blocking=True)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type='cuda',
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ),
        ):
            batch_outputs = model(inputs)

        for sample, model_output in zip(inputs, batch_outputs):
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

            next_instance_id = stitch_patch_instance_results_v4(
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
                    patch_output_dir / f'{patch_tag}_instance.tif',
                    patch_instance_map.astype(np.uint16),
                )

    print('\n[Step 5] "Post-processing and save results..."')
    _emit_status(progress_callback, 'Post-processing segmentation result...')
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

    # Save result before final post-processing
    pre_post_instance_padded = instance_map_padded.copy()
    pre_post_confidence_padded = confidence_map_padded.copy()

    z_raw, y_raw, x_raw = raw_volume.shape
    pre_post_instance = pre_post_instance_padded[:z_raw, :y_raw, :x_raw]
    pre_post_confidence = pre_post_confidence_padded[:z_raw, :y_raw, :x_raw]

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

    # Crop back to original shape
    z_raw, y_raw, x_raw = raw_volume.shape
    instance_map = instance_map_padded[:z_raw, :y_raw, :x_raw]
    confidence_map = confidence_map_padded[:z_raw, :y_raw, :x_raw]

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    stem = Path(image_path).stem

    instance_path = output_dir / f'{stem}_instance_map.tif'
    confidence_path = output_dir / f'{stem}_confidence_map.tif'
    log_path = output_dir / f'{stem}_log.json'

    save_volume(instance_path, instance_map.astype(np.uint32))
    save_volume(confidence_path, confidence_map.astype(np.float32))

    log_info['total_time_sec'] = time.time() - t_total_start

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_info, f, indent=2)

    result = {
        'instance_map_path': str(instance_path),
        'confidence_map_path': str(confidence_path),
        'log_json_path': str(log_path),
        'num_infer_patches': len(infer_coords),
        'num_skipped_background_patches': len(skipped_coords),
        'instance_map': instance_map,
        'confidence_map': confidence_map,
        'log_info': log_info,
    }
    return result


# ============================================================
# Basic I/O
# ============================================================
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, padded_volume, infer_coords, patch_size):
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


def read_volume(image_path: Union[str, Path]) -> np.ndarray:
    """
    Read a 3D volume from .tif/.tiff or .zarr and return a numpy array with shape (Z, Y, X).

    Notes:
        - This function assumes the input is already a 3D volume or can be squeezed to 3D.
        - If the input has singleton dimensions, they will be removed.
    """
    image_path = str(image_path)
    suffix = Path(image_path).suffix.lower()

    if suffix in ['.tif', '.tiff']:
        vol = tifffile.imread(image_path)
    elif suffix == '.zarr':
        z = zarr.open(image_path, mode='r')
        vol = np.asarray(z)
    else:
        raise ValueError(f'Unsupported file format: {image_path}')

    vol = np.asarray(vol)
    vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(
            f'Expected a 3D volume after squeeze, got shape {vol.shape}'
        )

    return vol.astype(np.float32, copy=False)


def save_volume(output_path: Union[str, Path], volume: np.ndarray) -> None:
    """
    Save a volume either as .tif/.tiff or .zarr based on file extension.
    """
    output_path = str(output_path)
    suffix = Path(output_path).suffix.lower()

    if suffix in ['.tif', '.tiff']:
        tifffile.imwrite(output_path, volume)
    elif suffix == '.zarr':
        z = zarr.open(
            output_path, mode='w', shape=volume.shape, dtype=volume.dtype
        )
        z[:] = volume
    else:
        raise ValueError(f'Unsupported output format: {output_path}')


def normalize_img(
    volume: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize the whole volume to [0, 255] using user-provided percentiles.

    Returns:
        normalized_volume: float32 array in [0, 255]
        stats: dictionary with normalization metadata
    """
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError('Percentiles must satisfy 0 <= lower < upper <= 100')

    p_low = float(np.percentile(volume, lower_percentile))
    p_high = float(np.percentile(volume, upper_percentile))

    if p_high <= p_low:
        # Fallback for degenerate inputs
        p_low = float(volume.min())
        p_high = float(volume.max())
        normalized = np.zeros_like(volume, dtype=np.float32)
        stats = {
            'raw_min': float(volume.min()),
            'raw_max': float(volume.max()),
            'p_low': p_low,
            'p_high': p_high,
            'note': 'Degenerate image; output is all zeros.',
        }
        return normalized, stats

    clipped = np.clip(volume, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)
    normalized = normalized.astype(np.float32, copy=False)

    stats = {
        'raw_min': float(volume.min()),
        'raw_max': float(volume.max()),
        'p_low': p_low,
        'p_high': p_high,
    }
    return normalized, stats


# ============================================================
# Sliding-window planning and padding
# ============================================================


def compute_padded_length(length: int, patch: int, stride: int) -> int:
    """
    Compute the padded size so that the sliding windows fully cover the axis.

    Cases:
        - If length <= patch: pad up to patch
        - Otherwise: patch + n * stride, with the smallest n covering the full axis
    """
    if length <= patch:
        return patch

    n_steps = math.ceil((length - patch) / stride)
    return patch + n_steps * stride


def pad_volume_for_sliding_window(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict]:
    """
    Pad the input volume so that all sliding windows fit exactly.

    Padding is applied only at the end of each dimension.
    """
    z, y, x = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    z_pad_target = compute_padded_length(z, pz, sz)
    y_pad_target = compute_padded_length(y, py, sy)
    x_pad_target = compute_padded_length(x, px, sx)

    pad_z = z_pad_target - z
    pad_y = y_pad_target - y
    pad_x = x_pad_target - x

    pad_width = (
        (0, pad_z),
        (0, pad_y),
        (0, pad_x),
    )

    padded = np.pad(volume, pad_width=pad_width, mode='reflect')

    pad_info = {
        'original_shape': [int(z), int(y), int(x)],
        'padded_shape': [
            int(padded.shape[0]),
            int(padded.shape[1]),
            int(padded.shape[2]),
        ],
        'pad_width': [[0, int(pad_z)], [0, int(pad_y)], [0, int(pad_x)]],
    }
    return padded, pad_info


def generate_patch_coords(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
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

    kept_scores = []
    kept_labels = []
    kept_original_indices = []

    if 'pred_scores' not in model_output or 'pred_masks' not in model_output:
        return {
            'patch_instance_map': patch_instance_map,
            'patch_confidence': patch_confidence,
            'kept_scores': kept_scores,
            'kept_labels': kept_labels,
            'kept_original_indices': kept_original_indices,
        }

    scores = model_output['pred_scores'].detach().float()
    masks = model_output['pred_masks'].detach().float()

    if 'pred_classes' in model_output:
        labels = model_output['pred_classes'].detach()
    else:
        labels = torch.zeros_like(scores, dtype=torch.long)

    if masks.numel() == 0 or scores.numel() == 0:
        return {
            'patch_instance_map': patch_instance_map,
            'patch_confidence': patch_confidence,
            'kept_scores': kept_scores,
            'kept_labels': kept_labels,
            'kept_original_indices': kept_original_indices,
        }

    if masks.min() < 0 or masks.max() > 1:
        mask_prob = masks.sigmoid()
    else:
        mask_prob = masks

    # ------------------------------------------------------------
    # 1. Compute mask quality score
    # ------------------------------------------------------------
    mask_bin_tmp = mask_prob > mask_thresh

    mask_scores = (mask_prob.flatten(1) * mask_bin_tmp.flatten(1)).sum(
        dim=1
    ) / (mask_bin_tmp.flatten(1).sum(dim=1) + 1e-6)

    # Final query score = classification score * mask quality score
    final_scores = scores * mask_scores

    # Keep queries that satisfy both final score and mask quality thresholds
    keep = (final_scores > score_thresh) & (mask_scores > mask_thresh)

    if keep.sum().item() == 0:
        return {
            'patch_instance_map': patch_instance_map,
            'patch_confidence': patch_confidence,
            'kept_scores': kept_scores,
            'kept_labels': kept_labels,
            'kept_original_indices': kept_original_indices,
        }

    original_indices = torch.arange(scores.shape[0], device=scores.device)

    final_scores = final_scores[keep]
    mask_scores = mask_scores[keep]
    labels = labels[keep]
    mask_prob = mask_prob[keep]
    original_indices = original_indices[keep]

    # ------------------------------------------------------------
    # 2. Sort queries by final score
    # ------------------------------------------------------------
    order = torch.argsort(final_scores, descending=True)
    final_scores = final_scores[order]
    mask_scores = mask_scores[order]
    labels = labels[order]
    mask_prob = mask_prob[order]
    original_indices = original_indices[order]

    if topk_postprocess is not None and mask_prob.shape[0] > topk_postprocess:
        final_scores = final_scores[:topk_postprocess]
        mask_scores = mask_scores[:topk_postprocess]
        labels = labels[:topk_postprocess]
        mask_prob = mask_prob[:topk_postprocess]
        original_indices = original_indices[:topk_postprocess]

    num_queries = mask_prob.shape[0]

    # ------------------------------------------------------------
    # 3. Conflict resolving before voxel-level argmax
    # ------------------------------------------------------------
    # New rule:
    #   For two conflicting masks on the same z-slice:
    #     1) If intersection / min(area_a, area_b) > 0.8 -> merge
    #     2) Else if IoU(intersection / union) > 0.5 -> merge
    #     3) If merged, assign the merged 2D structure to the query whose
    #        adjacent z-slice mask is more similar to the merged structure.
    #     4) If neither condition is satisfied, keep both queries unchanged.

    mask_bin = mask_prob > mask_thresh
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
                cc_mask_np = cc_map == cc_id
                if int(cc_mask_np.sum()) < conflict_min_pixels:
                    continue

                cc_mask = torch.from_numpy(cc_mask_np).to(
                    device=mask_prob.device,
                    dtype=torch.bool,
                )

                participate = (
                    (mask_bin[:, z] & cc_mask[None, :, :])
                    .flatten(1)
                    .any(dim=1)
                )

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

                        should_merge = (iom > merge_iom_thresh) or (
                            iou > merge_iou_thresh
                        )
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
                                _safe_iou_bool(
                                    merged_mask, mask_bin[q, z - 1]
                                ),
                            )

                        if z < D - 1:
                            adj_iou = max(
                                adj_iou,
                                _safe_iou_bool(
                                    merged_mask, mask_bin[q, z + 1]
                                ),
                            )

                        score_tie = float(final_scores[q].item())

                        if adj_iou > best_adj_iou or (
                            abs(adj_iou - best_adj_iou) < 1e-6
                            and score_tie > best_score_tie
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

                    mask_prob_resolved[best_q, z][merged_mask] = merged_prob[
                        merged_mask
                    ]
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
        mask_prob_resolved, dim=0, index=best_idx.unsqueeze(0)
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

        kept_scores.append(float(final_scores[local_q].item()))
        kept_labels.append(int(labels[local_q].item()))
        kept_original_indices.append(int(original_indices[local_q].item()))

        new_label_id += 1

    return {
        'patch_instance_map': patch_instance_map,
        'patch_confidence': patch_confidence,
        'kept_scores': kept_scores,
        'kept_labels': kept_labels,
        'kept_original_indices': kept_original_indices,
    }


def stitch_patch_instance_results_v4_fixed(
    global_instance_map,
    confidence_sum,
    confidence_count,
    patch_instance_map,
    patch_confidence,
    z0,
    y0,
    x0,
    next_instance_id,
    max_coords,
    patch_size=(32, 256, 256),
    stride=(16, 224, 224),
    face_iom_thresh=0.2,
):
    dz, dy, dx = patch_size
    sz, sy, sx = stride
    mz, my, mx = (dz - sz) // 2, (dy - sy) // 2, (dx - sx) // 2

    max_z, max_y, max_x = max_coords

    limit_z, limit_y, limit_x = global_instance_map.shape

    # --- Step 1: dynamic core boundary ---
    z_start = 0 if z0 == 0 else mz
    z_end = dz if z0 == max_z else mz + sz

    y_start = 0 if y0 == 0 else my
    y_end = dy if y0 == max_y else my + sy

    x_start = 0 if x0 == 0 else mx
    x_end = dx if x0 == max_x else mx + sx

    core_instance = patch_instance_map[
        z_start:z_end, y_start:y_end, x_start:x_end
    ]

    gz0, gy0, gx0 = z0 + z_start, y0 + y_start, x0 + x_start
    gz1, gy1, gx1 = z0 + z_end, y0 + y_end, x0 + x_end

    #  --- Step 2: face matching ---
    b_local, b_global = [], []
    if gz0 > 0:
        b_local.append(core_instance[0, :, :].flatten())
        b_global.append(
            global_instance_map[gz0 - 1, gy0:gy1, gx0:gx1].flatten()
        )
    if gy0 > 0:
        b_local.append(core_instance[:, 0, :].flatten())
        b_global.append(
            global_instance_map[gz0:gz1, gy0 - 1, gx0:gx1].flatten()
        )
    if gx0 > 0:
        b_local.append(core_instance[:, :, 0].flatten())
        b_global.append(
            global_instance_map[gz0:gz1, gy0:gy1, gx0 - 1].flatten()
        )

    b_local_arr = np.concatenate(b_local) if b_local else np.array([])
    b_global_arr = np.concatenate(b_global) if b_global else np.array([])

    global_area_dict = {}
    if b_global_arr.size > 0:
        g_ids, g_counts = np.unique(b_global_arr, return_counts=True)
        global_area_dict = dict(zip(g_ids, g_counts))

    # --- Step 3: id remap on local ids that appear in core ---
    local_ids_in_core = np.unique(core_instance)
    local_ids_in_core = local_ids_in_core[local_ids_in_core > 0]
    id_remap = {}

    for lid in local_ids_in_core:
        matched_gid = None
        if b_local_arr.size > 0:
            mask_l = b_local_arr == lid
            area_l = mask_l.sum()
            if area_l > 0:
                overlapping_gids, counts = np.unique(
                    b_global_arr[mask_l], return_counts=True
                )
                best_gid, best_iom = None, 0
                for gid, inter in zip(overlapping_gids, counts):
                    if gid == 0:
                        continue
                    area_g = global_area_dict.get(gid, 0)
                    iom = inter / min(area_l, area_g)
                    if iom > best_iom:
                        best_iom, best_gid = iom, gid
                if best_iom >= face_iom_thresh:
                    matched_gid = best_gid

        if matched_gid is not None:
            id_remap[lid] = matched_gid
        else:
            id_remap[lid] = next_instance_id
            next_instance_id += 1

    # --- Step 4: fast rendering write-back with LUT ---
    if len(id_remap) > 0:
        max_lid = int(patch_instance_map.max())
        lut = np.zeros(max_lid + 1, dtype=np.uint32)
        for lid, gid in id_remap.items():
            lut[lid] = gid

        remapped_patch = lut[patch_instance_map]

        gz1_full = min(z0 + dz, limit_z)
        gy1_full = min(y0 + dy, limit_y)
        gx1_full = min(x0 + dx, limit_x)

        lz1, ly1, lx1 = gz1_full - z0, gy1_full - y0, gx1_full - x0

        remapped_crop = remapped_patch[:lz1, :ly1, :lx1]
        conf_crop = patch_confidence[:lz1, :ly1, :lx1]

        valid_mask = remapped_crop > 0
        if valid_mask.any():
            global_instance_map[z0:gz1_full, y0:gy1_full, x0:gx1_full][
                valid_mask
            ] = remapped_crop[valid_mask]
            confidence_sum[z0:gz1_full, y0:gy1_full, x0:gx1_full][
                valid_mask
            ] += conf_crop[valid_mask]
            confidence_count[z0:gz1_full, y0:gy1_full, x0:gx1_full][
                valid_mask
            ] += 1

    return next_instance_id


def stitch_patch_instance_results_v4(
    global_instance_map,
    confidence_sum,
    confidence_count,
    patch_instance_map,
    patch_confidence,
    z0,
    y0,
    x0,
    next_instance_id,
    max_coords,
    patch_size=(32, 256, 256),
    stride=(16, 224, 224),
    face_iom_thresh=0.2,
):
    dz, dy, dx = patch_size
    sz, sy, sx = stride
    mz, my, mx = (dz - sz) // 2, (dy - sy) // 2, (dx - sx) // 2

    max_z, max_y, max_x = max_coords
    limit_z, limit_y, limit_x = global_instance_map.shape

    # --- Step 1: dynamic core boundary ---
    z_start = 0 if z0 == 0 else mz
    z_end = dz if z0 == max_z else mz + sz

    y_start = 0 if y0 == 0 else my
    y_end = dy if y0 == max_y else my + sy

    x_start = 0 if x0 == 0 else mx
    x_end = dx if x0 == max_x else mx + sx

    core_instance = patch_instance_map[
        z_start:z_end, y_start:y_end, x_start:x_end
    ]

    gz0, gy0, gx0 = z0 + z_start, y0 + y_start, x0 + x_start
    gz1, gy1, gx1 = z0 + z_end, y0 + y_end, x0 + x_end

    # --- Step 2: face matching ---
    b_local, b_global = [], []
    if gz0 > 0:
        b_local.append(core_instance[0, :, :].reshape(-1))
        b_global.append(
            global_instance_map[gz0 - 1, gy0:gy1, gx0:gx1].reshape(-1)
        )
    if gy0 > 0:
        b_local.append(core_instance[:, 0, :].reshape(-1))
        b_global.append(
            global_instance_map[gz0:gz1, gy0 - 1, gx0:gx1].reshape(-1)
        )
    if gx0 > 0:
        b_local.append(core_instance[:, :, 0].reshape(-1))
        b_global.append(
            global_instance_map[gz0:gz1, gy0:gy1, gx0 - 1].reshape(-1)
        )

    b_local_arr = (
        np.concatenate(b_local)
        if b_local
        else np.array([], dtype=core_instance.dtype)
    )
    b_global_arr = (
        np.concatenate(b_global)
        if b_global
        else np.array([], dtype=global_instance_map.dtype)
    )

    global_area_dict = {}
    if b_global_arr.size > 0:
        g_ids, g_counts = np.unique(b_global_arr, return_counts=True)
        global_area_dict = dict(zip(g_ids.tolist(), g_counts.tolist()))

    # --- Step 3: id remap on local ids that appear in core ---
    local_ids_in_core = np.unique(core_instance)
    local_ids_in_core = local_ids_in_core[local_ids_in_core > 0]
    id_remap = {}

    for lid in local_ids_in_core:
        matched_gid = None
        if b_local_arr.size > 0:
            mask_l = b_local_arr == lid
            area_l = int(mask_l.sum())
            if area_l > 0:
                overlapping_gids, counts = np.unique(
                    b_global_arr[mask_l], return_counts=True
                )
                best_gid, best_iom = None, 0.0
                for gid, inter in zip(overlapping_gids, counts):
                    if gid == 0:
                        continue
                    area_g = global_area_dict.get(int(gid), 0)
                    if area_g <= 0:
                        continue
                    iom = float(inter) / float(min(area_l, area_g))
                    if iom > best_iom:
                        best_iom = iom
                        best_gid = int(gid)
                if best_iom >= face_iom_thresh:
                    matched_gid = best_gid

        if matched_gid is not None:
            id_remap[int(lid)] = int(matched_gid)
        else:
            id_remap[int(lid)] = int(next_instance_id)
            next_instance_id += 1

    if len(id_remap) == 0:
        return next_instance_id

    # --- Step 4: fast rendering write-back with LUT ---
    max_local_id = int(patch_instance_map.max())
    lut = np.zeros(max_local_id + 1, dtype=np.uint32)
    for lid, gid in id_remap.items():
        lut[lid] = gid

    remapped_patch = lut[patch_instance_map]

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
            z0:gz1_full, y0:gy1_full, x0:gx1_full
        ]
        conf_sum_crop = confidence_sum[z0:gz1_full, y0:gy1_full, x0:gx1_full]
        conf_cnt_crop = confidence_count[z0:gz1_full, y0:gy1_full, x0:gx1_full]

        global_crop[valid_mask] = remapped_crop[valid_mask]
        conf_sum_crop[valid_mask] += conf_crop[valid_mask]
        conf_cnt_crop[valid_mask] += 1

    return next_instance_id


def filter_instances_by_size(
    instance_map: np.ndarray,
    confidence_map: np.ndarray,
    min_area: int,
    intensity_volume: np.ndarray | None = None,
    min_size: int = 0,
    max_size: int | None = None,
    background_threshold: float | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
            f'instance_map and confidence_map must have the same shape, '
            f'got {instance_map.shape} and {confidence_map.shape}'
        )

    if (
        intensity_volume is not None
        and intensity_volume.shape != instance_map.shape
    ):
        raise ValueError(
            f'intensity_volume must have the same shape as instance_map, '
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

            # Key point:
            # sk_label on an integer label image labels connected regions
            # of equal value. background=0 ignores background.
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
            f'[2D same-ID CC filtering] Removed {removed_2d_cc} small CCs, '
            f'{removed_2d_voxels} pixels/voxels in total.'
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
    valid[1:] = True

    # Size filtering
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
            f'Final IDs: 1 ~ {valid_ids.size}'
        )

    return new_instance.astype(np.uint32), new_confidence.astype(np.float32)


def finalize_instance_and_confidence(
    global_instance_map: np.ndarray,
    confidence_sum: np.ndarray,
    confidence_count: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finalize:
        - instance_map: connected components of the union foreground
        - confidence_map: average confidence in overlap regions
    """
    confidence_map = np.zeros_like(confidence_sum, dtype=np.float32)
    instance_map = global_instance_map.astype(np.uint32, copy=False)

    valid = confidence_count > 0
    confidence_map = np.zeros_like(confidence_sum, dtype=np.float32)
    confidence_map[valid] = confidence_sum[valid] / confidence_count[valid]
    confidence_map[instance_map == 0] = 0.0

    return instance_map, confidence_map
