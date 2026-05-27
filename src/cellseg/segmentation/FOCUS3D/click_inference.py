import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def _is_windows_focus3d_backend():
    """
    Decide whether to use the Windows no-Detectron2 backend.

    Default:
        Windows -> inference_win.py
        Linux   -> inference.py

    Override:
        CELLSEG_FOCUS3D_BACKEND=windows
        CELLSEG_FOCUS3D_BACKEND=detectron2
    """
    backend = os.environ.get('CELLSEG_FOCUS3D_BACKEND', 'auto').strip().lower()

    if backend in {'windows', 'win', 'nod2', 'no_detectron2', 'pytorch'}:
        return True

    if backend in {'detectron2', 'd2', 'linux'}:
        return False

    return os.name == 'nt'


import sys

_FOCUS3D_DIR = Path(__file__).resolve().parent
if str(_FOCUS3D_DIR) not in sys.path:
    sys.path.insert(0, str(_FOCUS3D_DIR))

if _is_windows_focus3d_backend():
    # Windows / no-Detectron2 path
    if __package__:
        from .inference_win import normalize_img, read_volume
    else:
        from inference_win import normalize_img, read_volume
else:
    # Linux / Detectron2 path
    if __package__:
        from .inference import normalize_img, read_volume
    else:
        from inference import normalize_img, read_volume
# ============================================================
# Refine local area
# ============================================================


def _load_clicked_inference_image(image):
    """
    Support both image path and already-loaded ndarray.

    Returns
    -------
    image_zyx : np.ndarray
        3D image in Z/Y/X order.
    """
    if isinstance(image, (str, Path)):
        image_zyx = read_volume(image)
    else:
        image_zyx = np.asarray(image)

    image_zyx = np.squeeze(image_zyx)

    if image_zyx.ndim != 3:
        raise ValueError(
            f'Clicked inference expects a 3D image in Z/Y/X order, '
            f'got shape={image_zyx.shape}.'
        )

    return image_zyx.astype(np.float32, copy=False)


def crop_centered_patch_3d(
    volume: np.ndarray,
    center_zyx: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (32, 96, 96),
    pad_mode: str = 'reflect',
) -> Tuple[np.ndarray, Dict]:
    """
    Crop a 3D patch centered at the given coordinate.

    If the patch crosses the image boundary, padding is applied.
    The returned patch always has shape patch_size.

    Args:
        volume: 3D volume with shape (Z, Y, X).
        center_zyx: Click coordinate in original image coordinates.
        patch_size: Patch size in (Z, Y, X).
        pad_mode: Numpy padding mode. Usually "reflect" or "edge".

    Returns:
        patch: Cropped patch with shape patch_size.
        info: Metadata for mapping patch coordinates back to original volume.
    """
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3D volume, got shape {volume.shape}')

    z_size, y_size, x_size = volume.shape
    pz, py, px = patch_size
    cz, cy, cx = [int(v) for v in center_zyx]

    if not (0 <= cz < z_size and 0 <= cy < y_size and 0 <= cx < x_size):
        raise ValueError(
            f'center_zyx={center_zyx} is outside volume shape {volume.shape}'
        )

    # For even patch sizes, the clicked voxel is placed at patch_size // 2.
    start_z = cz - pz // 2
    start_y = cy - py // 2
    start_x = cx - px // 2

    end_z = start_z + pz
    end_y = start_y + py
    end_x = start_x + px

    pad_before_z = max(0, -start_z)
    pad_before_y = max(0, -start_y)
    pad_before_x = max(0, -start_x)

    pad_after_z = max(0, end_z - z_size)
    pad_after_y = max(0, end_y - y_size)
    pad_after_x = max(0, end_x - x_size)

    pad_width = (
        (pad_before_z, pad_after_z),
        (pad_before_y, pad_after_y),
        (pad_before_x, pad_after_x),
    )

    # Reflect padding may fail for degenerate dimensions, so edge padding is safer there.
    actual_pad_mode = pad_mode
    if pad_mode == 'reflect' and any(s <= 1 for s in volume.shape):
        actual_pad_mode = 'edge'

    if any(p[0] > 0 or p[1] > 0 for p in pad_width):
        padded = np.pad(volume, pad_width=pad_width, mode=actual_pad_mode)
    else:
        padded = volume

    crop_z0 = start_z + pad_before_z
    crop_y0 = start_y + pad_before_y
    crop_x0 = start_x + pad_before_x

    patch = padded[
        crop_z0 : crop_z0 + pz,
        crop_y0 : crop_y0 + py,
        crop_x0 : crop_x0 + px,
    ]

    if patch.shape != tuple(patch_size):
        raise RuntimeError(
            f'Patch shape mismatch: got {patch.shape}, expected {patch_size}'
        )

    click_local_zyx = (
        cz - start_z,
        cy - start_y,
        cx - start_x,
    )

    # Valid region in original volume coordinates.
    valid_global_z0 = max(start_z, 0)
    valid_global_y0 = max(start_y, 0)
    valid_global_x0 = max(start_x, 0)

    valid_global_z1 = min(end_z, z_size)
    valid_global_y1 = min(end_y, y_size)
    valid_global_x1 = min(end_x, x_size)

    # Corresponding valid region in patch coordinates.
    valid_patch_z0 = valid_global_z0 - start_z
    valid_patch_y0 = valid_global_y0 - start_y
    valid_patch_x0 = valid_global_x0 - start_x

    valid_patch_z1 = valid_patch_z0 + (valid_global_z1 - valid_global_z0)
    valid_patch_y1 = valid_patch_y0 + (valid_global_y1 - valid_global_y0)
    valid_patch_x1 = valid_patch_x0 + (valid_global_x1 - valid_global_x0)

    info = {
        'patch_start_zyx': (int(start_z), int(start_y), int(start_x)),
        'patch_end_zyx': (int(end_z), int(end_y), int(end_x)),
        'click_global_zyx': (int(cz), int(cy), int(cx)),
        'click_local_zyx': tuple(int(v) for v in click_local_zyx),
        'valid_global_slices': (
            slice(valid_global_z0, valid_global_z1),
            slice(valid_global_y0, valid_global_y1),
            slice(valid_global_x0, valid_global_x1),
        ),
        'valid_patch_slices': (
            slice(valid_patch_z0, valid_patch_z1),
            slice(valid_patch_y0, valid_patch_y1),
            slice(valid_patch_x0, valid_patch_x1),
        ),
        'pad_width': pad_width,
    }

    return patch.astype(np.float32, copy=False), info


def select_clicked_query_instance(
    model_output: Dict,
    click_local_zyx: Tuple[int, int, int],
    patch_shape: Tuple[int, int, int] = (32, 96, 96),
    score_thresh: float = 0.05,
    click_prob_thresh: float = 0.10,
    mask_thresh: float = 0.50,
    min_voxels: int = 20,
    keep_clicked_component: bool = True,
    adaptive_threshold_if_needed: bool = True,
    min_adaptive_thresh: float = 0.05,
) -> Dict:
    """
    Select the query corresponding to the clicked point and return one instance mask.

    Query ranking score:
        rank_score = click_prob * pred_score * mask_quality

    Args:
        model_output: One output dictionary from model([input])[0].
        click_local_zyx: Click coordinate inside the cropped patch.
        patch_shape: Expected patch shape in (Z, Y, X).
        score_thresh: Minimum query classification score.
        click_prob_thresh: Minimum mask probability at the clicked voxel.
        mask_thresh: Default mask binarization threshold.
        min_voxels: Minimum size of the final instance.
        keep_clicked_component: If True, only keep the connected component containing the click.
        adaptive_threshold_if_needed: If True, lower the mask threshold when the clicked voxel
            is below mask_thresh but still above click_prob_thresh.
        min_adaptive_thresh: Lower bound of adaptive threshold.

    Returns:
        A dictionary containing selected mask, probability map, query index, and debug info.
    """
    D, H, W = patch_shape
    cz, cy, cx = [int(v) for v in click_local_zyx]

    if not (0 <= cz < D and 0 <= cy < H and 0 <= cx < W):
        raise ValueError(
            f'click_local_zyx={click_local_zyx} is outside patch shape {patch_shape}'
        )

    empty_mask = np.zeros(patch_shape, dtype=np.uint8)
    empty_prob = np.zeros(patch_shape, dtype=np.float32)

    if 'pred_scores' not in model_output or 'pred_masks' not in model_output:
        return {
            'success': False,
            'reason': 'model_output does not contain pred_scores and pred_masks',
            'instance_mask_patch': empty_mask,
            'instance_prob_patch': empty_prob,
            'selected_query': None,
        }

    scores = model_output['pred_scores'].detach().float()
    masks = model_output['pred_masks'].detach().float()

    # Make scores shape (Q,).
    if scores.ndim > 1:
        scores = scores.max(dim=-1).values
    scores = scores.flatten()

    # Accept masks with shape (Q, D, H, W) or (Q, 1, D, H, W).
    if masks.ndim == 5 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 4:
        raise ValueError(
            f'Expected pred_masks with shape (Q,D,H,W), got {masks.shape}'
        )

    if masks.shape[0] != scores.shape[0]:
        raise ValueError(
            f'Number of masks and scores mismatch: masks={masks.shape[0]}, scores={scores.shape[0]}'
        )

    if masks.numel() == 0 or scores.numel() == 0:
        return {
            'success': False,
            'reason': 'empty pred_masks or pred_scores',
            'instance_mask_patch': empty_mask,
            'instance_prob_patch': empty_prob,
            'selected_query': None,
        }

    # Resize masks to patch_shape if needed.
    if tuple(masks.shape[-3:]) != tuple(patch_shape):
        masks = F.interpolate(
            masks[:, None],
            size=patch_shape,
            mode='trilinear',
            align_corners=False,
        )[:, 0]

    # Convert logits to probabilities if needed.
    if bool((masks.min() < 0 or masks.max() > 1).item()):
        mask_prob = masks.sigmoid()
    else:
        mask_prob = masks.clamp(0.0, 1.0)

    # Probability of each query at the clicked voxel.
    # Use a local neighborhood around the clicked voxel instead of one single voxel.
    # This is more robust when the user clicks on a boundary, membrane, weak signal,
    # or a small local hole inside the predicted mask.
    click_radius_zyx = (1, 3, 3)  # local window size: z +/-1, y/x +/-3

    rz, ry, rx = click_radius_zyx

    z0 = max(0, cz - rz)
    z1 = min(D, cz + rz + 1)

    y0 = max(0, cy - ry)
    y1 = min(H, cy + ry + 1)

    x0 = max(0, cx - rx)
    x1 = min(W, cx + rx + 1)

    local_prob = mask_prob[:, z0:z1, y0:y1, x0:x1]

    # Max response in the local click neighborhood.
    click_prob = local_prob.flatten(1).max(dim=1).values

    # Same mask-quality definition as your current patch postprocess.
    mask_bin = mask_prob > mask_thresh
    mask_area = mask_bin.flatten(1).sum(dim=1)

    mask_quality = (mask_prob.flatten(1) * mask_bin.flatten(1)).sum(dim=1) / (
        mask_area + 1e-6
    )

    # Rank queries by whether they explain the clicked point.
    rank_score = (
        click_prob * scores.clamp(min=0.0) * mask_quality.clamp(min=0.0)
    )

    valid = torch.ones_like(rank_score, dtype=torch.bool)
    valid &= scores >= float(score_thresh)
    valid &= click_prob >= float(click_prob_thresh)
    valid &= mask_area >= int(min_voxels)

    if valid.any():
        rank_score_valid = rank_score.clone()
        rank_score_valid[~valid] = -1.0
        selected_query = int(torch.argmax(rank_score_valid).item())
    else:
        # Fallback: choose the query with the highest clicked-point probability.
        selected_query = int(torch.argmax(click_prob).item())

        best_click_prob = float(click_prob[selected_query].item())

        # Do not fail too early. For clicked-instance inference, a low click probability
        # can still be useful if this is the best query around the clicked region.
        if best_click_prob < float(click_prob_thresh):
            print(
                f'[Warning] best local click probability is low: '
                f'{best_click_prob:.4f} < click_prob_thresh={click_prob_thresh:.4f}. '
                f'Still trying to return the best local query.'
            )

    selected_prob = (
        mask_prob[selected_query].detach().cpu().numpy().astype(np.float32)
    )
    selected_click_prob = float(click_prob[selected_query].item())
    selected_score = float(scores[selected_query].item())
    selected_mask_quality = float(mask_quality[selected_query].item())
    selected_rank_score = float(rank_score[selected_query].item())

    # If the clicked voxel is slightly below mask_thresh, use a lower threshold.
    used_mask_thresh = float(mask_thresh)
    if adaptive_threshold_if_needed and selected_click_prob < mask_thresh:
        used_mask_thresh = max(
            float(min_adaptive_thresh),
            float(selected_click_prob) * 0.85,
        )

    instance_mask = selected_prob > used_mask_thresh

    # Keep only the connected component that contains the clicked voxel.
    if keep_clicked_component:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        labeled_cc, num_cc = ndi.label(instance_mask, structure=structure)

        clicked_cc_id = int(labeled_cc[cz, cy, cx])

        if clicked_cc_id > 0:
            instance_mask = labeled_cc == clicked_cc_id
        else:
            return {
                'success': False,
                'reason': 'selected query mask does not contain the clicked voxel after thresholding',
                'instance_mask_patch': empty_mask,
                'instance_prob_patch': selected_prob,
                'selected_query': selected_query,
                'click_prob': selected_click_prob,
                'score': selected_score,
                'mask_quality': selected_mask_quality,
                'rank_score': selected_rank_score,
                'used_mask_thresh': used_mask_thresh,
            }

    if int(instance_mask.sum()) < int(min_voxels):
        return {
            'success': False,
            'reason': f'selected instance is too small: {int(instance_mask.sum())} voxels',
            'instance_mask_patch': empty_mask,
            'instance_prob_patch': selected_prob,
            'selected_query': selected_query,
            'click_prob': selected_click_prob,
            'score': selected_score,
            'mask_quality': selected_mask_quality,
            'rank_score': selected_rank_score,
            'used_mask_thresh': used_mask_thresh,
        }

    return {
        'success': True,
        'reason': 'ok',
        'instance_mask_patch': instance_mask.astype(np.uint8),
        'instance_prob_patch': selected_prob,
        'selected_query': selected_query,
        'click_prob': selected_click_prob,
        'score': selected_score,
        'mask_quality': selected_mask_quality,
        'rank_score': selected_rank_score,
        'used_mask_thresh': used_mask_thresh,
    }


def infer_clicked_instance(
    image,
    coord_zyx,
    model,
    patch_size=(32, 96, 96),
    lower_percentile=0.0,
    upper_percentile=100.0,
    normalize=True,
    image_is_normalized=False,
    pad_mode='reflect',
    score_thresh=0.05,
    click_prob_thresh=0.10,
    mask_thresh=0.50,
    min_voxels=20,
    keep_clicked_component=True,
    return_full_size=True,
    show_result=True,
    z_show_radius=2,
    use_amp=True,
    amp_dtype='float16',
):
    """
    Infer only one clicked instance from a large 3D image.

    Timing definition:
        The timer starts immediately after the image has been successfully loaded
        or accepted as a numpy array. It includes normalization, patch cropping,
        model forward, query selection, full-size mask construction, and optional
        visualization preparation.

    Args:
        image: Input 3D volume path or numpy array with shape (Z, Y, X).
        coord_zyx: Click coordinate in original image coordinates.
        model: Loaded Mask2Former-style model.
        patch_size: Local patch size, default is (32, 96, 96).
        lower_percentile: Lower percentile for normalization.
        upper_percentile: Upper percentile for normalization.
        normalize: Whether to normalize the input volume.
        image_is_normalized: Set True if image is already normalized.
        pad_mode: Padding mode for boundary clicks.
        score_thresh: Minimum query classification score.
        click_prob_thresh: Minimum query probability at the clicked point.
        mask_thresh: Mask binarization threshold.
        min_voxels: Minimum final instance size.
        keep_clicked_component: Keep only the connected component containing the click.
        return_full_size: If True, return a full-size binary mask.
        show_result: If True, show z-5 to z+5 segmentation overlay.
        z_show_radius: Number of z-slices before and after the clicked z to show.
        use_amp: Use mixed precision on CUDA.
        amp_dtype: "float16" or "bfloat16".

    Returns:
        result: Dictionary containing the selected instance mask, selected query,
                timing, crop information, and debug information.
    """
    # ------------------------------------------------------------
    # 1. Read or accept image
    # ------------------------------------------------------------
    if isinstance(image, (str, Path)):
        raw_volume = read_volume(image)
    else:
        raw_volume = np.asarray(image)
        raw_volume = np.squeeze(raw_volume)

        if raw_volume.ndim != 3:
            raise ValueError(
                f'Expected a 3D image after squeeze, got shape {raw_volume.shape}'
            )

        raw_volume = raw_volume.astype(np.float32, copy=False)

    # Start timing after image import succeeds.
    t_after_image_import = time.time()

    # ------------------------------------------------------------
    # 2. Normalize image
    # ------------------------------------------------------------
    # First crop raw patch from the original image.
    raw_patch, crop_info = crop_centered_patch_3d(
        raw_volume,
        center_zyx=coord_zyx,
        patch_size=patch_size,
        pad_mode=pad_mode,
    )

    # Then normalize only this small patch.
    if normalize and not image_is_normalized:
        patch, norm_stats = normalize_img(
            raw_patch,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
    else:
        patch = raw_patch.astype(np.float32, copy=False)
        norm_stats = None

    # ------------------------------------------------------------
    # 4. Run one-patch model inference
    # ------------------------------------------------------------
    device = _get_model_device(model)
    model.eval()

    if amp_dtype == 'bfloat16':
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    amp_enabled = bool(use_amp and device.type == 'cuda')

    sample = {
        'image': torch.from_numpy(patch)
        .unsqueeze(0)
        .to(device, non_blocking=True),
        'coord': crop_info['patch_start_zyx'],
    }

    t_model_start = time.time()

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type='cuda',
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ),
    ):
        outputs = model([sample])

    if device.type == 'cuda':
        torch.cuda.synchronize()

    model_forward_time_sec = time.time() - t_model_start

    model_output = outputs[0]

    # ------------------------------------------------------------
    # 5. Select the query corresponding to the clicked point
    # ------------------------------------------------------------
    selected = select_clicked_query_instance(
        model_output=model_output,
        click_local_zyx=crop_info['click_local_zyx'],
        patch_shape=patch_size,
        score_thresh=score_thresh,
        click_prob_thresh=click_prob_thresh,
        mask_thresh=mask_thresh,
        min_voxels=min_voxels,
        keep_clicked_component=keep_clicked_component,
    )

    result = {
        **selected,
        'crop_info': crop_info,
        'norm_stats': norm_stats,
        'patch': patch,
        'model_output': model_output,
        'model_forward_time_sec': float(model_forward_time_sec),
    }

    # ------------------------------------------------------------
    # 6. Build full-size mask
    # ------------------------------------------------------------
    if return_full_size or show_result:
        full_mask = np.zeros(raw_volume.shape, dtype=np.uint8)

        if selected['success']:
            valid_global_slices = crop_info['valid_global_slices']
            valid_patch_slices = crop_info['valid_patch_slices']

            full_mask[valid_global_slices] = selected['instance_mask_patch'][
                valid_patch_slices
            ]

        result['instance_mask_full'] = full_mask

    # ------------------------------------------------------------
    # 7. Record total clicked-instance inference time
    # ------------------------------------------------------------
    clicked_infer_time_sec = time.time() - t_after_image_import
    result['clicked_infer_time_sec'] = float(clicked_infer_time_sec)

    print(
        f'[Clicked inference] total time after image import: '
        f'{clicked_infer_time_sec:.4f} sec'
    )
    print(
        f'[Clicked inference] model forward time: '
        f'{model_forward_time_sec:.4f} sec'
    )

    if selected.get('success', False):
        print(
            f'[Clicked inference] selected_query={selected.get("selected_query")}, '
            f'click_prob={selected.get("click_prob", 0):.4f}, '
            f'score={selected.get("score", 0):.4f}, '
            f'mask_quality={selected.get("mask_quality", 0):.4f}'
        )
    else:
        print(f'[Clicked inference] failed: {selected.get("reason")}')

    # ------------------------------------------------------------
    # 8. Show z-5 to z+5 visualization
    # ------------------------------------------------------------
    if show_result:
        show_clicked_instance_z_slices(
            raw_volume=raw_volume,
            instance_mask_full=result['instance_mask_full'],
            coord_zyx=coord_zyx,
            crop_info=crop_info,
            z_radius=z_show_radius,
        )

    return result


def show_clicked_instance_z_slices(
    raw_volume: np.ndarray,
    instance_mask_full: np.ndarray,
    coord_zyx,
    crop_info: dict,
    z_radius: int = 2,
    alpha: float = 0.45,
    figsize_per_slice: float = 2.2,
):
    """
    Show segmentation results from z-2 to z+2 around the clicked point.

    Args:
        raw_volume: Original raw image with shape (Z, Y, X).
        instance_mask_full: Binary instance mask with the same shape as raw_volume.
        coord_zyx: Click coordinate in original image coordinates.
        crop_info: Crop metadata returned by crop_centered_patch_3d().
        z_radius: Number of z-slices shown before and after the clicked z.
        alpha: Overlay transparency.
        figsize_per_slice: Figure width for each subplot.
    """
    z_size, y_size, x_size = raw_volume.shape
    cz, cy, cx = [int(v) for v in coord_zyx]

    z0 = max(0, cz - z_radius)
    z1 = min(z_size, cz + z_radius + 1)
    z_list = list(range(z0, z1))

    # Show only the XY region corresponding to the local patch.
    valid_global_slices = crop_info['valid_global_slices']
    _, y_slice, x_slice = valid_global_slices

    n = len(z_list)
    n_cols = min(n, 6)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_slice * n_cols, figsize_per_slice * n_rows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis('off')

    for idx, z in enumerate(z_list):
        ax = axes.ravel()[idx]

        img2d = raw_volume[z, y_slice, x_slice]
        mask2d = instance_mask_full[z, y_slice, x_slice].astype(bool)

        ax.imshow(img2d, cmap='gray')

        # Mask overlay. Non-mask pixels are transparent.
        # Build a pure red RGBA overlay for the instance mask.
        red_overlay = np.zeros((*mask2d.shape, 4), dtype=np.float32)
        red_overlay[mask2d, 0] = 1.0  # Red channel
        red_overlay[mask2d, 1] = 0.0  # Green channel
        red_overlay[mask2d, 2] = 0.0  # Blue channel
        red_overlay[mask2d, 3] = alpha  # Alpha channel

        ax.imshow(red_overlay)

        # Mark clicked point only on the clicked z-slice.
        if z == cz:
            local_y = cy - y_slice.start
            local_x = cx - x_slice.start
            ax.scatter([local_x], [local_y], c='cyan', s=25, marker='x')

        ax.set_title(f'z={z}', fontsize=9)
        ax.axis('off')

    fig.suptitle(
        f'Clicked instance visualization | center z={cz}, y={cy}, x={cx}',
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device where the model parameters are located.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
