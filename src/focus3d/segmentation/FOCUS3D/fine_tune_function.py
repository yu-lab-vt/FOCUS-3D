import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
from tqdm.auto import tqdm

def check_exists(path, name, allow_empty_string=False):
    if allow_empty_string and str(path).strip() == '':
        print(f'{name}: not used')
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{name} not found: {path}')

    print(f'{name}: {path}')
    return path

# ============================================================
# Cell 4. Patch extraction helper functions
# ============================================================


def read_volume(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in ['.tif', '.tiff']:
        arr = tifffile.imread(str(path))
    elif suffix == '.zarr':
        import zarr

        arr = np.asarray(zarr.open(str(path), mode='r'))
    else:
        raise ValueError(f'Unsupported file format: {path}')

    arr = np.squeeze(np.asarray(arr))

    if arr.ndim != 3:
        raise ValueError(
            f'Expected a 3D volume, got shape {arr.shape}: {path}'
        )

    return arr


def save_tif(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def compute_axis_starts(length, patch_len, stride_len):
    """
    Generate patch starts along one axis.

    Rules:
    - always include 0
    - step by stride
    - always include the last start so the tail is covered
    """
    length = int(length)
    patch_len = int(patch_len)
    stride_len = int(stride_len)

    if length <= patch_len:
        return [0]

    starts = list(range(0, length - patch_len + 1, stride_len))
    last_start = length - patch_len

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def generate_patch_coords(volume_shape, patch_size, stride):
    z, y, x = volume_shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    z_starts = compute_axis_starts(z, pz, sz)
    y_starts = compute_axis_starts(y, py, sy)
    x_starts = compute_axis_starts(x, px, sx)

    coords = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                coords.append((z0, y0, x0))

    return coords


def crop_with_padding(volume, start, patch_size, pad_mode):
    """
    Crop a 3D patch. If the input volume is smaller than the patch,
    pad the patch at the end of each axis.
    """
    z0, y0, x0 = start
    pz, py, px = patch_size

    z1 = min(z0 + pz, volume.shape[0])
    y1 = min(y0 + py, volume.shape[1])
    x1 = min(x0 + px, volume.shape[2])

    patch = volume[z0:z1, y0:y1, x0:x1]

    pad_z = pz - patch.shape[0]
    pad_y = py - patch.shape[1]
    pad_x = px - patch.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        patch = np.pad(
            patch,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode=pad_mode,
        )

    return patch


def relabel_instances(label_patch):
    """
    Relabel positive instance IDs to consecutive IDs within each patch.

    Background remains 0.
    """
    label_patch = np.asarray(label_patch)
    ids = np.unique(label_patch)
    ids = ids[ids > 0]

    if len(ids) == 0:
        return np.zeros_like(label_patch, dtype=np.uint16)

    out = np.zeros_like(label_patch, dtype=np.uint32)

    for new_id, old_id in enumerate(ids, start=1):
        out[label_patch == old_id] = new_id

    if out.max() <= np.iinfo(np.uint16).max:
        return out.astype(np.uint16)

    return out.astype(np.uint32)


def find_paired_files(image_dir, label_dir):
    """
    Match image and label files by file stem.

    Example:
    images/00000.tif
    labels/00000.tif
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_files = []
    for pat in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.zarr', '*.ZARR']:
        image_files.extend(image_dir.glob(pat))

    label_files = []
    for pat in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.zarr', '*.ZARR']:
        label_files.extend(label_dir.glob(pat))

    image_map = {p.stem: p for p in image_files}
    label_map = {p.stem: p for p in label_files}

    common_stems = sorted(set(image_map.keys()) & set(label_map.keys()))

    if len(common_stems) == 0:
        raise FileNotFoundError(
            'No paired image/label files found. '
            'Image and label files should have the same file stem.'
        )

    pairs = [(image_map[s], label_map[s]) for s in common_stems]
    return pairs


def normalize_volume_percentile(image, p_low=1, p_high=99):
    """
    Normalize a full 3D image volume to [0, 1] using global percentile clipping.
    """
    image = np.asarray(image, dtype=np.float32)

    lo, hi = np.percentile(image, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32), float(lo), float(hi)

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)

    return image.astype(np.float32), float(lo), float(hi)


def count_label_instances(label_patch):
    """
    Count instance number in a label patch.

    Example:
    label values {0, 10, 11} -> 2 instances.
    """
    ids = np.unique(label_patch)
    ids = ids[ids > 0]
    return int(len(ids))


def export_training_patches(
    source_image_dir,
    source_label_dir,
    patch_image_dir,
    patch_label_dir,
    patch_size,
    stride,
    min_label_instances=1,
    background_threshold=None,
    max_patches_per_volume=None,
    recreate=True,
    seed=2026,
    norm_p_low=1,
    norm_p_high=99,
):
    """
    Export paired image/label patches for FOCUS-3D fine-tuning.

    Logic:
    1. Read full image and label volume.
    2. Normalize the full image volume using global percentile clipping.
       Default: 1%-99%.
    3. Generate sliding-window patch coordinates.
    4. Crop image/label patches.
    5. Keep only patches with at least `min_label_instances` instances.
       Example: label values {0, 10, 11} means 2 instances.
    6. Optionally remove weak-background patches using normalized image intensity.
    7. Relabel instances inside each patch and save.
    """
    random.seed(seed)
    np.random.seed(seed)

    patch_image_dir = Path(patch_image_dir)
    patch_label_dir = Path(patch_label_dir)

    if recreate:
        if patch_image_dir.exists():
            shutil.rmtree(patch_image_dir)
        if patch_label_dir.exists():
            shutil.rmtree(patch_label_dir)

    patch_image_dir.mkdir(parents=True, exist_ok=True)
    patch_label_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_paired_files(source_image_dir, source_label_dir)

    print(f"Found {len(pairs)} paired image/label volume(s).")
    print("Patch image dir:", patch_image_dir)
    print("Patch label dir:", patch_label_dir)
    print(f"Image normalization: global percentile {norm_p_low}% - {norm_p_high}%")
    print(f"Minimum instances per patch: {min_label_instances}")

    summary = []
    total_saved = 0

    for image_path, label_path in tqdm(
        pairs,
        desc="Export patches",
        unit="vol",
    ):
        image_raw = read_volume(image_path)
        label = read_volume(label_path)

        if image_raw.shape != label.shape:
            raise ValueError(
                f"Image and label shape mismatch:\n"
                f"Image: {image_path}, shape={image_raw.shape}\n"
                f"Label: {label_path}, shape={label.shape}"
            )

        # ------------------------------------------------------------
        # Normalize the whole 3D volume before patch extraction
        # ------------------------------------------------------------
        image, norm_lo, norm_hi = normalize_volume_percentile(
            image_raw,
            p_low=norm_p_low,
            p_high=norm_p_high,
        )

        coords = generate_patch_coords(image.shape, patch_size, stride)

        valid_coords = []
        instance_counts = []

        for coord in coords:
            label_patch = crop_with_padding(
                label,
                coord,
                patch_size,
                pad_mode="constant",
            )

            n_instances = count_label_instances(label_patch)

            if n_instances < int(min_label_instances):
                continue

            if background_threshold is not None:
                image_patch = crop_with_padding(
                    image,
                    coord,
                    patch_size,
                    pad_mode="reflect",
                )

                # image_patch is already normalized to [0, 1]
                if float(np.percentile(image_patch, 99.9)) < float(background_threshold):
                    continue

            valid_coords.append(coord)
            instance_counts.append(n_instances)

        if max_patches_per_volume is not None and len(valid_coords) > int(max_patches_per_volume):
            selected_indices = random.sample(
                range(len(valid_coords)),
                int(max_patches_per_volume),
            )
            selected_indices = sorted(selected_indices)

            valid_coords = [valid_coords[i] for i in selected_indices]
            instance_counts = [instance_counts[i] for i in selected_indices]

        saved_this_volume = 0

        for coord in valid_coords:
            z0, y0, x0 = coord

            image_patch = crop_with_padding(
                image,
                coord,
                patch_size,
                pad_mode="reflect",
            )

            label_patch = crop_with_padding(
                label,
                coord,
                patch_size,
                pad_mode="constant",
            )

            # Relabel after filtering.
            # Example: {0, 10, 11} -> {0, 1, 2}
            label_patch = relabel_instances(label_patch)

            patch_name = (
                f"{image_path.stem}"
                f"_z{z0:04d}_y{y0:04d}_x{x0:04d}.tif"
            )

            save_tif(patch_image_dir / patch_name, image_patch)
            save_tif(patch_label_dir / patch_name, label_patch)

            saved_this_volume += 1
            total_saved += 1

        if len(instance_counts) > 0:
            min_instances_saved = int(np.min(instance_counts))
            max_instances_saved = int(np.max(instance_counts))
            mean_instances_saved = float(np.mean(instance_counts))
        else:
            min_instances_saved = 0
            max_instances_saved = 0
            mean_instances_saved = 0.0

        summary.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "shape": image.shape,
                "norm_p_low": norm_p_low,
                "norm_p_high": norm_p_high,
                "norm_lo": norm_lo,
                "norm_hi": norm_hi,
                "all_patches": len(coords),
                "saved_patches": saved_this_volume,
                "min_label_instances": min_label_instances,
                "min_instances_saved": min_instances_saved,
                "max_instances_saved": max_instances_saved,
                "mean_instances_saved": mean_instances_saved,
            }
        )

    print("\nPatch export finished.")
    print("Total saved patches:", total_saved)

    return summary

def cfg_value(x):
    """
    Convert Python values to strings accepted by Detectron2/YACS opts.
    """
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return 'True' if x else 'False'
    if isinstance(x, (list, tuple)):
        return str(list(x))
    return str(x)
