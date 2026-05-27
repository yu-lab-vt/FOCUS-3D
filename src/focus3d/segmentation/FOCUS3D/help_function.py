import glob
import os
import random
import shutil
import types
from math import ceil
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tifffile
import torch
import torch.nn.functional as F
import zarr
from inference import patch_postprocess_argmax
from scipy import ndimage
from scipy.ndimage import binary_erosion, gaussian_filter, zoom


# --------------------------------------------------------------------
# Data/file management
# --------------------------------------------------------------------
def manage_files(source_dir, target_dir=None, num_files=None):
    """
    统计源目录中的文件数目，并将指定数量的随机文件移动到目标目录。

    Parameters
    ----------
    source_dir : str
        源目录路径。
    target_dir : str, optional
        目标目录路径。若为 None 则不执行移动。
    num_files : int, optional
        要移动的文件数量。若为 None 或大于文件总数，则移动所有文件。

    Returns
    -------
    tuple
        (total_files, moved_files) 分别为源目录文件总数和实际移动的文件数。
    """
    # 检查源目录
    if not os.path.isdir(source_dir):
        print(f'错误：源目录不存在 - {source_dir}')
        return 0, 0

    # 获取所有文件（不包括子目录）
    files = [
        f
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
    ]
    total = len(files)
    print(f'源目录中共有 {total} 个文件。')

    # 如果没有指定移动目标或数量为0，则不移动
    if target_dir is None or num_files == 0:
        return total, 0

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 确定要移动的文件数量
    if num_files is None or num_files > total:
        num_to_move = total
    else:
        num_to_move = num_files

    # 随机选择文件
    selected = random.sample(files, num_to_move)

    # 移动文件
    moved = 0
    for filename in selected:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)
        try:
            shutil.move(src, dst)
            moved += 1
        except Exception as e:
            print(f'移动文件 {filename} 时出错: {e}')

    print(f'已移动 {moved} 个文件到 {target_dir}')
    return total, moved


def copy_random_files(
    src_images_dir: str,
    src_labels_dir: str,
    dst_images_dir: str,
    dst_labels_dir: str,
    num_files: int = 2000,
    seed: int = 42,
):
    """
    从源图像目录随机选取 num_files 个 .tif 文件，复制到目标图像目录，
    同时将对应的标签文件（同名）从源标签目录复制到目标标签目录。

    Args:
        src_images_dir: 源图像文件夹路径
        src_labels_dir: 源标签文件夹路径
        dst_images_dir: 目标图像文件夹路径
        dst_labels_dir: 目标标签文件夹路径
        num_files: 要复制的文件数量
        seed: 随机种子，用于可重复性
    """
    # 创建目标目录（如果不存在）
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    # 获取所有 .tif 文件（包括 .tiff）
    image_paths = []
    for ext in ['.tif', '.tiff']:
        image_paths.extend(Path(src_images_dir).glob(f'*{ext}'))

    # 只保留文件（排除目录）
    image_paths = [p for p in image_paths if p.is_file()]

    total = len(image_paths)
    if total == 0:
        print('源图像目录中没有找到 .tif 文件')
        return

    # 如果请求的数量超过总数，则复制全部
    if num_files > total:
        print(
            f'请求数量 ({num_files}) 超过总文件数 ({total})，将复制全部文件。'
        )
        num_files = total

    # 随机选择
    random.seed(seed)
    selected = random.sample(image_paths, num_files)

    copied = 0
    for img_path in selected:
        # 获取文件名（不含扩展名）
        base = img_path.stem
        ext = img_path.suffix

        # 构造对应的标签路径（假设标签文件名相同，扩展名相同）
        label_path = Path(src_labels_dir) / f'{base}{ext}'
        if not label_path.exists():
            print(f'警告：标签文件 {label_path} 不存在，跳过 {img_path.name}')
            continue

        # 目标路径
        dst_img = Path(dst_images_dir) / img_path.name
        dst_label = Path(dst_labels_dir) / label_path.name

        # 复制
        shutil.copy2(str(img_path), str(dst_img))
        shutil.copy2(str(label_path), str(dst_label))
        copied += 1

    print(f'完成！共复制 {copied} 个图像及其对应标签。')


def remove_unmatched_images(images_dir, labels_dir):
    """
    删除 images_dir 中那些在 labels_dir 中没有对应标签文件的图像文件。

    Args:
        images_dir (str): 图像文件夹路径
        labels_dir (str): 标签文件夹路径

    Returns:
        list: 被删除的文件路径列表
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise FileNotFoundError(f'图像目录不存在: {images_dir}')
    if not labels_path.exists():
        raise FileNotFoundError(f'标签目录不存在: {labels_dir}')

    # 获取图像文件名（不含扩展名）
    image_names = set()
    for ext in ['.tif', '.tiff']:
        for f in images_path.glob(f'*{ext}'):
            image_names.add(f.stem)

    # 获取标签文件名（不含扩展名）
    label_names = set()
    for ext in ['.tif', '.tiff']:
        for f in labels_path.glob(f'*{ext}'):
            label_names.add(f.stem)

    # 找出在图像中存在但标签中不存在的文件
    to_delete = image_names - label_names

    deleted_files = []
    for name in to_delete:
        for ext in ['.tif', '.tiff']:
            file_path = images_path / f'{name}{ext}'
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
                print(f'已删除: {file_path}')

    print(f'\n共删除 {len(deleted_files)} 个文件')
    return deleted_files


def tif_to_zarr(tif_path, zarr_path, crop_zyx=None, chunks=(16, 256, 256)):
    img = tifffile.imread(tif_path)
    print('input shape:', img.shape, 'dtype:', img.dtype)

    if img.ndim != 3:
        raise ValueError(f'Expected 3D TIFF, got shape={img.shape}')

    # 这里虽然 tifffile 报 axes='QYX'，但如果你确认第一维就是 z-stack，
    # 就直接按 ZYX 理解，不需要 transpose。
    img = np.asarray(img)

    if crop_zyx is not None:
        if len(crop_zyx) != 3:
            raise ValueError(
                'crop_zyx must be a tuple of three slices (z, y, x).'
            )
        img = img[crop_zyx]
        print('cropped shape:', img.shape)

    if 0 in img.shape:
        raise ValueError(f'Crop produced empty array: shape={img.shape}')

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    chunks = tuple(min(s, c) for s, c in zip(img.shape, chunks))

    z = zarr.open(
        zarr_path,
        mode='w',
        shape=img.shape,
        dtype=img.dtype,
        chunks=chunks,
        zarr_version=2,
    )

    z[:] = img
    z.store.close()  # 尽量显式关闭

    print(f'Conversion complete: {tif_path} -> {zarr_path}')


def convert_instance_seg_mat(
    input_dir, output_dir, output_format='tif', dtype=np.uint32
):
    """
    Convert instance segmentation results from .mat files to .tif or .zarr files.

    Supports:
        - standard .mat files loaded by scipy.io.loadmat
        - HDF5-based MATLAB v7.3 .mat files loaded by h5py

    Parameters
    ----------
    input_dir : str
        Path to the folder containing input .mat files.

    output_dir : str
        Path to the folder where converted files will be saved.

    output_format : str
        Output format. Options:
            - "tif"  : save as .tif
            - "zarr" : save as .zarr

    dtype : numpy dtype
        Output dtype. For instance segmentation, np.uint32 is recommended.
        Use np.uint16 only if instance IDs are guaranteed <= 65535.
    """

    output_format = output_format.lower()

    if output_format not in ['tif', 'tiff', 'zarr']:
        raise ValueError(
            f'Unsupported output_format: {output_format}. '
            "Expected 'tif', 'tiff', or 'zarr'."
        )

    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(input_dir, '*.mat')))

    if not mat_files:
        print(f'No .mat files found in {input_dir}')
        return

    print(f'Found {len(mat_files)} .mat files to process')

    for mat_path in mat_files:
        try:
            volume = None

            # Try MATLAB v7.3 / HDF5 .mat first
            try:
                with h5py.File(mat_path, 'r') as f:
                    for key in f.keys():
                        data = f[key]

                        if isinstance(data, h5py.Dataset) and data.ndim == 3:
                            volume = np.asarray(data[()])

                            # Keep your original orientation correction
                            volume = np.transpose(volume, (0, 2, 1))
                            break

            except OSError:
                # Fall back to standard MATLAB .mat
                mat_data = scipy.io.loadmat(mat_path)

                for key, value in mat_data.items():
                    if key.startswith('__'):
                        continue

                    if (
                        isinstance(value, np.ndarray)
                        and value.ndim == 3
                        and value.size > 100
                    ):
                        volume = value
                        break

            if volume is None:
                print(
                    f'Warning: No valid 3D volume found in {mat_path}, skipping...'
                )
                continue

            volume_out = volume.astype(dtype)

            base_name = os.path.splitext(os.path.basename(mat_path))[0]

            if output_format in ['tif', 'tiff']:
                output_path = os.path.join(output_dir, f'{base_name}.tif')
                tifffile.imwrite(output_path, volume_out)

            elif output_format == 'zarr':
                output_path = os.path.join(output_dir, f'{base_name}.zarr')
                zarr.save(output_path, volume_out)

            print(
                f'Converted: {mat_path} -> {output_path} '
                f'(Shape: {volume_out.shape}, dtype: {volume_out.dtype})'
            )

        except Exception as e:
            print(f'Error processing {mat_path}: {e}')
            continue

    print('\nConversion complete.')


# --------------------------------------------------------------------
# Label checking
# --------------------------------------------------------------------
def check_label_continuity(label_folder):
    """
    检查标签文件夹中的 TIFF 文件，确保：
    1) 标签值连续（0, 1, 2, ..., max_label 全部出现）
    2) 最大标签值 ≥ 10

    Args:
        label_folder (str): 存放标签 TIFF 文件的文件夹路径

    Returns:
        dict: 包含总体状态和每个文件详细信息的字典
    """
    label_path = Path(label_folder)
    if not label_path.exists():
        raise FileNotFoundError(f'文件夹不存在: {label_folder}')

    # 获取所有 .tif 和 .tiff 文件
    tif_files = list(label_path.glob('*.tif')) + list(
        label_path.glob('*.tiff')
    )
    if not tif_files:
        print(f'警告: 在 {label_folder} 中未找到任何 TIFF 文件')
        return {'status': 'no_files', 'files': []}

    results = []
    all_passed = True

    for tif_file in tif_files:
        label_array = tifffile.imread(tif_file)

        # 统计唯一值
        max_label = int(label_array.max())
        unique_count = len(np.unique(label_array))

        # 条件1: 唯一值数量 == 最大标签值 + 1
        condition1 = unique_count == max_label + 1
        # 条件2: 最大标签值 ≥ 10
        condition2 = max_label >= 10

        passed = condition1 and condition2
        if not passed:
            all_passed = False

        # 确定失败原因
        if not condition1:
            reason = f'标签不连续: 唯一值数量 {unique_count}，最大标签 {max_label}，期望 {max_label + 1}'
        elif not condition2:
            reason = f'最大标签 {max_label} < 10'
        else:
            reason = None

        results.append(
            {
                'file': tif_file.name,
                'passed': passed,
                'unique_count': unique_count,
                'max_label': max_label,
                'reason': reason,
                'error': None,
            }
        )

    # 打印汇总信息
    print('\n=== 标签连续性检查 ===')
    print(f'文件夹: {label_folder}')
    print(f'总文件数: {len(tif_files)}')
    print(f'通过: {sum(1 for r in results if r["passed"])}')
    print(f'失败: {sum(1 for r in results if not r["passed"])}')
    print('\n详细结果:')
    for r in results:
        status = '✓ 通过' if r['passed'] else '✗ 失败'
        if r['error']:
            print(f'  {r["file"]}: {status} (错误: {r["error"]})')
        elif r['reason']:
            print(f'  {r["file"]}: {status} ({r["reason"]})')
    #     else:
    #         print(f"  {r['file']}: {status} (唯一值数量={r['unique_count']}, 最大标签={r['max_label']})")

    return {
        'status': 'passed' if all_passed else 'failed',
        'total_files': len(tif_files),
        'passed_count': sum(1 for r in results if r['passed']),
        'failed_count': sum(1 for r in results if not r['passed']),
        'results': results,
    }


def remap_labels(input_dir, output_dir=None, min_max_label=10):
    """
    将标签文件重新映射为连续标签（背景0保持不变，前景从1开始连续），
    并可选地过滤掉最大标签小于阈值的文件。

    Args:
        input_dir (str): 输入标签文件夹路径
        output_dir (str, optional): 输出文件夹路径。如果为None，则覆盖原文件
        min_max_label (int): 最小最大标签阈值，只有max_label >= min_max_label的文件才会被保存

    Returns:
        dict: 包含处理统计信息的字典
            - 'total_files': 总处理文件数
            - 'saved_files': 实际保存的文件数（满足阈值）
            - 'filtered_files': 因不满足阈值而被跳过的文件数
            - 'details': 每个文件的处理详情列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f'输入目录不存在: {input_dir}')

    # 创建输出目录（如果指定且与原目录不同）
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path  # 覆盖原文件时直接使用输入目录

    # 获取所有tif文件
    tif_files = sorted(
        [f for f in input_path.glob('*.tif')]
        + [f for f in input_path.glob('*.tiff')]
    )
    if not tif_files:
        print(f'警告: 在 {input_dir} 中未找到任何TIFF文件')
        return {
            'total_files': 0,
            'saved_files': 0,
            'filtered_files': 0,
            'details': [],
        }

    results = []
    saved_count = 0
    filtered_count = 0

    for tif_file in tif_files:
        try:
            label = tifffile.imread(tif_file)
        except Exception as e:
            print(f'读取 {tif_file.name} 时出错: {e}')
            results.append(
                {'file': tif_file.name, 'status': 'error', 'error': str(e)}
            )
            continue

        # 获取所有唯一值
        unique_ids = np.unique(label)
        foreground_ids = unique_ids[unique_ids != 0]

        # 创建映射：原始ID -> 新ID
        mapping = {0: 0}
        for new_id, old_id in enumerate(foreground_ids, start=1):
            mapping[old_id] = new_id

        # 应用映射
        new_label = np.zeros_like(label)
        for old_id, new_id in mapping.items():
            new_label[label == old_id] = new_id

        max_label = new_label.max()
        # 检查是否满足阈值
        if max_label >= min_max_label:
            # 保存文件
            output_file = output_path / tif_file.name
            tifffile.imwrite(output_file, new_label)
            saved_count += 1
            status = 'saved'
        else:
            filtered_count += 1
            status = 'filtered'

        results.append(
            {
                'file': tif_file.name,
                'status': status,
                'original_unique': unique_ids.tolist(),
                'new_unique': np.unique(new_label).tolist(),
                'max_label': int(max_label),
            }
        )

    print('\n=== 标签重映射完成 ===')
    print(f'输入目录: {input_dir}')
    print(f'输出目录: {output_dir if output_dir else input_dir}')
    print(f'总文件数: {len(tif_files)}')
    print(f'保存文件数: {saved_count}')
    print(f'过滤文件数: {filtered_count} (max_label < {min_max_label})')

    return {
        'total_files': len(tif_files),
        'saved_files': saved_count,
        'filtered_files': filtered_count,
        'details': results,
    }


def get_max_label_in_folder(label_dir, threshold=300):
    """
    扫描指定文件夹中的 TIFF 标签文件，返回所有文件中的最大标签值，
    并输出最大标签值超过 threshold 的文件名。

    Args:
        label_dir (str): 存放标签 TIFF 文件的文件夹路径
        threshold (int): 判断是否输出文件名的阈值，默认 300

    Returns:
        global_max (int): 所有标签文件中的最大值，如果没有找到文件则返回 0
        files_over_threshold (list): 最大标签值超过 threshold 的文件名列表
    """
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f'文件夹不存在: {label_dir}')

    global_max = 0
    files_over_threshold = []

    for filename in os.listdir(label_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            filepath = os.path.join(label_dir, filename)

            img = tifffile.imread(filepath)
            max_val = int(img.max())

            if max_val > global_max:
                global_max = max_val

            if max_val > threshold:
                files_over_threshold.append(filename)
                print(f'{filename}: max label = {max_val}')

    print(f'All labels max value: {global_max}')
    print(
        f'Number of files with max label > {threshold}: {len(files_over_threshold)}'
    )

    return global_max, files_over_threshold


def relabel_sequential(label_map: np.ndarray) -> np.ndarray:
    """
    Remap non-zero IDs to consecutive IDs starting from 1.
    Background 0 remains unchanged.
    """
    label_map = np.asarray(label_map)
    unique_ids = np.unique(label_map)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return np.zeros_like(label_map)

    # Mapping array approach is fast if max id is not extremely huge
    max_id = int(unique_ids.max())
    mapping = np.zeros(max_id + 1, dtype=label_map.dtype)
    mapping[unique_ids] = np.arange(
        1, len(unique_ids) + 1, dtype=label_map.dtype
    )

    return mapping[label_map]


def expand_mask(mask, xy_radius=3):
    # 创建各向异性结构元素：Z方向不膨胀，XY平面半径为3的圆盘（或方形）
    # 使用方形结构（更简单），半径3表示在XY平面膨胀3层
    struct = np.zeros((1, 2 * xy_radius + 1, 2 * xy_radius + 1), dtype=bool)
    # 填满整个XY圆盘（方形也可，根据需求）
    struct[0, :, :] = True
    # 膨胀一次即可，因为结构元素已经包含了所需半径
    expanded = ndimage.binary_dilation(mask, structure=struct)
    return expanded


def process_one_file(seg_path, raw_path, out_path, diff_threshold=10):
    """
    处理单个分割文件，根据原始图像灰度差值筛选细胞，重新编号并保存
    """
    # 读取分割标签和原始图像
    seg = tifffile.imread(seg_path)
    raw = tifffile.imread(raw_path)

    if seg.shape != raw.shape:
        raise ValueError(
            f'分割文件 {seg_path} 与原始图像 {raw_path} 尺寸不一致'
        )

    # 获取所有细胞ID（排除背景0）
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]

    keep_ids = []  # 记录通过筛选的原始ID
    for cid in cell_ids:
        # 当前细胞的二值mask
        mask_cell = seg == cid
        # 膨胀两圈
        mask_expanded = expand_mask(mask_cell)
        # 背景区域 = 膨胀区域 - 细胞区域
        mask_bg = mask_expanded & (~mask_cell)

        # 如果背景区域没有像素（例如细胞触及边界导致膨胀后无背景），则跳过该细胞（差值无法计算或视为无效）
        if np.sum(mask_bg) == 0:
            continue

        # 计算细胞区域和背景区域的原始图像平均灰度值
        mean_cell = np.mean(raw[mask_cell])
        mean_bg = np.mean(raw[mask_bg])
        diff = mean_cell - mean_bg

        if diff >= diff_threshold:
            keep_ids.append(cid)

    # 如果没有细胞通过筛选，输出全0数组
    if len(keep_ids) == 0:
        new_seg = np.zeros_like(seg)
    else:
        # 重新映射ID：新ID从1开始连续
        new_id_map = {old: new + 1 for new, old in enumerate(keep_ids)}
        new_seg = np.zeros_like(seg)
        for old, new in new_id_map.items():
            new_seg[seg == old] = new

    # 保存结果
    tifffile.imwrite(out_path, new_seg, compression='zlib')
    print(
        f'处理完成: {os.path.basename(seg_path)} -> 保留 {len(keep_ids)} 个细胞'
    )


# --------------------------------------------------------------------
# Training data generation
# --------------------------------------------------------------------
def percentile_normalize(img, p_low=1.0, p_high=99.0, return_stats=False):
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)

    if hi <= lo:
        out = np.zeros_like(img, dtype=np.float32)
        if return_stats:
            return out, lo, hi
        return out

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    img = img.astype(np.float32)

    if return_stats:
        return img, lo, hi
    return img


def random_crop_3d(img, patch_size):
    """
    Randomly crop a 3D patch from img.

    Args:
        img: 3D numpy array with shape (D, H, W)
        patch_size: tuple/list (pd, ph, pw)

    Returns:
        patch: 3D numpy array
    """
    pd, ph, pw = patch_size
    d, h, w = img.shape

    if d < pd or h < ph or w < pw:
        return None

    z0 = random.randint(0, d - pd)
    y0 = random.randint(0, h - ph)
    x0 = random.randint(0, w - pw)

    return img[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]


def resize_3d_by_radius(
    img, cell_radius, target_radius, order=1, scale_z=True
):
    """
    Resize a 3D image according to the ratio target_radius / cell_radius.

    Args:
        img: 3D numpy array with shape (D, H, W)
        cell_radius: original cell radius
        target_radius: target cell radius
        order: interpolation order for scipy.ndimage.zoom
               0 = nearest, 1 = linear, 3 = cubic

    Returns:
        resized_img: 3D numpy array
    """
    scale = float(target_radius) / float(cell_radius)

    if scale_z:
        zoom_factors = (scale, scale, scale)
    else:
        zoom_factors = (1.0, scale, scale)

    resized = zoom(img, zoom=zoom_factors, order=order)
    return resized.astype(np.float32)


def sample_patches_from_one_volume(
    img,
    patch_size,
    intensity_quantile=0.3,
    max_trials=300,
):
    """
    Sample candidate patches from one normalized 3D volume and keep only
    patches whose mean intensity is above a specified quantile.

    Args:
        img: normalized 3D numpy array, shape (D, H, W)
        patch_size: tuple/list (pd, ph, pw)
        intensity_quantile: patch mean threshold quantile in [0, 1]
        max_trials: number of random crop trials for this volume

    Returns:
        kept_patches: list of 3D numpy arrays
    """
    candidate_patches = []
    candidate_means = []

    for _ in range(max_trials):
        patch = random_crop_3d(img, patch_size)
        if patch is None:
            break
        candidate_patches.append(patch)
        candidate_means.append(float(patch.mean()))

    if len(candidate_patches) == 0:
        return []

    threshold = np.quantile(candidate_means, intensity_quantile)

    kept_patches = [
        patch
        for patch, mean_val in zip(candidate_patches, candidate_means)
        if mean_val >= threshold
    ]

    return kept_patches


def build_mae_patches(
    input_dir,
    output_dir,
    cell_radius,
    patch_size,
    target_radius,
    num_patches_needed,
    intensity_quantile=0.3,
    max_trials_per_volume=300,
    valid_suffixes=('.tif', '.tiff'),
    random_seed=42,
    scale_z=True,
):
    """
    Randomly read 3D images from input_dir, resize them according to the
    radius ratio, normalize them using 1%/99% percentiles, sample patches,
    filter low-intensity patches, and save patches to output_dir.

    Processing stops when:
    1) num_patches_needed patches have been saved, or
    2) all files in input_dir have been processed.

    Args:
        input_dir: directory containing input 3D images
        output_dir: directory to save output patches
        cell_radius: original cell radius in the source images
        patch_size: target patch size, e.g. (32, 256, 256)
        target_radius: target cell radius after resizing
        num_patches_needed: total number of patches to save
        intensity_quantile: discard patches whose mean intensity is below
                            this quantile among candidate patches from the
                            same volume
        max_trials_per_volume: number of random crop attempts per volume
        valid_suffixes: file suffixes to read
        random_seed: random seed for reproducibility

    Returns:
        saved_count: number of saved patches
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_suffixes
    ]

    random.shuffle(file_list)

    saved_count = 0

    for img_path in file_list:
        if saved_count >= num_patches_needed:
            break

        try:
            img = tifffile.imread(str(img_path))
        except Exception as e:
            print(f'Failed to read {img_path}: {e}')
            continue

        img = np.asarray(img)

        if img.ndim != 3:
            print(
                f'Skip {img_path}, expected 3D image but got shape {img.shape}'
            )
            continue

        img = img.astype(np.float32)

        # Resize according to cell radius ratio
        resized_img = resize_3d_by_radius(
            img,
            cell_radius=cell_radius,
            target_radius=target_radius,
            order=1,
            scale_z=scale_z,
        )

        # Normalize using 1st and 99th percentiles
        resized_img = percentile_normalize(resized_img, p_low=1.0, p_high=99.0)

        # Sample and filter patches
        kept_patches = sample_patches_from_one_volume(
            resized_img,
            patch_size=patch_size,
            intensity_quantile=intensity_quantile,
            max_trials=max_trials_per_volume,
        )

        if len(kept_patches) == 0:
            print(f'No valid patches kept from {img_path.name}')
            continue

        random.shuffle(kept_patches)

        for patch in kept_patches:
            if saved_count >= num_patches_needed:
                break

            out_name = f'{img_path.stem}_patch_{saved_count:06d}.tif'
            out_path = output_dir / out_name

            # Save patch as float32 TIFF
            tifffile.imwrite(str(out_path), patch.astype(np.float32))
            saved_count += 1

        print(f'Processed {img_path.name}, total saved patches: {saved_count}')

    print(f'Finished. Total saved patches: {saved_count}')
    return saved_count


def resize_3d_by_radius(
    img, cell_radius, target_radius, order=1, scale_z=True
):
    """
    Resize a 3D image according to target_radius / cell_radius.
    """
    scale = float(target_radius) / float(cell_radius)

    if scale_z:
        zoom_factors = (scale, scale, scale)
    else:
        zoom_factors = (1.0, scale, scale)

    resized = zoom(img, zoom=zoom_factors, order=order)
    return resized


def _relabel_one_block(label_block, erode=False):
    """
    Re-label one label block so that instance ids become 1, 2, 3, ...
    Background remains 0.
    Optional binary erosion is applied instance by instance.
    """
    label_block = np.asarray(label_block)
    new_label_block = np.zeros_like(label_block, dtype=np.int16)

    unique_vals = np.unique(label_block)
    new_id = 1

    structure = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=bool,
    )

    for val in unique_vals:
        if val == 0:
            continue

        instance_mask = label_block == val

        if erode:
            instance_mask = binary_erosion(
                instance_mask, structure=structure, iterations=1
            )

        if np.any(instance_mask):
            new_label_block[instance_mask] = new_id
            new_id += 1

    return new_label_block


def _get_starts(orig_len, target_len):
    """
    Evenly distribute sliding-window starts so the whole dimension is covered.
    """
    if orig_len <= target_len:
        return [0]

    n = ceil(orig_len / target_len)
    starts = np.linspace(0, orig_len - target_len, n, dtype=int)
    return starts.tolist()


def build_mask2former_blocks(
    input_dir,
    output_dir,
    label_dir=None,
    label_output_dir=None,
    target_size=(32, 96, 96),
    cell_radius=None,
    target_radius=None,
    scale_z=True,
    p_low=1.0,
    p_high=99.0,
    bg_threshold=0,
    block_filter_percentile=97.0,
    erode=False,
    num_blocks_total=None,
    scale_factors=(0.75, 1.0, 1.25, 1.5),
    blocks_per_volume=400,
    main_scale=1.0,
    main_scale_ratio=0.5,
    shuffle_files=True,
):
    """
    Process all 3D tif volumes and save cropped/resized blocks.

    Multi-scale sampling rule:
      - main_scale, usually 1.0, contributes main_scale_ratio of blocks.
      - all other scales share the remaining ratio equally.
      - Example:
            scale_factors=(0.75, 1.0, 1.25, 1.5)
            blocks_per_volume=200
            main_scale_ratio=0.5
        gives:
            1.0  -> 100 blocks
            0.75 -> 33 blocks
            1.25 -> 33 blocks
            1.5  -> 34 blocks
    """

    os.makedirs(output_dir, exist_ok=True)

    if label_dir is not None:
        if label_output_dir is None:
            label_output_dir = (
                output_dir.replace('images', 'labels')
                if 'images' in output_dir
                else output_dir + '_labels'
            )
        os.makedirs(label_output_dir, exist_ok=True)

    file_list = glob.glob(os.path.join(input_dir, '*.tif')) + glob.glob(
        os.path.join(input_dir, '*.tiff')
    )

    rng = np.random.default_rng(None)
    if shuffle_files:
        rng.shuffle(file_list)
    if len(file_list) == 0:
        print(f'No .tif files found in {input_dir}')
        return 0

    scale_factors = tuple(float(s) for s in scale_factors)

    if main_scale not in scale_factors:
        raise ValueError(
            f'main_scale={main_scale} must be included in scale_factors={scale_factors}'
        )

    if not (0.0 <= main_scale_ratio <= 1.0):
        raise ValueError(
            f'main_scale_ratio must be in [0, 1], got {main_scale_ratio}'
        )

    total_blocks_saved = 0
    tD, tH, tW = target_size

    for file_path in file_list:
        if (
            num_blocks_total is not None
            and total_blocks_saved >= num_blocks_total
        ):
            break

        try:
            vol_raw = tifffile.imread(file_path)
        except Exception as e:
            print(f'Failed to read image {file_path}: {e}')
            continue

        vol_raw = np.asarray(vol_raw)
        if vol_raw.ndim != 3:
            print(
                f'Skipping {file_path}: expected 3D volume, got shape {vol_raw.shape}'
            )
            continue

        label_raw = None
        if label_dir is not None:
            base_name_with_ext = os.path.basename(file_path)
            label_path = os.path.join(label_dir, base_name_with_ext)

            if not os.path.exists(label_path):
                print(
                    f'Warning: label file not found for {base_name_with_ext}, skip this image.'
                )
                continue

            try:
                label_raw = tifffile.imread(label_path)
                label_raw = np.asarray(label_raw)
            except Exception as e:
                print(
                    f'Warning: failed to read label {label_path}: {e}, skip this image.'
                )
                continue

            if label_raw.ndim != 3:
                print(
                    f'Warning: label {label_path} is not 3D, skip this image.'
                )
                continue

        vol_raw = vol_raw.astype(np.float32)

        # Step 1: optional radius normalization.
        if (
            cell_radius is not None
            and target_radius is not None
            and float(cell_radius) != float(target_radius)
        ):
            vol_base = resize_3d_by_radius(
                vol_raw,
                cell_radius=cell_radius,
                target_radius=target_radius,
                order=1,
                scale_z=scale_z,
            ).astype(np.float32)

            if label_raw is not None:
                label_base = resize_3d_by_radius(
                    label_raw,
                    cell_radius=cell_radius,
                    target_radius=target_radius,
                    order=0,
                    scale_z=scale_z,
                ).astype(np.int16)
            else:
                label_base = None
        else:
            vol_base = vol_raw.astype(np.float32)
            label_base = (
                label_raw.astype(np.int16) if label_raw is not None else None
            )

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Compute scale quotas for this volume.
        scale_quota = None
        if blocks_per_volume is not None:
            main_quota = int(round(blocks_per_volume * main_scale_ratio))
            other_total = int(blocks_per_volume) - main_quota

            other_scales = [
                s for s in scale_factors if abs(s - main_scale) > 1e-6
            ]
            scale_quota = dict.fromkeys(scale_factors, 0)
            scale_quota[float(main_scale)] = main_quota

            if len(other_scales) > 0:
                base_other = other_total // len(other_scales)
                remainder = other_total % len(other_scales)

                for i, s in enumerate(other_scales):
                    scale_quota[s] = base_other + (1 if i < remainder else 0)

            print(f'Scale quota for {base_name}: {scale_quota}')

        for scale in scale_factors:
            if (
                num_blocks_total is not None
                and total_blocks_saved >= num_blocks_total
            ):
                break

            scale = float(scale)
            scale_tag = f's{int(round(scale * 100)):03d}'

            if scale_z:
                aug_factors = (scale, scale, scale)
            else:
                aug_factors = (1.0, scale, scale)

            if scale != 1.0:
                vol_scaled = zoom(vol_base, aug_factors, order=1).astype(
                    np.float32
                )

                if label_base is not None:
                    label_scaled = zoom(
                        label_base, aug_factors, order=0
                    ).astype(np.int16)
                else:
                    label_scaled = None
            else:
                vol_scaled = vol_base.copy()
                label_scaled = (
                    label_base.copy() if label_base is not None else None
                )

            # Step 2: percentile clipping + normalization for this scaled volume.
            vol, lo, hi = percentile_normalize(
                vol_scaled,
                p_low=p_low,
                p_high=p_high,
                return_stats=True,
            )

            if hi <= lo:
                bg_threshold_norm = 1.0
            else:
                bg_threshold_norm = (bg_threshold - lo) / (hi - lo)
                bg_threshold_norm = float(np.clip(bg_threshold_norm, 0.0, 1.0))

            label_vol = label_scaled

            D, H, W = vol.shape
            print(
                f'Processing {file_path}, scale={scale:.2f}: '
                f'shape after scaling = {vol.shape}'
            )

            resize_needed = [2 * tD > D, 2 * tH > H, 2 * tW > W]
            quota = scale_quota[scale] if scale_quota is not None else None

            # Case A: all dimensions are resized directly to target_size.
            if all(resize_needed):
                if quota is not None and quota <= 0:
                    print(f'  Skipped scale={scale:.2f}, quota=0')
                    continue

                factors = (tD / D, tH / H, tW / W)
                resized_img = zoom(vol, factors, order=1).astype(np.float32)

                resized_label = None
                if label_vol is not None:
                    resized_label = zoom(label_vol, factors, order=0).astype(
                        np.int16
                    )

                if (
                    np.percentile(resized_img, block_filter_percentile)
                    > bg_threshold_norm
                ):
                    out_path = os.path.join(
                        output_dir, f'{base_name}_{scale_tag}_resized.tif'
                    )
                    tifffile.imwrite(out_path, resized_img.astype(np.float32))

                    if resized_label is not None:
                        resized_label = _relabel_one_block(
                            resized_label, erode=erode
                        )
                        label_out_path = os.path.join(
                            label_output_dir,
                            f'{base_name}_{scale_tag}_resized.tif',
                        )
                        tifffile.imwrite(
                            label_out_path, resized_label.astype(np.int16)
                        )

                    total_blocks_saved += 1
                    print(
                        f'  Saved resized block from {file_path}, scale={scale:.2f}'
                    )
                else:
                    print(
                        f'  Skipped resized block, scale={scale:.2f}, low intensity'
                    )

                continue

            # Case B: resize only short dimensions, then crop long dimensions.
            zoom_factors = [1.0, 1.0, 1.0]
            for i, (orig, targ, need_resize) in enumerate(
                zip([D, H, W], [tD, tH, tW], resize_needed)
            ):
                if need_resize:
                    zoom_factors[i] = targ / orig

            resized_img = zoom(vol, zoom_factors, order=1).astype(np.float32)

            resized_label = None
            if label_vol is not None:
                resized_label = zoom(label_vol, zoom_factors, order=0).astype(
                    np.int16
                )

            new_D, new_H, new_W = resized_img.shape

            starts_D = _get_starts(new_D, tD) if not resize_needed[0] else [0]
            starts_H = _get_starts(new_H, tH) if not resize_needed[1] else [0]
            starts_W = _get_starts(new_W, tW) if not resize_needed[2] else [0]

            valid_starts = []

            for sd in starts_D:
                for sh in starts_H:
                    for sw in starts_W:
                        block = resized_img[
                            sd : sd + tD, sh : sh + tH, sw : sw + tW
                        ]

                        if block.shape != (tD, tH, tW):
                            print(
                                f'  Warning: block shape {block.shape} '
                                f'at ({sd}, {sh}, {sw}) != target'
                            )
                            continue

                        if (
                            np.percentile(block, block_filter_percentile)
                            <= bg_threshold_norm
                        ):
                            continue

                        valid_starts.append((sd, sh, sw))

            if quota is not None and len(valid_starts) > quota:
                # Evenly select valid blocks to avoid keeping only early spatial regions.
                select_idx = np.linspace(0, len(valid_starts) - 1, quota)
                select_idx = np.round(select_idx).astype(int)
                valid_starts = [valid_starts[i] for i in select_idx]

            block_idx = 0

            for sd, sh, sw in valid_starts:
                if (
                    num_blocks_total is not None
                    and total_blocks_saved >= num_blocks_total
                ):
                    break

                block = resized_img[sd : sd + tD, sh : sh + tH, sw : sw + tW]

                out_path = os.path.join(
                    output_dir,
                    f'{base_name}_{scale_tag}_d{sd:04d}_h{sh:04d}_w{sw:04d}.tif',
                )
                tifffile.imwrite(out_path, block.astype(np.float32))

                if resized_label is not None:
                    label_block = resized_label[
                        sd : sd + tD, sh : sh + tH, sw : sw + tW
                    ]
                    label_block = _relabel_one_block(label_block, erode=erode)

                    label_out_path = os.path.join(
                        label_output_dir,
                        f'{base_name}_{scale_tag}_d{sd:04d}_h{sh:04d}_w{sw:04d}.tif',
                    )
                    tifffile.imwrite(
                        label_out_path, label_block.astype(np.int16)
                    )

                total_blocks_saved += 1
                block_idx += 1

            print(
                f'  Saved {block_idx} / {len(valid_starts)} selected blocks '
                f'from {file_path}, scale={scale:.2f}'
            )

    print(f'Processing complete. Total saved blocks: {total_blocks_saved}')
    return total_blocks_saved


# --------------------------------------------------------------------
# MAE reconstruction debug
# --------------------------------------------------------------------
def load_mae_model(checkpoint_path, device, model_kwargs=None):
    """
    Load a trained MaskedAutoencoderViT3D model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Device to load the model on.
        model_kwargs (dict): Arguments for initializing the model (same as training).
                             If None, uses the default large 3D MAE parameters.

    Returns:
        model (nn.Module): Loaded model in eval mode.
    """
    if model_kwargs is None:
        # Use the same parameters as in the training script
        model_kwargs = {
            'img_size': (32, 256, 256),
            'patch_size': 16,
            'in_chans': 1,
            'norm_pix_loss': False,
            'embed_dim': 1008,
            'depth': 24,
            'num_heads': 16,
            'decoder_embed_dim': 768,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
            'mlp_ratio': 4,
        }

    # Instantiate the model
    from mask2former.modeling.backbone.models_mae import MaskedAutoencoderViT3D

    model = MaskedAutoencoderViT3D(**model_kwargs)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle possible DataParallel wrapping (keys starting with 'module.')
    if isinstance(checkpoint, dict):
        # 按优先级尝试常见键名
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 如果没有这些键，假设整个 checkpoint 就是 state_dict
            state_dict = checkpoint
    else:
        # 如果不是字典，直接当作 state_dict
        state_dict = checkpoint

    # 处理 DataParallel 包装（移除 'module.' 前缀）
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f'Model loaded from {checkpoint_path}')
    return model


def get_masked_sample(model, dataset, idx, mask_ratio=0.75, device='cuda'):
    """
    Load a sample, apply masking, and get model reconstruction.
    Returns:
        original: (C, D, H, W) tensor on CPU
        masked_input: (C, D, H, W) tensor (what the encoder sees after masking)
        reconstruction: (C, D, H, W) tensor (model output)
        mask: (N_patches,) binary mask where 1 = removed patches
    """
    model.eval()
    with torch.no_grad():
        # Load a single sample and add batch dimension
        sample, _ = dataset[idx]  # (C, D, H, W)
        sample = sample.unsqueeze(0).to(device)  # (1, C, D, H, W)

        # Forward through encoder to get latent, mask, ids_restore
        latent, mask, ids_restore = model.forward_encoder(sample, mask_ratio)
        # Decoder prediction (patch-level)
        pred = model.forward_decoder(
            latent, ids_restore
        )  # (1, num_patches, p^3 * C)
        # Reconstruct full volume
        rec = model.unpatchify(pred)  # (1, C, D, H, W)

        # Create a masked version of the input for visualization:
        # For patches that are masked (mask=1), we can set them to zero or a constant.
        # Here we set them to zero to visualize what the encoder sees.
        # First, get the patchified version
        patches = model.patchify(sample)  # (1, num_patches, p^3 * C)
        # Mask patches: set removed patches to zero
        patches_masked = patches * (
            1 - mask.unsqueeze(-1).float()
        )  # mask=1 -> set to zero
        # Unpatchify to get the "visible" volume
        visible_vol = model.unpatchify(patches_masked)  # (1, C, D, H, W)

        return (
            sample.squeeze(0).cpu(),
            visible_vol.squeeze(0).cpu(),
            rec.squeeze(0).cpu(),
            mask.squeeze(0).cpu(),
        )


def visualize_3d_reconstruction(
    original, masked, reconstructed, mask, title_prefix=''
):
    """
    original, masked, reconstructed: (C, D, H, W) tensors (C=1)
    mask: (num_patches,) mask (optional, not used directly here)
    Display middle slices along depth, height, width.
    """
    # Remove channel dimension for plotting
    orig = original.squeeze(0).numpy()  # (D, H, W)
    masked_vol = masked.squeeze(0).numpy()  # (D, H, W)
    recon = reconstructed.squeeze(0).numpy()  # (D, H, W)

    D, H, W = orig.shape
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        title_prefix
        + ' | Left: Original | Middle: Masked Input | Right: Reconstruction',
        fontsize=14,
    )

    # Row 0: Depth slice (XY plane at mid depth)
    axes[0, 0].imshow(orig[mid_d, :, :], cmap='gray')
    axes[0, 0].set_title(f'Original (depth {mid_d})')
    axes[0, 1].imshow(masked_vol[mid_d, :, :], cmap='gray')
    axes[0, 1].set_title('Masked Input')
    axes[0, 2].imshow(recon[mid_d, :, :], cmap='gray')
    axes[0, 2].set_title('Reconstruction')

    # Row 1: Coronal slice (XZ plane at mid height)
    axes[1, 0].imshow(orig[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'Original (height {mid_h})')
    axes[1, 1].imshow(masked_vol[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 1].set_title('Masked Input')
    axes[1, 2].imshow(recon[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 2].set_title('Reconstruction')

    # Row 2: Sagittal slice (YZ plane at mid width)
    axes[2, 0].imshow(orig[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 0].set_title(f'Original (width {mid_w})')
    axes[2, 1].imshow(masked_vol[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 1].set_title('Masked Input')
    axes[2, 2].imshow(recon[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 2].set_title('Reconstruction')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


import re

import matplotlib.font_manager as fm
import numpy as np


def visualize_3d_reconstruction_xy(
    original,
    masked,
    reconstructed,
    mask=None,
    title_prefix='',
    patch_size=(8, 12, 12),
    z_idx=None,
    figsize=(14, 5),
    title_fontsize=28,
    mask_alpha=0.65,
    percentile=(1, 99.8),
    save_dir=None,
    save_name=None,
    show=True,
):
    """
    Visualize MAE reconstruction on one XY slice and optionally save as SVG.

    Parameters
    ----------
    original, masked, reconstructed:
        Tensor or ndarray with shape (C, D, H, W) or (D, H, W).

    mask:
        MAE patch mask, usually shape (num_patches,) or (1, num_patches).
        1 means masked patch, 0 means visible patch.

    patch_size:
        Patch size used by MAE, e.g. (8, 12, 12).

    z_idx:
        Which z slice to visualize. If None, use middle slice.

    save_dir:
        Folder to save SVG figures. If None, figure is not saved.

    save_name:
        SVG file name. If None, use title_prefix.

    show:
        Whether to display figure with plt.show().
    """

    # --------------------------------------------------
    # Font fallback
    # --------------------------------------------------
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    if 'Calibri' in available_fonts:
        font_family = 'Calibri'
    elif 'Liberation Sans' in available_fonts:
        font_family = 'Liberation Sans'
    elif 'Arial' in available_fonts:
        font_family = 'Arial'
    else:
        font_family = 'DejaVu Sans'

    title_fontdict = {
        'fontsize': title_fontsize,
        'fontfamily': font_family,
        'fontweight': 'normal',
    }

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        x = np.asarray(x)

        if x.ndim == 4:
            x = x.squeeze(0)

        return x.astype(np.float32)

    orig = to_numpy(original)  # (D, H, W)
    masked_vol = to_numpy(masked)  # (D, H, W)
    recon = to_numpy(reconstructed)  # (D, H, W)

    D, H, W = orig.shape

    if z_idx is None:
        z_idx = D // 2

    raw_xy = orig[z_idx]
    masked_xy = masked_vol[z_idx]
    recon_xy = recon[z_idx]

    # Shared intensity range
    vmin, vmax = np.percentile(orig, percentile)

    if vmax <= vmin:
        vmin, vmax = float(orig.min()), float(orig.max())

    # --------------------------------------------------
    # Build pixel-level mask overlay
    # --------------------------------------------------
    mask_xy = None

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)

        mask_np = mask_np.reshape(-1)

        pd, ph, pw = patch_size
        gd, gh, gw = D // pd, H // ph, W // pw
        expected_len = gd * gh * gw

        if mask_np.size == expected_len:
            patch_mask = mask_np.reshape(gd, gh, gw)

            voxel_mask = np.repeat(
                np.repeat(
                    np.repeat(patch_mask, pd, axis=0),
                    ph,
                    axis=1,
                ),
                pw,
                axis=2,
            )

            voxel_mask = voxel_mask[:D, :H, :W]
            mask_xy = voxel_mask[z_idx] > 0.5

    # Fallback: infer masked area from zeroed input
    if mask_xy is None:
        eps = 1e-6
        mask_xy = (np.abs(masked_xy) < eps) & (np.abs(raw_xy) > eps)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Raw
    axes[0].imshow(
        raw_xy,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(
        'Raw',
        pad=14,
        **title_fontdict,
    )

    # 2. Masked Input
    axes[1].imshow(
        masked_xy,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
    )

    gray_overlay = np.zeros((*mask_xy.shape, 4), dtype=np.float32)
    gray_overlay[..., 0] = 0.65
    gray_overlay[..., 1] = 0.65
    gray_overlay[..., 2] = 0.65
    gray_overlay[..., 3] = mask_xy.astype(np.float32) * mask_alpha

    axes[1].imshow(gray_overlay)

    axes[1].set_title(
        'Masked Input',
        pad=14,
        **title_fontdict,
    )

    # 3. Reconstruction
    axes[2].imshow(
        recon_xy,
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title(
        'Reconstruction',
        pad=14,
        **title_fontdict,
    )

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    # --------------------------------------------------
    # Save as SVG
    # --------------------------------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            if title_prefix:
                save_name = title_prefix
            else:
                save_name = f'mae_reconstruction_z{z_idx}'

        save_name = re.sub(r'[^\w\-\.]+', '_', save_name)

        if not save_name.lower().endswith('.svg'):
            save_name += '.svg'

        save_path = os.path.join(save_dir, save_name)

        fig.savefig(
            save_path,
            format='svg',
            bbox_inches='tight',
            pad_inches=0.05,
        )

        print(f'Saved SVG to: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# Inference batch
# ------------------------------------------------------------
def preprocess_image(image_path, gaussian_sigma=None):
    """the same as training"""
    image = tifffile.imread(image_path)
    # image = np.array(image, dtype=np.uint8, copy=True)  # <-- this converts to standard np.ndarray
    # image = image.astype(np.float32) / 255.0
    image = np.array(image, dtype=np.float32, copy=True)
    if gaussian_sigma is not None:
        if isinstance(gaussian_sigma, (int, float)):
            if gaussian_sigma > 0:
                image = gaussian_filter(image, sigma=gaussian_sigma)
        else:
            image = gaussian_filter(image, sigma=gaussian_sigma)

    image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
    return image_tensor


def visualize_sample(
    image_path,
    model,
    device,
    output_dir=None,
    score_thresh=0.8,
    mask_thresh=0.5,
    overlap_ratio_thresh=0.1,
    gaussian_sigma=None,
):
    """
    Run inference on one 3D image, postprocess it with patch_postprocess,
    visualize the middle slice of the raw image and the colored mask, and optionally
    save the 3D label map.
    """

    def label_slice_to_rgb(label_slice, cmap_name='tab20'):
        """
        Convert a 2D label slice to a colored RGB image.
        Background stays black.
        """
        h, w = label_slice.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        max_label = int(label_slice.max())
        if max_label == 0:
            return rgb

        cmap = plt.get_cmap(cmap_name, max(max_label, 20))
        for lab in range(1, max_label + 1):
            color = cmap((lab - 1) % cmap.N)[:3]
            rgb[label_slice == lab] = color

        return rgb

    # Preprocess image
    image_tensor = preprocess_image(
        image_path, gaussian_sigma=gaussian_sigma
    ).to(device)  # (C, D, H, W)
    inputs = [{'image': image_tensor}]

    with torch.no_grad():
        outputs = model(inputs)[0]

    image_np = image_tensor[0].cpu().numpy()  # (D, H, W)
    D, H, W = image_np.shape

    raw_scores = outputs.get('pred_scores', None)
    raw_masks = outputs.get('pred_masks', None)

    print('\n=== Instances summary before filtering ===')
    if raw_scores is None or raw_masks is None:
        print('Model output does not contain pred_scores or pred_masks.')
    else:
        print(f'num_instances = {len(raw_scores)}')

    print('\n=== Raw output stats ===')
    if raw_scores is not None:
        print(
            f'scores: shape={tuple(raw_scores.shape)}, '
            f'min={raw_scores.min().item():.6f}, '
            f'max={raw_scores.max().item():.6f}, '
            f'mean={raw_scores.mean().item():.6f}'
        )

    if raw_masks is not None:
        print(
            f'masks: shape={tuple(raw_masks.shape)}, '
            f'min={raw_masks.min().item():.6f}, '
            f'max={raw_masks.max().item():.6f}, '
            f'mean={raw_masks.mean().item():.6f}'
        )

    # post = patch_postprocess(
    #     model_output=outputs,
    #     patch_shape=(D, H, W),
    #     score_thresh=score_thresh,
    #     mask_thresh=mask_thresh,
    #     topk_postprocess=300,
    # )
    post = patch_postprocess_argmax(
        model_output=outputs,
        patch_shape=(D, H, W),
        score_thresh=score_thresh,
        mask_thresh=mask_thresh,
        topk_postprocess=300,  # 可先试 50 / 80 / 100
    )

    label_map = post['patch_instance_map']
    kept_scores = post['kept_scores']
    kept_labels = post['kept_labels']
    kept_original_indices = post['kept_original_indices']

    print('\n=== After patch_postprocess ===')
    print(f'kept_instances = {len(kept_scores)}')

    # for new_label_id, (score, cls_id, orig_idx) in enumerate(
    #     zip(kept_scores, kept_labels, kept_original_indices), start=1
    # ):
    #     voxels = int((label_map == new_label_id).sum())
    #     print(
    #         f"kept instance {new_label_id:03d} | "
    #         f"orig_idx={orig_idx:03d} | "
    #         f"score={score:.4f} | class={cls_id} | voxels={voxels}"
    #     )

    # Visualize the middle slice
    slice_idx = D // 2
    img_slice = image_np[slice_idx]
    mask_slice = label_map[slice_idx]
    color_mask_slice = label_slice_to_rgb(mask_slice)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f'Image slice (depth {slice_idx})')
    axes[0].axis('off')

    axes[1].imshow(color_mask_slice)
    axes[1].set_title(f'Predicted masks (score > {score_thresh})')
    axes[1].axis('off')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        tif_path = os.path.join(
            output_dir,
            os.path.basename(image_path).replace('.tif', '_pred.tif'),
        )
        tifffile.imwrite(tif_path, label_map, compression=None)
        print(f'Saved 3D prediction to {tif_path}')

    plt.show()

    return {
        'label_map': label_map,
        'kept_scores': kept_scores,
        'kept_labels': kept_labels,
        'kept_original_indices': kept_original_indices,
    }


def preprocess_volumes(
    input_dir,
    output_dir,
    label_dir=None,
    label_output_dir=None,
    target_size=(32, 256, 256),
    scaleTerm=255,
    clipping=0,
    bgIntensity=150,
    erode=False,
    num_blocks_total=None,
):
    """
    Process all .tif volumes in input_dir and save them as cropped/resized blocks
    of size target_size to output_dir.
    If label_dir is provided, performs synchronized processing on label volumes.

    For each volume:
    - If a dimension < 2 * target_size, that dimension is resized to the target.
    - If a dimension >= 2 * target_size, the volume is cropped along that dimension
      using sliding windows with overlapping (if needed) so that the entire dimension
      is exactly covered. The number of windows is chosen as ceil(orig_size / target),
      and window start positions are evenly spaced (including the last one at the end).
    - If label_dir is provided, labels are processed with nearest-neighbor interpolation
      to preserve instance IDs. Blocks are saved only if the corresponding image block
      passes the intensity filter.

    Args:
        input_dir (str): Path to folder containing input .tif files.
        output_dir (str): Path to folder where processed image blocks will be saved.
        label_dir (str, optional): Path to folder containing label .tif files. Defaults to None.
        target_size (tuple): Desired output size (depth, height, width).
        scaleTerm (int): Upper bound for intensity clipping.
        clipping (int): Lower bound for intensity clipping.
        bgIntensity (int): Threshold for filtering background blocks.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if label_dir:
        label_output_dir = (
            output_dir.replace('images', 'labels')
            if 'images' in output_dir
            else output_dir + '_labels'
        )
        if label_output_dir is None:
            raise ValueError(
                'label_output_dir must be specified if label_dir is provided'
            )
        os.makedirs(label_output_dir, exist_ok=True)

    # Get list of all .tif files (case-insensitive)
    file_list = glob.glob(os.path.join(input_dir, '*.tif')) + glob.glob(
        os.path.join(input_dir, '*.tiff')
    )
    if not file_list:
        print(f'No .tif files found in {input_dir}')
        return

    if num_blocks_total is not None:
        import random

        random.shuffle(file_list)
        total_blocks_saved = 0

    tD, tH, tW = target_size
    # Pre-calculate the scaled background intensity threshold
    bgIntensity_scaled = (
        (bgIntensity - clipping) / (scaleTerm - clipping) * 255
    )

    for file_path in file_list:
        # Read the volume (assumes 3D shape: D, H, W)
        vol = tifffile.imread(file_path)

        # Check for label file
        label_path = None
        label_vol = None
        if label_dir:
            base_name = os.path.basename(file_path)
            potential_label = os.path.join(label_dir, base_name)
            if os.path.exists(potential_label):
                label_path = potential_label
                label_vol = tifffile.imread(label_path)
                # Ensure label is integer type
                if label_vol.dtype != np.int16:
                    label_vol = label_vol.astype(np.int16)
            else:
                print(
                    f'Warning: Label file not found for {base_name}, skipping label processing.'
                )

        # Preprocess image volume
        vol = np.clip(vol, clipping, scaleTerm)
        vol = (vol - clipping) / (scaleTerm - clipping) * 255

        orig_shape = vol.shape
        if len(orig_shape) != 3:
            print(
                f'Skipping {file_path}: expected 3D volume, got shape {orig_shape}'
            )
            continue

        D, H, W = orig_shape
        print(f'Processing {file_path}: shape {orig_shape}')

        # Determine for each dimension whether to resize (True) or crop (False)
        resize_needed = [2 * tD > D, 2 * tH > H, 2 * tW > W]

        # If all dimensions are to be resized, just resize the whole volume and save
        if all(resize_needed):
            # Compute zoom factors
            factors = (tD / D, tH / H, tW / W)
            resized = zoom(vol, factors, order=1)  # linear interpolation

            # Resize label if exists (use order=0 for nearest neighbor)
            resized_label = None
            if label_vol is not None:
                resized_label = zoom(label_vol, factors, order=0)

            # Save as a single block
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            if (
                num_blocks_total is not None
                and total_blocks_saved >= num_blocks_total
            ):
                break
            # Check filter condition before saving
            if np.percentile(resized, 97) > bgIntensity_scaled:
                out_path = os.path.join(output_dir, f'{base_name}_resized.tif')
                tifffile.imwrite(out_path, resized.astype(np.int8))

                if resized_label is not None:
                    label_out_path = os.path.join(
                        label_output_dir, f'{base_name}_resized.tif'
                    )
                    tifffile.imwrite(
                        label_out_path, resized_label.astype(np.int16)
                    )
                    print(f'  Saved resized volume and label to {out_path}')
                else:
                    print(f'  Saved resized volume to {out_path}')
            else:
                print('  Skipped resized volume (low intensity)')
            continue

        # Otherwise, first resize only the dimensions that are < 2*target.
        zoom_factors = [1.0, 1.0, 1.0]
        for i, (orig, targ, need_resize) in enumerate(
            zip([D, H, W], [tD, tH, tW], resize_needed)
        ):
            if need_resize:
                zoom_factors[i] = targ / orig

        # Apply zoom
        resized_vol = zoom(vol, zoom_factors, order=1)
        # Apply zoom to label (order=0 to preserve integer labels)
        resized_label_vol = None
        if label_vol is not None:
            resized_label_vol = zoom(label_vol, zoom_factors, order=0)

        new_D, new_H, new_W = resized_vol.shape

        # For dimensions that were cropped, generate start positions
        def get_starts(orig_len, target_len):
            if orig_len <= target_len:
                return [0]
            n = ceil(orig_len / target_len)
            starts = np.linspace(0, orig_len - target_len, n, dtype=int)
            return starts.tolist()

        starts_D = get_starts(new_D, tD) if not resize_needed[0] else [0]
        starts_H = get_starts(new_H, tH) if not resize_needed[1] else [0]
        starts_W = get_starts(new_W, tW) if not resize_needed[2] else [0]

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        block_idx = 0
        for sd in starts_D:
            for sh in starts_H:
                for sw in starts_W:
                    if (
                        num_blocks_total is not None
                        and total_blocks_saved >= num_blocks_total
                    ):
                        break
                    block = resized_vol[
                        sd : sd + tD, sh : sh + tH, sw : sw + tW
                    ]

                    # Filter check
                    if np.percentile(block, 97) <= bgIntensity_scaled:
                        continue

                    # Sanity check block shape
                    if block.shape != (tD, tH, tW):
                        print(
                            f'  Warning: block shape {block.shape} at ({sd},{sh},{sw}) != target'
                        )
                        continue

                    # Save image block
                    out_path = os.path.join(
                        output_dir,
                        f'{base_name}_d{sd:04d}_h{sh:04d}_w{sw:04d}.tif',
                    )
                    tifffile.imwrite(out_path, block.astype(np.int8))
                    total_blocks_saved += 1
                    # Save label block if exists
                    if resized_label_vol is not None:
                        label_block = resized_label_vol[
                            sd : sd + tD, sh : sh + tH, sw : sw + tW
                        ]
                        unique_vals = np.unique(label_block)
                        new_label_block = np.zeros_like(
                            label_block, dtype=np.int16
                        )
                        new_id = 1
                        eroded_label_block = np.zeros_like(
                            label_block, dtype=np.int16
                        )
                        structure = np.array(
                            [
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            ],
                            dtype=bool,
                        )
                        for val in unique_vals:
                            if val == 0:  # Skip background
                                continue
                            if erode == True:
                                instance_mask = label_block == val
                                eroded_mask = binary_erosion(
                                    instance_mask,
                                    structure=structure,
                                    iterations=1,
                                )
                                new_label_block[eroded_mask] = new_id
                            else:
                                new_label_block[label_block == val] = new_id
                            new_id += 1
                        label_out_path = os.path.join(
                            label_output_dir,
                            f'{base_name}_d{sd:04d}_h{sh:04d}_w{sw:04d}.tif',
                        )

                        tifffile.imwrite(
                            label_out_path, new_label_block.astype(np.int16)
                        )

                    block_idx += 1
                    if (
                        num_blocks_total is not None
                        and total_blocks_saved >= num_blocks_total
                    ):
                        break
                if (
                    num_blocks_total is not None
                    and total_blocks_saved >= num_blocks_total
                ):
                    break
            if (
                num_blocks_total is not None
                and total_blocks_saved >= num_blocks_total
            ):
                break
        print(f'  Saved {block_idx} blocks from {file_path}')
        if (
            num_blocks_total is not None
            and total_blocks_saved >= num_blocks_total
        ):
            break
    print(f'Processing complete. Output saved to {output_dir}')


def evaluate_dataset_instance_metrics(
    image_paths,
    gt_dir,
    model,
    device,
    output_dir=None,
    score_thresh=0.8,
    mask_thresh=0.5,
    overlap_ratio_thresh=0.5,
    gaussian_sigma=None,
    iou_thresh=0.5,
    save_pred=False,
):
    """
    Run prediction on a list of images and evaluate instance-level
    precision / recall / F1 against ground truth label maps.

    Args:
        image_paths: list of image paths
        gt_dir: folder containing GT tif files with the same basename
        model: predictor model
        device: torch device
        output_dir: optional folder to save prediction tif files
        score_thresh, mask_thresh, overlap_ratio_thresh, gaussian_sigma:
            forwarded to visualize_sample
        iou_thresh: IoU threshold for instance matching
        save_pred: whether to save predicted label maps

    Returns:
        dict with dataset mean metrics and per-sample details (including fp_ids/fn_ids)
    """
    per_sample_results = []

    for image_path in image_paths:
        print(f'\nProcessing {image_path} ...')

        pred_result = visualize_sample(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=output_dir if save_pred else None,
            score_thresh=score_thresh,
            mask_thresh=mask_thresh,
            overlap_ratio_thresh=overlap_ratio_thresh,
            gaussian_sigma=gaussian_sigma,
        )

        pred_label_map = pred_result['label_map']

        gt_path = os.path.join(gt_dir, os.path.basename(image_path))
        if not os.path.exists(gt_path):
            print(f'[Warning] GT not found, skip: {gt_path}')
            continue

        gt_label_map = tifffile.imread(gt_path)

        if pred_label_map.shape != gt_label_map.shape:
            raise ValueError(
                f'Shape mismatch for {os.path.basename(image_path)}: '
                f'pred {pred_label_map.shape} vs gt {gt_label_map.shape}'
            )

        metrics = compute_instance_metrics(
            pred_label_map=pred_label_map,
            gt_label_map=gt_label_map,
            iou_thresh=iou_thresh,
        )

        # Retrieve fp_ids and fn_ids from the metrics dict
        fp_ids = metrics.get('fp_ids', [])
        fn_ids = metrics.get('fn_ids', [])

        sample_result = {
            'image_path': image_path,
            'gt_path': gt_path,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'num_pred': metrics['num_pred'],
            'num_gt': metrics['num_gt'],
            'fp_ids': fp_ids,  # list of false positive instance IDs in prediction
            'fn_ids': fn_ids,  # list of false negative instance IDs in ground truth
        }
        per_sample_results.append(sample_result)

        # Print sample summary including fp_ids and fn_ids (limit displayed length)
        print(
            f'[{os.path.basename(image_path)}] '
            f'TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]} | '
            f'P={metrics["precision"]:.4f} '
            f'R={metrics["recall"]:.4f} '
            f'F1={metrics["f1"]:.4f}'
        )
        # Print false positive IDs (if any)
        if fp_ids:
            fp_str = ', '.join(str(x) for x in fp_ids[:10])  # show at most 10
            if len(fp_ids) > 10:
                fp_str += f' ... ({len(fp_ids)} total)'
            print(f'  FP instance IDs (pred): {fp_str}')
        # Print false negative IDs (if any)
        if fn_ids:
            fn_str = ', '.join(str(x) for x in fn_ids[:10])
            if len(fn_ids) > 10:
                fn_str += f' ... ({len(fn_ids)} total)'
            print(f'  FN instance IDs (gt):   {fn_str}')

    if len(per_sample_results) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'sum_tp': 0,
            'sum_fp': 0,
            'sum_fn': 0,
            'num_samples': 0,
            'per_sample_results': [],
        }

    sum_tp = int(np.sum([x['tp'] for x in per_sample_results]))
    sum_fp = int(np.sum([x['fp'] for x in per_sample_results]))
    sum_fn = int(np.sum([x['fn'] for x in per_sample_results]))

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print('\n=== Dataset summary ===')
    print(f'num_samples = {len(per_sample_results)}')
    print(f'precision   = {precision:.4f}')
    print(f'recall      = {recall:.4f}')
    print(f'f1          = {f1:.4f}')
    print(f'sum_tp      = {sum_tp}')
    print(f'sum_fp      = {sum_fp}')
    print(f'sum_fn      = {sum_fn}')

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'sum_tp': sum_tp,
        'sum_fp': sum_fp,
        'sum_fn': sum_fn,
        'num_samples': len(per_sample_results),
        'per_sample_results': per_sample_results,
    }


def compute_instance_metrics(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Fast instance-level precision / recall / F1 using sparse overlap counting
    and global greedy matching on IoU.
    """
    # pred_label_map = relabel_sequential(pred_label_map)
    # gt_label_map = relabel_sequential(gt_label_map)

    pred_ids, pred_sizes = np.unique(pred_label_map, return_counts=True)
    gt_ids, gt_sizes = np.unique(gt_label_map, return_counts=True)

    # remove background
    pred_fg = pred_ids > 0
    gt_fg = gt_ids > 0
    pred_ids = pred_ids[pred_fg]
    pred_sizes = pred_sizes[pred_fg]
    gt_ids = gt_ids[gt_fg]
    gt_sizes = gt_sizes[gt_fg]

    num_pred = len(pred_ids)
    num_gt = len(gt_ids)

    # 情况1：无任何前景
    if num_pred == 0 and num_gt == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'num_pred': 0,
            'num_gt': 0,
            'matched_pairs': [],
            'fp_ids': [],  # 新增
            'fn_ids': [],  # 新增
        }

    # 情况2：只有真值前景，没有预测前景
    if num_pred == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': int(num_gt),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_pred': 0,
            'num_gt': int(num_gt),
            'matched_pairs': [],
            'fp_ids': [],  # 新增
            'fn_ids': [int(gid) for gid in gt_ids],  # 新增：所有真值都是假阴性
        }

    # 情况3：只有预测前景，没有真值前景
    if num_gt == 0:
        return {
            'tp': 0,
            'fp': int(num_pred),
            'fn': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_pred': int(num_pred),
            'num_gt': 0,
            'matched_pairs': [],
            'fp_ids': [
                int(pid) for pid in pred_ids
            ],  # 新增：所有预测都是假阳性
            'fn_ids': [],  # 新增
        }

    # size lookup by sequential id
    pred_size_arr = np.bincount(pred_label_map.ravel())
    gt_size_arr = np.bincount(gt_label_map.ravel())

    # Count intersections only on overlapping foreground voxels
    fg_mask = (pred_label_map > 0) & (gt_label_map > 0)
    pred_fg_flat = pred_label_map[fg_mask].astype(np.int64, copy=False)
    gt_fg_flat = gt_label_map[fg_mask].astype(np.int64, copy=False)

    # 情况4：没有任何重叠的前景体素
    if pred_fg_flat.size == 0:
        tp = 0
        fp = int(num_pred)
        fn = int(num_gt)
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_pred': int(num_pred),
            'num_gt': int(num_gt),
            'matched_pairs': [],
            'fp_ids': [int(pid) for pid in pred_ids],  # 新增：所有预测未匹配
            'fn_ids': [int(gid) for gid in gt_ids],  # 新增：所有真值未匹配
        }

    # Encode pair (pred_id, gt_id) -> unique integer
    factor = int(gt_label_map.max()) + 1
    pair_code = pred_fg_flat * factor + gt_fg_flat

    pair_ids, inter_counts = np.unique(pair_code, return_counts=True)

    pred_overlap_ids = pair_ids // factor
    gt_overlap_ids = pair_ids % factor

    # Build candidate list with IoU
    candidates = []
    for pid, gid, inter in zip(pred_overlap_ids, gt_overlap_ids, inter_counts):
        union = int(pred_size_arr[pid]) + int(gt_size_arr[gid]) - int(inter)
        iou = float(inter) / float(union) if union > 0 else 0.0
        if iou >= iou_thresh:
            candidates.append((iou, int(pid), int(gid)))

    # Global greedy matching by descending IoU
    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    matched_pairs = []

    for iou, pid, gid in candidates:
        if pid in matched_pred or gid in matched_gt:
            continue
        matched_pred.add(pid)
        matched_gt.add(gid)
        matched_pairs.append((pid, gid, iou))

    tp = len(matched_pairs)
    fp = num_pred - tp
    fn = num_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # 计算未匹配的预测ID和真值ID
    fp_ids = [int(pid) for pid in pred_ids if pid not in matched_pred]
    fn_ids = [int(gid) for gid in gt_ids if gid not in matched_gt]

    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'num_pred': int(num_pred),
        'num_gt': int(num_gt),
        'matched_pairs': matched_pairs,
        'fp_ids': fp_ids,  # 新增
        'fn_ids': fn_ids,  # 新增
    }


# ------------------------------------------------------------
# DEBUG mask2former
# ------------------------------------------------------------
@torch.no_grad()
def compute_point_feature_attention_maps(
    image_tensor,
    model,
    point_zyx=None,
    random_point=False,
    normalize=True,
):
    """
    Args:
        image_tensor: (C, D, H, W), 已经 preprocess 后并放到 device
        model: MaskFormer model
        point_zyx: 原始图像坐标 (z, y, x)
        random_point: True 时随机取点
        normalize: 是否对 feature 做 L2 normalize

    Returns:
        attn_maps_up: dict, 每个尺度上采样后的 attention map, shape=(D,H,W)
        raw_attn_maps: dict, 每个尺度原始 attention map
        point_zyx: 实际使用的原图坐标
    """

    model.eval()

    C, D, H, W = image_tensor.shape

    if random_point or point_zyx is None:
        z = np.random.randint(0, D)
        y = np.random.randint(0, H)
        x = np.random.randint(0, W)
        point_zyx = (z, y, x)
    else:
        z, y, x = point_zyx
        z = int(np.clip(z, 0, D - 1))
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        point_zyx = (z, y, x)

    # backbone 只吃 batched tensor
    x_in = image_tensor.unsqueeze(0)  # (1,C,D,H,W)

    backbone_output = model.backbone(x_in)
    feats = _extract_feature_dict(backbone_output)
    feats = _sort_feats_by_resolution(feats)

    attn_maps_up = {}
    raw_attn_maps = {}

    print('\n=== Backbone multi-scale features ===')
    print(f'Original image shape: D,H,W = {(D, H, W)}')
    print(f'Selected point z,y,x = {point_zyx}')

    for name, feat in feats.items():
        # feat: (1,C,d,h,w)
        feat = feat[0]  # (C,d,h,w)
        c, d, h, w = feat.shape

        # 原图坐标映射到当前 feature 坐标
        zz = int(round(z / max(D - 1, 1) * max(d - 1, 1)))
        yy = int(round(y / max(H - 1, 1) * max(h - 1, 1)))
        xx = int(round(x / max(W - 1, 1) * max(w - 1, 1)))

        f = feat.float()

        if normalize:
            f = F.normalize(f, dim=0)

        q = f[:, zz, yy, xx]  # (C,)
        attn = torch.einsum('c,cdhw->dhw', q, f)  # (d,h,w)

        # 归一化到 0-1，方便显示
        attn_min = attn.min()
        attn_max = attn.max()
        attn_vis = (attn - attn_min) / (attn_max - attn_min + 1e-6)

        attn_up = F.interpolate(
            attn_vis[None, None],
            size=(D, H, W),
            mode='trilinear',
            align_corners=False,
        )[0, 0]

        raw_attn_maps[name] = attn_vis.detach().cpu().numpy()
        attn_maps_up[name] = attn_up.detach().cpu().numpy()

        scale = (D / d, H / h, W / w)
        print(
            f'{name}: feat shape={(c, d, h, w)}, '
            f'approx scale zyx={scale}, '
            f'mapped point={(zz, yy, xx)}'
        )

    return attn_maps_up, raw_attn_maps, point_zyx


def _extract_feature_dict(backbone_output):
    """
    尽量兼容 detectron2 / Mask2Former backbone 输出:
      - dict: {"res2": tensor, "res3": tensor, ...}
      - list/tuple: [tensor1, tensor2, ...]
    tensor 期望 shape: (B, C, D, H, W)
    """
    feats = {}

    if isinstance(backbone_output, dict):
        for k, v in backbone_output.items():
            if torch.is_tensor(v) and v.ndim == 5:
                feats[k] = v
    elif isinstance(backbone_output, (list, tuple)):
        for i, v in enumerate(backbone_output):
            if torch.is_tensor(v) and v.ndim == 5:
                feats[f'feat_{i}'] = v
    else:
        raise TypeError(
            f'Unsupported backbone output type: {type(backbone_output)}'
        )

    if len(feats) == 0:
        raise RuntimeError('No 5D feature map found in backbone output.')

    return feats


def _sort_feats_by_resolution(feats):
    """
    按空间分辨率从大到小排序。
    """
    items = list(feats.items())
    items = sorted(
        items, key=lambda kv: np.prod(kv[1].shape[-3:]), reverse=True
    )
    return dict(items)


def visualize_feature_attention_for_sample(
    image_path,
    model,
    device,
    output_dir,
    point_zyx=None,
    random_point=False,
    gaussian_sigma=None,
    alpha=0.55,
    title_fontsize=28,
):
    """
    对单个 3D patch:
      1. preprocess
      2. backbone 多尺度 feature
      3. 指定点 feature affinity
      4. 每个尺度上采样回原图尺寸
      5. 保存 tif + 指定 z slice overlay png
    """

    os.makedirs(output_dir, exist_ok=True)

    image_tensor = preprocess_image(
        image_path,
        gaussian_sigma=gaussian_sigma,
    ).to(device)  # (C, D, H, W)

    image_np = image_tensor[0].detach().cpu().numpy()
    D, H, W = image_np.shape

    attn_maps_up, raw_attn_maps, point_zyx = (
        compute_point_feature_attention_maps(
            image_tensor=image_tensor,
            model=model,
            point_zyx=point_zyx,
            random_point=random_point,
            normalize=True,
        )
    )

    z0, y0, x0 = point_zyx

    base = os.path.splitext(os.path.basename(image_path))[0]

    # ------------------------------------------------------------
    # 保存 3D attention map
    # ------------------------------------------------------------
    for name, attn in attn_maps_up.items():
        save_path = os.path.join(
            output_dir,
            f'{base}_{name}_point_{z0}_{y0}_{x0}_attn_up.tif',
        )
        tifffile.imwrite(save_path, attn.astype(np.float32), compression=None)
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------
    # 画指定 z slice
    # ------------------------------------------------------------
    n_features = len(attn_maps_up)
    n_cols = n_features + 1

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(5.5 * n_cols, 5.8),
    )

    if n_cols == 1:
        axes = [axes]

    # 第一列：Raw
    axes[0].imshow(image_np[z0], cmap='gray')
    axes[0].scatter([x0], [y0], c='red', s=60)
    axes[0].set_title(
        'Raw',
        fontsize=title_fontsize,
        fontweight='normal',
        pad=16,
    )
    axes[0].axis('off')

    # 后面列：Feature Res2 ~ Feature Res5
    feature_titles = [f'Backbone Feature{i}' for i in range(2, 2 + n_features)]

    for ax, title, (name, attn) in zip(
        axes[1:],
        feature_titles,
        attn_maps_up.items(),
    ):
        ax.imshow(image_np[z0], cmap='gray')
        ax.imshow(attn[z0], cmap='jet', alpha=alpha, vmin=0, vmax=1)
        ax.scatter([x0], [y0], c='white', s=60)
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight='normal',
            pad=16,
        )
        ax.axis('off')

    plt.tight_layout()

    svg_path = os.path.join(
        output_dir,
        f'{base}_point_{z0}_{y0}_{x0}_feature_attention.svg',
    )
    plt.savefig(svg_path, dpi=200, bbox_inches='tight')
    print(f'Saved overlay png: {svg_path}')

    plt.show()

    return {
        'attn_maps_up': attn_maps_up,
        'raw_attn_maps': raw_attn_maps,
        'point_zyx': point_zyx,
    }


def _extract_pixel_decoder_feature_dict(pixel_decoder_output):
    """
    Convert pixel decoder output into a dict of 5D feature maps.

    Expected standard Mask2Former-style return:
        mask_features, transformer_encoder_features, multi_scale_features

    mask_features:
        (B, C, D, H, W)

    multi_scale_features:
        list of tensors, each (B, C, d, h, w)
    """
    feats = {}

    if not isinstance(pixel_decoder_output, (list, tuple)):
        raise TypeError(
            f'Unsupported pixel decoder output type: {type(pixel_decoder_output)}'
        )

    if len(pixel_decoder_output) < 3:
        raise RuntimeError(
            f'Expected pixel decoder output length >= 3, got {len(pixel_decoder_output)}'
        )

    mask_features = pixel_decoder_output[0]
    transformer_encoder_features = pixel_decoder_output[1]
    multi_scale_features = pixel_decoder_output[2]

    if torch.is_tensor(mask_features) and mask_features.ndim == 5:
        feats['mask_features'] = mask_features

    if (
        torch.is_tensor(transformer_encoder_features)
        and transformer_encoder_features.ndim == 5
    ):
        feats['transformer_encoder_features'] = transformer_encoder_features

    if isinstance(multi_scale_features, (list, tuple)):
        for i, v in enumerate(multi_scale_features):
            if torch.is_tensor(v) and v.ndim == 5:
                feats[f'multi_scale_{i}'] = v

    if len(feats) == 0:
        raise RuntimeError('No 5D feature map found in pixel decoder output.')

    return feats


@torch.no_grad()
def compute_point_pixel_decoder_attention_maps(
    image_tensor,
    model,
    point_zyx=None,
    random_point=False,
    normalize=True,
):
    """
    Compute point feature-affinity maps from pixel decoder outputs.

    Args:
        image_tensor: (C, D, H, W), already preprocessed and moved to device
        model: MaskFormer model
        point_zyx: original image coordinate (z, y, x)
        random_point: if True, randomly select one point
        normalize: whether to L2-normalize features along channel dimension

    Returns:
        attn_maps_up: dict, attention maps upsampled to original image size, each (D,H,W)
        raw_attn_maps: dict, raw-resolution attention maps
        point_zyx: selected original image coordinate
        pixel_feats: dict of raw pixel decoder features
    """

    model.eval()

    C, D, H, W = image_tensor.shape

    if random_point or point_zyx is None:
        z = np.random.randint(0, D)
        y = np.random.randint(0, H)
        x = np.random.randint(0, W)
        point_zyx = (z, y, x)
    else:
        z, y, x = point_zyx
        z = int(np.clip(z, 0, D - 1))
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        point_zyx = (z, y, x)

    x_in = image_tensor.unsqueeze(0)  # (1, C, D, H, W)

    # 1. Backbone features
    backbone_output = model.backbone(x_in)

    # 2. Pixel decoder features
    pixel_decoder_output = model.pixel_decoder.forward_features(
        backbone_output
    )
    pixel_feats = _extract_pixel_decoder_feature_dict(pixel_decoder_output)
    pixel_feats = _sort_feats_by_resolution(pixel_feats)

    attn_maps_up = {}
    raw_attn_maps = {}

    print('\n=== Pixel decoder features ===')
    print(f'Original image shape: D,H,W = {(D, H, W)}')
    print(f'Selected point z,y,x = {point_zyx}')

    for name, feat in pixel_feats.items():
        # feat: (1, C, d, h, w)
        feat = feat[0]  # (C, d, h, w)
        c, d, h, w = feat.shape

        zz = int(round(z / max(D - 1, 1) * max(d - 1, 1)))
        yy = int(round(y / max(H - 1, 1) * max(h - 1, 1)))
        xx = int(round(x / max(W - 1, 1) * max(w - 1, 1)))

        f = feat.float()

        if normalize:
            f = F.normalize(f, dim=0)

        q = f[:, zz, yy, xx]  # (C,)
        attn = torch.einsum('c,cdhw->dhw', q, f)  # (d,h,w)

        attn_min = attn.min()
        attn_max = attn.max()
        attn_vis = (attn - attn_min) / (attn_max - attn_min + 1e-6)

        attn_up = F.interpolate(
            attn_vis[None, None],
            size=(D, H, W),
            mode='trilinear',
            align_corners=False,
        )[0, 0]

        raw_attn_maps[name] = attn_vis.detach().cpu().numpy()
        attn_maps_up[name] = attn_up.detach().cpu().numpy()

        scale = (D / d, H / h, W / w)

        print(
            f'{name}: feat shape={(c, d, h, w)}, '
            f'approx scale zyx={scale}, '
            f'mapped point={(zz, yy, xx)}'
        )

    return attn_maps_up, raw_attn_maps, point_zyx, pixel_feats


def visualize_pixel_decoder_attention_for_sample(
    image_path,
    model,
    device,
    output_dir,
    point_zyx=None,
    random_point=False,
    gaussian_sigma=None,
    alpha=0.55,
    title_fontsize=28,
    point_size=70,
):
    """
    Visualize point feature-affinity maps from pixel decoder outputs.

    Display columns:
        1. Raw
        2. Fused Pixel Feature
        3. Feature Scale 1
        4. Feature Scale 2
        5. Feature Scale 3

    The transformer_encoder_features column is skipped.
    """

    os.makedirs(output_dir, exist_ok=True)

    image_tensor = preprocess_image(
        image_path,
        gaussian_sigma=gaussian_sigma,
    ).to(device)  # (C, D, H, W)

    image_np = image_tensor[0].detach().cpu().numpy()
    D, H, W = image_np.shape

    attn_maps_up, raw_attn_maps, point_zyx, pixel_feats = (
        compute_point_pixel_decoder_attention_maps(
            image_tensor=image_tensor,
            model=model,
            point_zyx=point_zyx,
            random_point=random_point,
            normalize=True,
        )
    )

    z0, y0, x0 = point_zyx
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ------------------------------------------------------------
    # Select maps to display
    # Skip transformer_encoder_features
    # ------------------------------------------------------------
    fused_item = None
    feature_items = []

    for name, attn in attn_maps_up.items():
        if name == 'transformer_encoder_features':
            continue

        if name == 'mask_features':
            fused_item = (name, attn)
        elif name.startswith('multi_scale_'):
            feature_items.append((name, attn))

    # Sort multi-scale features by index: multi_scale_0, multi_scale_1, ...
    def _get_scale_id(item):
        name = item[0]
        try:
            return int(name.split('_')[-1])
        except Exception:
            return 999

    feature_items = sorted(feature_items, key=_get_scale_id)

    # Keep only three feature scales for columns 3-5
    feature_items = feature_items[:3]

    display_items = []

    if fused_item is not None:
        display_items.append(
            ('Fused Pixel Feature', fused_item[0], fused_item[1])
        )

    for i, (name, attn) in enumerate(feature_items, start=1):
        display_items.append((f'Feature Scale {i}', name, attn))

    if len(display_items) == 0:
        raise RuntimeError(
            'No valid pixel decoder features found for visualization. '
            'Expected mask_features and/or multi_scale_* features.'
        )

    # ------------------------------------------------------------
    # Save displayed 3D attention maps only
    # ------------------------------------------------------------
    for title, name, attn in display_items:
        save_path = os.path.join(
            output_dir,
            f'{base}_pixel_decoder_{name}_point_{z0}_{y0}_{x0}_attn_up.tif',
        )
        tifffile.imwrite(save_path, attn.astype(np.float32), compression=None)
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------
    # Plot selected z slice
    # ------------------------------------------------------------
    n_cols = len(display_items) + 1

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(5.6 * n_cols, 5.8),
    )

    if n_cols == 1:
        axes = [axes]

    # Column 1: Raw
    axes[0].imshow(image_np[z0], cmap='gray')
    axes[0].scatter([x0], [y0], c='red', s=point_size)
    axes[0].set_title(
        'Raw',
        fontsize=title_fontsize,
        fontweight='normal',
        pad=16,
    )
    axes[0].axis('off')

    # Columns 2-5: selected pixel decoder features
    for ax, (title, name, attn) in zip(axes[1:], display_items):
        ax.imshow(image_np[z0], cmap='gray')
        ax.imshow(attn[z0], cmap='jet', alpha=alpha, vmin=0, vmax=1)
        ax.scatter([x0], [y0], c='white', s=point_size)

        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight='normal',
            pad=16,
        )
        ax.axis('off')

    plt.tight_layout()

    svg_path = os.path.join(
        output_dir,
        f'{base}_pixel_decoder_point_{z0}_{y0}_{x0}_feature_attention.svg',
    )
    plt.savefig(svg_path, dpi=200, bbox_inches='tight')
    print(f'Saved overlay png: {svg_path}')

    plt.show()

    return {
        'attn_maps_up': attn_maps_up,
        'raw_attn_maps': raw_attn_maps,
        'point_zyx': point_zyx,
        'display_items': display_items,
    }


def _find_transformer_decoder(model):
    """
    Try to find Mask2Former transformer decoder / predictor module.
    """
    candidates = [
        'sem_seg_head.predictor',
        'predictor',
        'transformer_decoder',
        'sem_seg_head.transformer_decoder',
    ]

    for name in candidates:
        obj = model
        ok = True
        for part in name.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, 'forward_prediction_heads'):
            print(f'Found transformer decoder at: model.{name}')
            return obj

    raise RuntimeError(
        'Could not find transformer decoder with forward_prediction_heads(). '
        'Please check whether it is model.sem_seg_head.predictor or another name.'
    )


def _find_pixel_decoder(model):
    """
    Try to find pixel decoder module.
    """
    candidates = [
        'pixel_decoder',
        'sem_seg_head.pixel_decoder',
    ]

    for name in candidates:
        obj = model
        ok = True
        for part in name.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, 'forward_features'):
            print(f'Found pixel decoder at: model.{name}')
            return obj

    raise RuntimeError('Could not find pixel decoder with forward_features().')


@torch.no_grad()
def debug_query_mask_dot_chain(
    image_path,
    model,
    device,
    point_zyx=(15, 30, 26),
    output_dir=None,
    gaussian_sigma=None,
    topk=10,
    save_tif=True,
    alpha=0.55,
):
    """
    Debug the exact chain:

        mask_features: (B, C, Dm, Hm, Wm)
        mask_embed:    (B, Q, C)
        pred_masks:    (B, Q, Dm, Hm, Wm)

        pred_masks = einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)

    Then find which query has the strongest response at point_zyx.

    Args:
        image_path: input 3D tif path
        model: Mask2Former model
        device: cuda/cpu
        point_zyx: original image coordinate, e.g. (15, 30, 26)
        output_dir: folder to save debug images
        gaussian_sigma: optional preprocessing smoothing
        topk: print top-k queries at this point
        save_tif: whether to save query mask volumes
        alpha: overlay alpha
    """

    model.eval()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Prepare input
    # ------------------------------------------------------------
    image_tensor = preprocess_image(
        image_path,
        gaussian_sigma=gaussian_sigma,
    ).to(device)  # (C, D, H, W)

    C_in, D, H, W = image_tensor.shape
    z, y, x = point_zyx
    z = int(np.clip(z, 0, D - 1))
    y = int(np.clip(y, 0, H - 1))
    x = int(np.clip(x, 0, W - 1))
    point_zyx = (z, y, x)

    inputs = [{'image': image_tensor}]

    image_np = image_tensor[0].detach().cpu().numpy()
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ------------------------------------------------------------
    # 2. Locate modules
    # ------------------------------------------------------------
    pixel_decoder = _find_pixel_decoder(model)
    transformer_decoder = _find_transformer_decoder(model)

    cache = {
        'mask_features': None,
        'transformer_encoder_features': None,
        'multi_scale_features': None,
        'decoder_output': None,
        'mask_embed': None,
        'outputs_mask': None,
        'outputs_class': None,
        'attn_mask': None,
    }

    # ------------------------------------------------------------
    # 3. Patch pixel_decoder.forward_features to capture mask_features
    # ------------------------------------------------------------
    old_pixel_forward_features = pixel_decoder.forward_features

    def wrapped_pixel_forward_features(*args, **kwargs):
        out = old_pixel_forward_features(*args, **kwargs)

        if isinstance(out, (list, tuple)) and len(out) >= 3:
            cache['mask_features'] = out[0]
            cache['transformer_encoder_features'] = out[1]
            cache['multi_scale_features'] = out[2]

        return out

    pixel_decoder.forward_features = wrapped_pixel_forward_features

    # ------------------------------------------------------------
    # 4. Patch transformer_decoder.forward_prediction_heads
    # ------------------------------------------------------------
    old_forward_prediction_heads = transformer_decoder.forward_prediction_heads

    def wrapped_forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size
    ):
        """
        This follows the normal Mask2Former logic, but records intermediate tensors.
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (B, Q, C)

        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        outputs_mask = torch.einsum(
            'bqc,bcdhw->bqdhw',
            mask_embed,
            mask_features,
        )

        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode='trilinear',
            align_corners=False,
        )

        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()

        attn_mask = attn_mask.detach()

        cache['decoder_output'] = decoder_output.detach()
        cache['mask_embed'] = mask_embed.detach()
        cache['outputs_mask'] = outputs_mask.detach()
        cache['outputs_class'] = outputs_class.detach()
        cache['attn_mask'] = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    transformer_decoder.forward_prediction_heads = types.MethodType(
        wrapped_forward_prediction_heads,
        transformer_decoder,
    )

    # ------------------------------------------------------------
    # 5. Run normal model forward
    # ------------------------------------------------------------
    try:
        outputs = model(inputs)[0]
    finally:
        # Always restore original functions
        pixel_decoder.forward_features = old_pixel_forward_features
        transformer_decoder.forward_prediction_heads = (
            old_forward_prediction_heads
        )

    # ------------------------------------------------------------
    # 6. Check captured tensors
    # ------------------------------------------------------------
    mask_features = cache['mask_features']
    mask_embed = cache['mask_embed']
    outputs_mask = cache['outputs_mask']
    outputs_class = cache['outputs_class']

    if mask_features is None:
        raise RuntimeError('Failed to capture mask_features.')

    if mask_embed is None or outputs_mask is None:
        raise RuntimeError(
            'Failed to capture mask_embed / outputs_mask. '
            'Maybe your forward_prediction_heads implementation is different.'
        )

    print('\n=== Captured query-mask dot-product chain ===')
    print(f'image_tensor:  {tuple(image_tensor.shape)}')
    print(f'mask_features: {tuple(mask_features.shape)}')
    print(f'mask_embed:    {tuple(mask_embed.shape)}')
    print(f'outputs_mask:  {tuple(outputs_mask.shape)}')
    print(f'outputs_class: {tuple(outputs_class.shape)}')
    print(f'point_zyx:     {point_zyx}')

    # ------------------------------------------------------------
    # 7. Verify dot product manually
    # ------------------------------------------------------------
    manual_outputs_mask = torch.einsum(
        'bqc,bcdhw->bqdhw',
        mask_embed,
        mask_features,
    )

    max_abs_diff = (manual_outputs_mask - outputs_mask).abs().max().item()
    mean_abs_diff = (manual_outputs_mask - outputs_mask).abs().mean().item()

    print('\n=== Dot-product verification ===')
    print(f'max_abs_diff  = {max_abs_diff:.8f}')
    print(f'mean_abs_diff = {mean_abs_diff:.8f}')

    # ------------------------------------------------------------
    # 8. Upsample query masks to original image size
    # ------------------------------------------------------------
    mask_logits_up = F.interpolate(
        outputs_mask.float(),
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )  # (B, Q, D, H, W)

    mask_prob_up = mask_logits_up.sigmoid()[0]  # (Q, D, H, W)

    # Query response at selected point
    point_scores = mask_prob_up[:, z, y, x]  # (Q,)
    top_vals, top_ids = torch.topk(
        point_scores, k=min(topk, point_scores.numel())
    )

    print('\n=== Top queries at selected point ===')

    cls_prob = outputs_class.softmax(dim=-1)[0]  # (Q, num_classes)

    for rank, (qid, val) in enumerate(
        zip(top_ids.tolist(), top_vals.tolist()), start=1
    ):
        logit_val = mask_logits_up[0, qid, z, y, x].item()

        cls_logits = outputs_class[0, qid].detach().cpu().numpy()
        cls_probs = cls_prob[qid].detach().cpu().numpy()

        pred_cls = int(cls_prob[qid].argmax().item())
        pred_cls_score = float(cls_prob[qid, pred_cls].item())

        # 如果你的 class 0 是 cell，class 1 是 no-object，一般看这个
        cell_score = float(cls_prob[qid, 0].item())
        noobj_score = (
            float(cls_prob[qid, 1].item()) if cls_prob.shape[1] > 1 else None
        )

        print(
            f'rank={rank:02d} | query={qid:03d} | '
            f'mask_prob_at_point={val:.6f} | '
            f'mask_logit_at_point={logit_val:.6f} | '
            f'pred_cls={pred_cls} | '
            f'pred_cls_score={pred_cls_score:.6f} | '
            f'cell_score={cell_score:.6f} | '
            f'noobj_score={noobj_score:.6f} | '
            f'cls_logits={cls_logits}'
        )

    best_q = int(top_ids[0].item())
    best_prob = float(top_vals[0].item())

    print('\n=== Selected query ===')
    print(f'best_query = {best_q}')
    print(f'prob_at_point = {best_prob:.6f}')

    # ------------------------------------------------------------
    # 9. Inspect local dot product at this coordinate
    # ------------------------------------------------------------
    mf = mask_features[0]  # (C, Dm, Hm, Wm)
    _, C_mask, Dm, Hm, Wm = mask_features.shape

    zz = int(round(z / max(D - 1, 1) * max(Dm - 1, 1)))
    yy = int(round(y / max(H - 1, 1) * max(Hm - 1, 1)))
    xx = int(round(x / max(W - 1, 1) * max(Wm - 1, 1)))

    query_vec = mask_embed[0, best_q]  # (C,)
    voxel_vec = mf[:, zz, yy, xx]  # (C,)
    dot_logit = torch.dot(query_vec, voxel_vec).item()
    raw_logit = outputs_mask[0, best_q, zz, yy, xx].item()

    print('\n=== Local dot product at mapped mask-feature coordinate ===')
    print(f'original point zyx       = {(z, y, x)}')
    print(f'mask_feature point zyx   = {(zz, yy, xx)}')
    print(f'dot(mask_embed[q], mask_features[:, z,y,x]) = {dot_logit:.6f}')
    print(f'outputs_mask[q,z,y,x]                       = {raw_logit:.6f}')
    print(
        f'abs diff                                      = {abs(dot_logit - raw_logit):.8f}'
    )

    # ------------------------------------------------------------
    # 10. Optional: channel contribution analysis
    # ------------------------------------------------------------
    channel_contrib = (query_vec * voxel_vec).detach().cpu().numpy()
    top_ch = np.argsort(np.abs(channel_contrib))[::-1][:20]

    print('\n=== Top channel contributions for selected point ===')
    for i, ch in enumerate(top_ch, start=1):
        print(
            f'rank={i:02d} | channel={int(ch):03d} | '
            f'contribution={channel_contrib[ch]:.6f}'
        )

    # ------------------------------------------------------------
    # 11. Save / visualize top-k query masks
    # ------------------------------------------------------------
    vis_query_ids = top_ids[:5].tolist()  # visualize top 5 queries

    selected_probs = {
        qid: mask_prob_up[qid].detach().cpu().numpy() for qid in vis_query_ids
    }

    selected_logits = {
        qid: mask_logits_up[0, qid].detach().cpu().numpy()
        for qid in vis_query_ids
    }

    if output_dir is not None and save_tif:
        for qid in vis_query_ids:
            prob_path = os.path.join(
                output_dir,
                f'{base}_query_{qid:03d}_prob_up_point_{z}_{y}_{x}.tif',
            )
            logit_path = os.path.join(
                output_dir,
                f'{base}_query_{qid:03d}_logit_up_point_{z}_{y}_{x}.tif',
            )

            tifffile.imwrite(
                prob_path,
                selected_probs[qid].astype(np.float32),
                compression=None,
            )
            tifffile.imwrite(
                logit_path,
                selected_logits[qid].astype(np.float32),
                compression=None,
            )

            print(f'Saved query {qid:03d} prob:  {prob_path}')
            print(f'Saved query {qid:03d} logit: {logit_path}')

    # Plot top-5 query masks on selected z slice
    n_vis = len(vis_query_ids)
    fig, axes = plt.subplots(
        2,
        n_vis + 1,
        figsize=(5 * (n_vis + 1), 10),
    )

    # Raw image
    axes[0, 0].imshow(image_np[z], cmap='gray')
    axes[0, 0].scatter([x], [y], c='red', s=40)
    axes[0, 0].set_title(f'Raw image z={z}')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image_np[z], cmap='gray')
    axes[1, 0].scatter([x], [y], c='red', s=40)
    axes[1, 0].set_title('Raw image')
    axes[1, 0].axis('off')

    for col, qid in enumerate(vis_query_ids, start=1):
        prob_map = selected_probs[qid]
        point_prob = float(mask_prob_up[qid, z, y, x].item())
        point_logit = float(mask_logits_up[0, qid, z, y, x].item())

        # Pure probability map
        axes[0, col].imshow(prob_map[z], cmap='jet', vmin=0, vmax=1)
        axes[0, col].scatter([x], [y], c='white', s=40)
        axes[0, col].set_title(
            f'Q{qid:03d}\nprob={point_prob:.4f}, logit={point_logit:.2f}'
        )
        axes[0, col].axis('off')

        # Overlay
        axes[1, col].imshow(image_np[z], cmap='gray')
        axes[1, col].imshow(
            prob_map[z], cmap='jet', alpha=alpha, vmin=0, vmax=1
        )
        axes[1, col].scatter([x], [y], c='white', s=40)
        axes[1, col].set_title(f'Overlay Q{qid:03d}')
        axes[1, col].axis('off')

    plt.tight_layout()

    if output_dir is not None:
        png_path = os.path.join(
            output_dir, f'{base}_top5_queries_point_{z}_{y}_{x}_dot_chain.png'
        )
        plt.savefig(png_path, dpi=200)
        print(f'Saved top-5 query overlay png: {png_path}')

    plt.show()

    return {
        'outputs': outputs,
        'mask_features': mask_features.detach().cpu(),
        'mask_embed': mask_embed.detach().cpu(),
        'outputs_mask': outputs_mask.detach().cpu(),
        'outputs_class': outputs_class.detach().cpu(),
        'mask_prob_up': mask_prob_up.detach().cpu(),
        'point_zyx': point_zyx,
        'best_query': best_q,
        'best_prob': best_prob,
        'top_query_ids': top_ids.detach().cpu().numpy(),
        'top_query_probs': top_vals.detach().cpu().numpy(),
        'mapped_mask_feature_point': (zz, yy, xx),
        'channel_contrib': channel_contrib,
    }


@torch.no_grad()
def debug_query_embed_feat_dot_chain(
    image_path,
    model,
    device,
    point_zyx=(15, 30, 26),
    output_dir=None,
    gaussian_sigma=None,
    feature_level=0,  # kept for compatibility; not used
    topk=10,
    vis_topk=5,
    normalize=True,  # kept for compatibility; not used
    save_tif=True,
    alpha=0.85,
    binary_thresh=0.5,
    drop_pre_decoder_prediction=True,
    cmap='magma',
    display_mode='prob_only',  # "prob_only" or "overlay"
    use_percentile_contrast=True,
    contrast_percentiles=(1, 99.5),
    add_colorbar=False,
    title_fontsize=36,
    row_label_fontsize=26,
    point_size=75,
):
    """
    Visualize query_pos / query_feat / query_feat + query_pos mask prediction evolution.

    Main changes:
        - Only visualize foreground queries that actually cover the selected point.
        - Background / no-object queries are filtered out.
        - Column titles are paper-friendly:
              Raw | Initial Mask | Layer 0 | Layer 1 | ...
        - No probability text is shown in figure titles.
    """

    import os
    import types

    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    import torch
    import torch.nn.functional as F

    model.eval()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Prepare input
    # ------------------------------------------------------------
    image_tensor = preprocess_image(
        image_path,
        gaussian_sigma=gaussian_sigma,
    ).to(device)  # [C, D, H, W]

    C_in, D, H, W = image_tensor.shape

    z, y, x = point_zyx
    z = int(np.clip(z, 0, D - 1))
    y = int(np.clip(y, 0, H - 1))
    x = int(np.clip(x, 0, W - 1))
    point_zyx = (z, y, x)

    inputs = [{'image': image_tensor}]
    image_np = image_tensor[0].detach().cpu().numpy()
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ------------------------------------------------------------
    # 2. Locate modules
    # ------------------------------------------------------------
    pixel_decoder = _find_pixel_decoder(model)
    transformer_decoder = _find_transformer_decoder(model)

    cache = {
        'mask_features': None,
        'multi_scale_features': None,
        'decoder_outputs_all': [],  # each: [B, Q, C]
        'outputs_class_all': [],  # each: [B, Q, num_classes + 1]
        'outputs_mask_all': [],  # each: [B, Q, Dm, Hm, Wm]
        'attn_mask_all': [],
    }

    # ------------------------------------------------------------
    # 3. Patch pixel_decoder.forward_features
    # ------------------------------------------------------------
    old_pixel_forward_features = pixel_decoder.forward_features

    def wrapped_pixel_forward_features(*args, **kwargs):
        out = old_pixel_forward_features(*args, **kwargs)

        if isinstance(out, (list, tuple)) and len(out) >= 3:
            cache['mask_features'] = out[0]
            cache['multi_scale_features'] = out[2]

        return out

    pixel_decoder.forward_features = wrapped_pixel_forward_features

    # ------------------------------------------------------------
    # 4. Patch transformer_decoder.forward_prediction_heads
    # ------------------------------------------------------------
    old_forward_prediction_heads = transformer_decoder.forward_prediction_heads

    def wrapped_forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size
    ):
        """
        Same logic as normal Mask2Former prediction head,
        but records every prediction-head call.
        """
        decoder_output = self.decoder_norm(output)  # [Q, B, C]
        decoder_output = decoder_output.transpose(0, 1)  # [B, Q, C]

        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        outputs_mask = torch.einsum(
            'bqc,bcdhw->bqdhw',
            mask_embed,
            mask_features,
        )

        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode='trilinear',
            align_corners=False,
        )

        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()

        attn_mask = attn_mask.detach()

        cache['decoder_outputs_all'].append(decoder_output.detach())
        cache['outputs_class_all'].append(outputs_class.detach())
        cache['outputs_mask_all'].append(outputs_mask.detach())
        cache['attn_mask_all'].append(attn_mask.detach())

        return outputs_class, outputs_mask, attn_mask

    transformer_decoder.forward_prediction_heads = types.MethodType(
        wrapped_forward_prediction_heads,
        transformer_decoder,
    )

    # ------------------------------------------------------------
    # 5. Run normal model forward
    # ------------------------------------------------------------
    try:
        outputs = model(inputs)[0]
    finally:
        pixel_decoder.forward_features = old_pixel_forward_features
        transformer_decoder.forward_prediction_heads = (
            old_forward_prediction_heads
        )

    # ------------------------------------------------------------
    # 6. Check captured tensors
    # ------------------------------------------------------------
    mask_features = cache['mask_features']
    decoder_outputs_all = cache['decoder_outputs_all']
    outputs_class_all = cache['outputs_class_all']
    outputs_mask_all = cache['outputs_mask_all']

    if mask_features is None:
        raise RuntimeError('Failed to capture mask_features.')

    if len(decoder_outputs_all) == 0:
        raise RuntimeError(
            'Failed to capture decoder outputs from forward_prediction_heads.'
        )

    if len(outputs_class_all) != len(decoder_outputs_all):
        raise RuntimeError(
            'outputs_class_all and decoder_outputs_all length mismatch.'
        )

    if len(outputs_mask_all) != len(decoder_outputs_all):
        raise RuntimeError(
            'outputs_mask_all and decoder_outputs_all length mismatch.'
        )

    B, C_mask, Dm, Hm, Wm = mask_features.shape

    # ------------------------------------------------------------
    # 7. Decide which prediction-head calls correspond to decoder layers
    # ------------------------------------------------------------
    if drop_pre_decoder_prediction and len(decoder_outputs_all) > 1:
        pred_start = 1
    else:
        pred_start = 0

    layer_query_feats = decoder_outputs_all[pred_start:]
    layer_outputs_class = outputs_class_all[pred_start:]
    layer_outputs_mask = outputs_mask_all[pred_start:]

    num_layers = len(layer_query_feats)

    # Use compact internal names and paper-friendly display names
    layer_names = [f'layer{i}' for i in range(num_layers)]
    layer_display_names = [f'Layer {i}' for i in range(num_layers)]

    final_outputs_class = layer_outputs_class[-1]
    final_outputs_mask = layer_outputs_mask[-1]

    print('\n=== Captured decoder prediction heads ===')
    print(f'total prediction-head calls captured = {len(decoder_outputs_all)}')
    print(
        f'drop_pre_decoder_prediction          = {drop_pre_decoder_prediction}'
    )
    print(f'visualized decoder-layer outputs     = {num_layers}')
    print(
        f'image_tensor                         = {tuple(image_tensor.shape)}'
    )
    print(
        f'mask_features                        = {tuple(mask_features.shape)}'
    )
    print(f'point_zyx                            = {point_zyx}')

    # ------------------------------------------------------------
    # 8. Prepare query sources
    # ------------------------------------------------------------
    query_pos_static = transformer_decoder.query_embed.weight.float()
    query_pos_static = query_pos_static.unsqueeze(0).repeat(
        B, 1, 1
    )  # [B, Q, C]

    query_feat_static = transformer_decoder.query_feat.weight.float()
    query_feat_static = query_feat_static.unsqueeze(0).repeat(
        B, 1, 1
    )  # [B, Q, C]

    query_init_sources = {
        'init_query_pos': query_pos_static,
        'init_query_feat': query_feat_static,
        'init_query_feat_plus_pos': query_feat_static + query_pos_static,
    }

    layer_query_sources = []

    for lname, qfeat in zip(layer_names, layer_query_feats):
        qfeat = qfeat.float()
        layer_query_sources.append(
            {
                'layer_name': lname,
                'query_pos': query_pos_static,
                'query_feat': qfeat,
                'query_feat_plus_pos': qfeat + query_pos_static,
            }
        )

    # ------------------------------------------------------------
    # 9. Helper: query source -> mask logits upsampled to original image size
    # ------------------------------------------------------------
    def query_to_masklogit_up(query_tensor_bqc):
        """
        query_tensor_bqc: [B, Q, C]

        Returns:
            mask_logits_up: [Q, D, H, W]
            mask_embed:     [B, Q, C_mask]
        """
        mask_embed = transformer_decoder.mask_embed(
            query_tensor_bqc
        )  # [B, Q, C_mask]

        mask_logits = torch.einsum(
            'bqc,bcdhw->bqdhw',
            mask_embed,
            mask_features.float(),
        )  # [B, Q, Dm, Hm, Wm]

        mask_logits_up = F.interpolate(
            mask_logits.float(),
            size=(D, H, W),
            mode='trilinear',
            align_corners=False,
        )[0]  # [Q, D, H, W]

        return mask_logits_up, mask_embed

    # Initial query maps
    init_masklogits_up = {}
    init_maskembeds = {}

    for src_name, qsrc in query_init_sources.items():
        logit_up, mask_emb = query_to_masklogit_up(qsrc)
        init_masklogits_up[src_name] = logit_up
        init_maskembeds[src_name] = mask_emb

    # Decoder-layer maps
    layer_source_masklogits = []
    layer_source_maskembeds = []

    for item in layer_query_sources:
        lname = item['layer_name']

        logits_this_layer = {}
        embeds_this_layer = {}

        for src_name in ['query_pos', 'query_feat', 'query_feat_plus_pos']:
            logit_up, mask_emb = query_to_masklogit_up(item[src_name])
            logits_this_layer[src_name] = logit_up
            embeds_this_layer[src_name] = mask_emb

        layer_source_masklogits.append(
            {
                'layer_name': lname,
                'logits': logits_this_layer,
            }
        )
        layer_source_maskembeds.append(
            {
                'layer_name': lname,
                'maskembeds': embeds_this_layer,
            }
        )

    # Standard query_feat branch
    layer_masklogits_up = [
        x['logits']['query_feat'] for x in layer_source_masklogits
    ]

    layer_maskembeds = [
        x['maskembeds']['query_feat'] for x in layer_source_maskembeds
    ]

    # ------------------------------------------------------------
    # 10. Verify last layer query_feat equals final outputs_mask
    # ------------------------------------------------------------
    final_mask_logits_up = F.interpolate(
        final_outputs_mask.float(),
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )[0]  # [Q, D, H, W]

    diff_final = (layer_masklogits_up[-1] - final_mask_logits_up).abs()

    print('\n=== Last-layer verification ===')
    print('Last visualized layer should match final outputs_mask.')
    print(f'max_abs_diff  = {diff_final.max().item():.8f}')
    print(f'mean_abs_diff = {diff_final.mean().item():.8f}')

    # ------------------------------------------------------------
    # 11. Select foreground queries that actually cover selected point
    # ------------------------------------------------------------
    final_mask_prob_up = torch.sigmoid(layer_masklogits_up[-1])  # [Q, D, H, W]
    point_scores = final_mask_prob_up[:, z, y, x]  # [Q]

    cls_prob = final_outputs_class.softmax(dim=-1)[0]  # [Q, C+1]
    no_object_idx = cls_prob.shape[-1] - 1

    pred_cls = cls_prob.argmax(dim=-1)  # [Q]
    is_foreground = pred_cls != no_object_idx

    fg_scores = cls_prob[:, :no_object_idx].max(dim=-1).values  # [Q]
    noobj_scores = cls_prob[:, no_object_idx]  # [Q]

    covers_point = point_scores >= float(binary_thresh)

    # Strict condition:
    #   1) final class is foreground
    #   2) final mask covers the selected point
    valid_query_mask = is_foreground & covers_point
    valid_query_ids = torch.where(valid_query_mask)[0]

    if valid_query_ids.numel() == 0:
        print(
            '\n[Warning] No query satisfies both conditions: '
            f'foreground class and point mask probability >= {binary_thresh}.'
        )
        print('No background query will be visualized.')
        return {
            'outputs': outputs,
            'point_zyx': point_zyx,
            'feature_level': feature_level,
            'vis_query_ids': np.asarray([], dtype=np.int64),
            'top_query_ids': np.asarray([], dtype=np.int64),
            'top_query_probs': np.asarray([], dtype=np.float32),
            'mask_features': mask_features.detach().cpu(),
            'final_outputs_class': final_outputs_class.detach().cpu(),
            'final_outputs_mask': final_outputs_mask.detach().cpu(),
            'final_mask_prob_up': final_mask_prob_up.detach().cpu(),
        }

    valid_point_scores = point_scores[valid_query_ids]
    sorted_scores, sorted_order = torch.sort(
        valid_point_scores, descending=True
    )
    sorted_query_ids = valid_query_ids[sorted_order]

    # topk controls how many valid foreground queries are considered;
    # vis_topk controls how many are visualized.
    topk_valid = min(int(topk), sorted_query_ids.numel())
    top_ids = sorted_query_ids[:topk_valid]
    top_vals = point_scores[top_ids]

    n_show = min(int(vis_topk), top_ids.numel())
    vis_query_ids = top_ids[:n_show].tolist()

    print('\n=== Foreground queries covering selected point ===')
    print(f'binary_thresh = {binary_thresh}')
    print(f'num_valid_foreground_queries = {valid_query_ids.numel()}')
    print(f'num_visualized_queries = {len(vis_query_ids)}')

    for rank, qid in enumerate(vis_query_ids, start=1):
        qid_int = int(qid)
        print(
            f'rank={rank:02d} | query={qid_int:03d} | '
            f'point_mask_prob={float(point_scores[qid_int].item()):.6f} | '
            f'foreground_score={float(fg_scores[qid_int].item()):.6f} | '
            f'no_object_score={float(noobj_scores[qid_int].item()):.6f} | '
            f'pred_cls={int(pred_cls[qid_int].item())}'
        )

    # ------------------------------------------------------------
    # 11.5 Print query vector statistics
    # ------------------------------------------------------------
    def _print_query_vector_stats(name, query_tensor_bqc, query_ids):
        q = query_tensor_bqc[0].detach().float().cpu()  # [Q, C]

        print(f'\n=== Query vector stats: {name} ===')
        for qid in query_ids:
            vec = q[qid]
            norm = float(vec.norm().item())
            first5 = vec[:5].numpy().tolist()
            first5_str = ', '.join([f'{v:.6f}' for v in first5])

            print(f'query={qid:03d} | norm={norm:.6f} | first5=[{first5_str}]')

    query_ids_to_print = vis_query_ids

    _print_query_vector_stats(
        'init_query_pos',
        query_init_sources['init_query_pos'],
        query_ids_to_print,
    )

    _print_query_vector_stats(
        'init_query_feat',
        query_init_sources['init_query_feat'],
        query_ids_to_print,
    )

    _print_query_vector_stats(
        'init_query_feat_plus_pos',
        query_init_sources['init_query_feat_plus_pos'],
        query_ids_to_print,
    )

    for item in layer_query_sources:
        lname = item['layer_name']

        _print_query_vector_stats(
            f'{lname}_query_pos',
            item['query_pos'],
            query_ids_to_print,
        )

        _print_query_vector_stats(
            f'{lname}_query_feat',
            item['query_feat'],
            query_ids_to_print,
        )

        _print_query_vector_stats(
            f'{lname}_query_feat_plus_pos',
            item['query_feat_plus_pos'],
            query_ids_to_print,
        )

    # ------------------------------------------------------------
    # 12. Print map statistics
    # ------------------------------------------------------------
    def print_map_stats(name, x):
        x = x.detach().float()
        prob = torch.sigmoid(x)

        x_flat = x.flatten()
        p_flat = prob.flatten()

        print(
            f'{name}: '
            f'logit min={x.min().item():.3f}, '
            f'p01={torch.quantile(x_flat, 0.01).item():.3f}, '
            f'median={torch.quantile(x_flat, 0.50).item():.3f}, '
            f'p99={torch.quantile(x_flat, 0.99).item():.3f}, '
            f'max={x.max().item():.3f} | '
            f'prob min={prob.min().item():.3f}, '
            f'median={torch.quantile(p_flat, 0.50).item():.3f}, '
            f'p99={torch.quantile(p_flat, 0.99).item():.3f}, '
            f'max={prob.max().item():.3f}'
        )

    print('\n=== Map statistics for visualized foreground queries ===')
    for qid in vis_query_ids:
        print(f'\n--- query {qid:03d} ---')

        print_map_stats(
            'mask_embed(init_query_pos)',
            init_masklogits_up['init_query_pos'][qid],
        )
        print_map_stats(
            'mask_embed(init_query_feat)',
            init_masklogits_up['init_query_feat'][qid],
        )
        print_map_stats(
            'mask_embed(init_query_feat_plus_pos)',
            init_masklogits_up['init_query_feat_plus_pos'][qid],
        )

        for item in layer_source_masklogits:
            lname = item['layer_name']
            logits_dict = item['logits']

            print_map_stats(
                f'mask_embed({lname}_query_pos)',
                logits_dict['query_pos'][qid],
            )
            print_map_stats(
                f'mask_embed({lname}_query_feat)',
                logits_dict['query_feat'][qid],
            )
            print_map_stats(
                f'mask_embed({lname}_query_feat_plus_pos)',
                logits_dict['query_feat_plus_pos'][qid],
            )

    # ------------------------------------------------------------
    # 13. Save tif: continuous logits and probabilities
    # ------------------------------------------------------------
    if output_dir is not None and save_tif:
        for qid in vis_query_ids:
            # Initial query sources
            for src_name, logit_allq in init_masklogits_up.items():
                logit_np = logit_allq[qid].detach().cpu().numpy()
                prob_np = torch.sigmoid(logit_allq[qid]).detach().cpu().numpy()

                tifffile.imwrite(
                    os.path.join(
                        output_dir, f'{base}_Q{qid:03d}_{src_name}_logit.tif'
                    ),
                    logit_np.astype(np.float32),
                    compression=None,
                )
                tifffile.imwrite(
                    os.path.join(
                        output_dir, f'{base}_Q{qid:03d}_{src_name}_prob.tif'
                    ),
                    prob_np.astype(np.float32),
                    compression=None,
                )

            # Decoder-layer query sources
            for item in layer_source_masklogits:
                lname = item['layer_name']
                logits_dict = item['logits']

                for src_name in [
                    'query_pos',
                    'query_feat',
                    'query_feat_plus_pos',
                ]:
                    layer_logit = logits_dict[src_name][qid]
                    layer_prob = torch.sigmoid(layer_logit)

                    tifffile.imwrite(
                        os.path.join(
                            output_dir,
                            f'{base}_Q{qid:03d}_{lname}_{src_name}_logit.tif',
                        ),
                        layer_logit.detach().cpu().numpy().astype(np.float32),
                        compression=None,
                    )

                    tifffile.imwrite(
                        os.path.join(
                            output_dir,
                            f'{base}_Q{qid:03d}_{lname}_{src_name}_prob.tif',
                        ),
                        layer_prob.detach().cpu().numpy().astype(np.float32),
                        compression=None,
                    )

        print(
            '\nSaved selected foreground query position / feature / '
            f'feature+position probability/logit maps to: {output_dir}'
        )

    # ------------------------------------------------------------
    # 14. Visualization helpers
    # ------------------------------------------------------------
    def _get_vmin_vmax(prob_slice):
        prob_slice = np.asarray(prob_slice, dtype=np.float32)
        valid = prob_slice[np.isfinite(prob_slice)]

        if valid.size == 0:
            return 0.0, 1.0

        if use_percentile_contrast:
            p_low, p_high = contrast_percentiles
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        return vmin, vmax

    def _show_probability_map(
        ax,
        image_slice,
        prob_slice,
        point_xy,
        title,
    ):
        """
        Show continuous probability map.

        display_mode:
            - "prob_only": show probability map directly.
            - "overlay": overlay probability map on raw image.
        """
        prob_slice = np.asarray(prob_slice, dtype=np.float32)
        vmin, vmax = _get_vmin_vmax(prob_slice)

        if display_mode == 'overlay':
            ax.imshow(image_slice, cmap='gray')
            im = ax.imshow(
                prob_slice,
                cmap=cmap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            point_color = '#00E5FF'

        elif display_mode == 'prob_only':
            im = ax.imshow(
                prob_slice,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            point_color = '#00E5FF'

        else:
            raise ValueError(
                f'Unsupported display_mode={display_mode}. '
                "Use 'prob_only' or 'overlay'."
            )

        ax.scatter(
            [point_xy[0]],
            [point_xy[1]],
            c=point_color,
            s=point_size,
            edgecolors='black',
            linewidths=1.2,
        )

        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight='normal',
            pad=18,
        )
        ax.axis('off')
        return im

    # ------------------------------------------------------------
    # 15. Plot selected z slice
    #     Save three separate figures:
    #       1) query position
    #       2) query feature
    #       3) query feature + position
    # ------------------------------------------------------------
    n_vis = len(vis_query_ids)

    plot_groups = [
        {
            'group_name': 'query_position',
            'row_label': 'Query Position',
            'init_key': 'init_query_pos',
            'layer_key': 'query_pos',
            'save_suffix': 'query_position',
        },
        {
            'group_name': 'query_feature',
            'row_label': 'Query Feature',
            'init_key': 'init_query_feat',
            'layer_key': 'query_feat',
            'save_suffix': 'query_feature',
        },
        {
            'group_name': 'query_feature_plus_position',
            'row_label': 'Feature + Position',
            'init_key': 'init_query_feat_plus_pos',
            'layer_key': 'query_feat_plus_pos',
            'save_suffix': 'feature_plus_position',
        },
    ]

    # Columns:
    #   Raw | Initial Mask | Layer 0 | Layer 1 | ... | Layer last
    n_cols = 2 + num_layers

    saved_png_paths = []

    for group in plot_groups:
        fig, axes = plt.subplots(
            n_vis,
            n_cols,
            figsize=(4.2 * n_cols, 4.0 * n_vis),
        )

        if n_vis == 1:
            axes = axes[None, :]

        init_key = group['init_key']
        layer_key = group['layer_key']
        row_label = group['row_label']

        for q_i, qid in enumerate(vis_query_ids):
            col = 0

            # --------------------------------------------------------
            # Column 0: Raw
            # --------------------------------------------------------
            axes[q_i, col].imshow(image_np[z], cmap='gray')
            axes[q_i, col].scatter(
                [x],
                [y],
                c='red',
                s=point_size,
                edgecolors='black',
                linewidths=1.2,
            )

            axes[q_i, col].set_title(
                'Raw',
                fontsize=title_fontsize,
                fontweight='normal',
                pad=18,
            )

            axes[q_i, col].set_ylabel(
                f'Q{qid:03d}\n{row_label}',
                fontsize=row_label_fontsize,
                rotation=0,
                labelpad=92,
                va='center',
            )

            axes[q_i, col].axis('off')
            col += 1

            # --------------------------------------------------------
            # Column 1: Initial Mask
            # --------------------------------------------------------
            init_logit = init_masklogits_up[init_key][qid]  # [D, H, W]
            init_prob_map = torch.sigmoid(init_logit).detach().cpu().numpy()

            im = _show_probability_map(
                axes[q_i, col],
                image_np[z],
                init_prob_map[z],
                point_xy=(x, y),
                title='Initial Mask',
            )

            if add_colorbar:
                fig.colorbar(im, ax=axes[q_i, col], fraction=0.046, pad=0.04)

            col += 1

            # --------------------------------------------------------
            # Columns 2...: each decoder layer
            # --------------------------------------------------------
            for layer_i, item in enumerate(layer_source_masklogits):
                layer_logit = item['logits'][layer_key][qid]  # [D, H, W]
                layer_prob_map = (
                    torch.sigmoid(layer_logit).detach().cpu().numpy()
                )

                im = _show_probability_map(
                    axes[q_i, col],
                    image_np[z],
                    layer_prob_map[z],
                    point_xy=(x, y),
                    title=layer_display_names[layer_i],
                )

                if add_colorbar:
                    fig.colorbar(
                        im, ax=axes[q_i, col], fraction=0.046, pad=0.04
                    )

                col += 1

        plt.tight_layout()

        if output_dir is not None:
            png_path = os.path.join(
                output_dir,
                f'{base}_{group["save_suffix"]}_evolution_point_{z}_{y}_{x}.png',
            )
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            saved_png_paths.append(png_path)
            print(f'Saved {group["group_name"]} evolution png: {png_path}')

        plt.show()

    # ------------------------------------------------------------
    # 16. Return useful tensors
    # ------------------------------------------------------------
    return {
        'outputs': outputs,
        'point_zyx': point_zyx,
        'feature_level': feature_level,
        'top_query_ids': top_ids.detach().cpu().numpy(),
        'top_query_probs': top_vals.detach().cpu().numpy(),
        'vis_query_ids': np.asarray(vis_query_ids, dtype=np.int64),
        'foreground_query_mask': is_foreground.detach().cpu(),
        'point_cover_mask': covers_point.detach().cpu(),
        'valid_query_mask': valid_query_mask.detach().cpu(),
        'foreground_scores': fg_scores.detach().cpu(),
        'no_object_scores': noobj_scores.detach().cpu(),
        'point_scores': point_scores.detach().cpu(),
        'mask_features': mask_features.detach().cpu(),
        # Static learned queries
        'query_pos_static': query_pos_static.detach().cpu(),
        'query_feat_static': query_feat_static.detach().cpu(),
        'query_feat_plus_pos_static': (query_feat_static + query_pos_static)
        .detach()
        .cpu(),
        # Init query sources
        'query_init_sources': {
            k: v.detach().cpu() for k, v in query_init_sources.items()
        },
        'init_masklogits_up': {
            k: v.detach().cpu() for k, v in init_masklogits_up.items()
        },
        'init_maskembeds': {
            k: v.detach().cpu() for k, v in init_maskembeds.items()
        },
        # Decoder layer outputs
        'layer_names': layer_names,
        'layer_display_names': layer_display_names,
        'layer_query_feats': [x.detach().cpu() for x in layer_query_feats],
        'layer_outputs_class': [x.detach().cpu() for x in layer_outputs_class],
        'layer_outputs_mask': [x.detach().cpu() for x in layer_outputs_mask],
        # Per-layer query pos / feat / feat+pos
        'layer_query_sources': [
            {
                'layer_name': item['layer_name'],
                'query_pos': item['query_pos'].detach().cpu(),
                'query_feat': item['query_feat'].detach().cpu(),
                'query_feat_plus_pos': item['query_feat_plus_pos']
                .detach()
                .cpu(),
            }
            for item in layer_query_sources
        ],
        # Per-layer mask logits from pos / feat / feat+pos
        'layer_source_masklogits': [
            {
                'layer_name': item['layer_name'],
                'logits': {
                    k: v.detach().cpu() for k, v in item['logits'].items()
                },
            }
            for item in layer_source_masklogits
        ],
        'layer_source_maskembeds': [
            {
                'layer_name': item['layer_name'],
                'maskembeds': {
                    k: v.detach().cpu() for k, v in item['maskembeds'].items()
                },
            }
            for item in layer_source_maskembeds
        ],
        # Final prediction
        'final_outputs_class': final_outputs_class.detach().cpu(),
        'final_outputs_mask': final_outputs_mask.detach().cpu(),
        'final_mask_prob_up': final_mask_prob_up.detach().cpu(),
        # Old-compatible keys: query_feat branch only
        'layer_masklogits_up': [x.detach().cpu() for x in layer_masklogits_up],
        'layer_maskembeds': [x.detach().cpu() for x in layer_maskembeds],
    }


@torch.no_grad()
def visualize_query_initialization_3d_tif_for_sample(
    image_path,
    model,
    device,
    output_dir='./debug_query_init_3d_tif',
    gaussian_sigma=None,
    topk=None,
    marker_radius=(1, 4, 4),  # (rz, ry, rx)
    marker_alpha=0.90,
    side_alpha_scale=0.45,
    raw_percentiles=(1, 99.5),
    raw_cmap='gray',  # "gray" or "magma"
    save_points_csv=True,
    center_dot=True,
    center_outline=True,
    level_shapes=None,
    level_colors=None,
    raw_brightness=0.55,
):
    """
    Save a 3D RGB TIFF showing feature-based query initialization positions.

    Output:
        1) *_query_init_3d_overlay.tif
           Shape: (Z, Y, X, 3), uint8 RGB.
           Raw image is used as background.
           Query init locations are drawn as 3D markers.

        2) *_query_init_points.csv
           Contains rank, score, probability, feature level, feature coordinate,
           and mapped original image coordinate.

    Marker design:
        - color: foreground probability
        - shape: feature level
            level 0 -> filled circle
            level 1 -> hollow circle
            level 2 -> x cross
        - center z-slice is emphasized with stronger alpha and optional white center.
    """

    import csv
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    import torch

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    if level_shapes is None:
        level_shapes = {
            0: 'filled_circle',  # level 0: solid circle
            1: 'hollow_circle',  # level 1: hollow circle
            2: 'x_cross',  # level 2: x marker
        }

    if level_colors is None:
        level_colors = {
            0: (255, 40, 40),  # red
            1: (0, 200, 255),  # cyan-blue
            2: (0, 255, 120),  # green
        }

    fallback_level_colors = [
        (255, 40, 40),  # red
        (0, 200, 255),  # cyan-blue
        (0, 255, 120),  # green
        (255, 0, 200),  # magenta
        (255, 160, 0),  # orange
    ]

    # ------------------------------------------------------------
    # 1. Read and normalize input image
    # ------------------------------------------------------------
    image_tensor = preprocess_image(
        image_path,
        gaussian_sigma=gaussian_sigma,
    ).to(device)  # [C, D, H, W]

    C_in, D, H, W = image_tensor.shape
    image_np = image_tensor[0].detach().cpu().numpy()
    base = os.path.splitext(os.path.basename(image_path))[0]

    lo, hi = np.percentile(image_np, raw_percentiles)
    if hi <= lo:
        raw_norm = np.zeros_like(image_np, dtype=np.float32)
    else:
        raw_norm = np.clip((image_np - lo) / (hi - lo), 0, 1).astype(
            np.float32
        )

    if raw_cmap is None or raw_cmap.lower() == 'gray':
        rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    else:
        cmap_raw = plt.get_cmap(raw_cmap)
        rgb = cmap_raw(raw_norm)[..., :3]

    rgb = (rgb * 255).astype(np.uint8)  # [D, H, W, 3]

    # Darken raw background so colored query markers are more visible
    rgb = np.clip(rgb.astype(np.float32) * raw_brightness, 0, 255).astype(
        np.uint8
    )

    # ------------------------------------------------------------
    # 2. Locate modules
    # ------------------------------------------------------------
    pixel_decoder = _find_pixel_decoder(model)
    transformer_decoder = _find_transformer_decoder(model)

    if not getattr(transformer_decoder, 'feature_query_init', False):
        raise RuntimeError(
            'transformer_decoder.feature_query_init is False. '
            'This visualization is only meaningful for feature-based query initialization.'
        )

    # ------------------------------------------------------------
    # 3. Forward backbone + pixel decoder
    # ------------------------------------------------------------
    x = image_tensor
    x = (x - model.pixel_mean) / model.pixel_std
    images = torch.stack([x], dim=0)  # [1, C, D, H, W]

    features = model.backbone(images)
    mask_features, transformer_encoder_features, multi_scale_features = (
        pixel_decoder.forward_features(features)
    )

    num_feature_levels = transformer_decoder.num_feature_levels
    assert len(multi_scale_features) == num_feature_levels

    # ------------------------------------------------------------
    # 4. Reproduce feature-based query initialization
    # ------------------------------------------------------------
    src = []
    level_meta = []
    global_start = 0

    for level_id in range(num_feature_levels):
        feat = multi_scale_features[level_id]  # [B, C, d, h, w]
        B, C_feat, d_i, h_i, w_i = feat.shape

        projected = transformer_decoder.input_proj[level_id](
            feat
        )  # [B, C, d, h, w]
        projected = projected.flatten(2)  # [B, C, S]

        level_emb = transformer_decoder.level_embed.weight[level_id].view(
            1, -1, 1
        )
        projected = projected + level_emb

        src_level = projected.permute(2, 0, 1).contiguous()  # [S, B, C]
        src.append(src_level)

        num_tokens = d_i * h_i * w_i
        level_meta.append(
            {
                'level': level_id,
                'shape': (d_i, h_i, w_i),
                'start': global_start,
                'end': global_start + num_tokens,
                'num_tokens': num_tokens,
            }
        )
        global_start += num_tokens

    memory = torch.cat(
        [s.permute(1, 0, 2) for s in src],
        dim=1,
    )  # [B, S_total, C]

    memory_for_select = transformer_decoder.enc_output_norm(
        transformer_decoder.enc_output(memory)
    )  # [B, S_total, C]

    enc_logits = transformer_decoder.class_embed(memory_for_select)

    if enc_logits.shape[-1] == transformer_decoder.num_classes + 1:
        fg_logits = enc_logits[..., : transformer_decoder.num_classes]
        dense_scores = fg_logits.max(dim=-1).values
        dense_classes = fg_logits.argmax(dim=-1)
        dense_probs = (
            enc_logits.softmax(dim=-1)[..., : transformer_decoder.num_classes]
            .max(dim=-1)
            .values
        )
    else:
        dense_scores = enc_logits.max(dim=-1).values
        dense_classes = enc_logits.argmax(dim=-1)
        dense_probs = enc_logits.softmax(dim=-1).max(dim=-1).values

    if topk is None:
        topk = transformer_decoder.num_queries

    topk = min(int(topk), dense_scores.shape[1])

    top_scores, top_indices = torch.topk(
        dense_scores[0],
        k=topk,
        dim=0,
        sorted=True,
    )

    top_probs = dense_probs[0, top_indices]
    top_classes = dense_classes[0, top_indices]

    # ------------------------------------------------------------
    # 5. Map global token index -> feature coordinate -> raw image coordinate
    # ------------------------------------------------------------
    rows = []

    for rank, token_idx_t in enumerate(top_indices):
        token_idx = int(token_idx_t.item())
        score = float(top_scores[rank].item())
        prob = float(top_probs[rank].item())
        pred_class = int(top_classes[rank].item())

        selected_level = None

        for meta in level_meta:
            if meta['start'] <= token_idx < meta['end']:
                selected_level = meta
                break

        if selected_level is None:
            raise RuntimeError(
                f'Cannot map token_idx={token_idx} to any feature level.'
            )

        level_id = selected_level['level']
        d_i, h_i, w_i = selected_level['shape']
        local_idx = token_idx - selected_level['start']

        zf = local_idx // (h_i * w_i)
        rem = local_idx % (h_i * w_i)
        yf = rem // w_i
        xf = rem % w_i

        z_img = (zf + 0.5) / d_i * D - 0.5
        y_img = (yf + 0.5) / h_i * H - 0.5
        x_img = (xf + 0.5) / w_i * W - 0.5

        z_img = float(np.clip(z_img, 0, D - 1))
        y_img = float(np.clip(y_img, 0, H - 1))
        x_img = float(np.clip(x_img, 0, W - 1))

        rows.append(
            {
                'rank': rank + 1,
                'query_id_sorted': rank,
                'token_index_global': token_idx,
                'score_logit': score,
                'score_prob': prob,
                'pred_class': pred_class,
                'level': level_id,
                'feature_d': d_i,
                'feature_h': h_i,
                'feature_w': w_i,
                'feature_z': int(zf),
                'feature_y': int(yf),
                'feature_x': int(xf),
                'image_z': z_img,
                'image_y': y_img,
                'image_x': x_img,
            }
        )

    # ------------------------------------------------------------
    # 6. Save query point table
    # ------------------------------------------------------------
    csv_path = None

    if save_points_csv:
        csv_path = os.path.join(output_dir, f'{base}_query_init_points.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f'Saved query-init points CSV: {csv_path}')

    # ------------------------------------------------------------
    # 7. Helper functions for voxel marker drawing
    # ------------------------------------------------------------
    rz, ry, rx = marker_radius
    rz = max(int(rz), 0)
    ry = max(int(ry), 1)
    rx = max(int(rx), 1)

    def _get_level_color(level_id):
        if level_id in level_colors:
            return np.array(level_colors[level_id], dtype=np.float32)

        return np.array(
            fallback_level_colors[level_id % len(fallback_level_colors)],
            dtype=np.float32,
        )

    def _blend_pixel(rgb_vol, zz, yy, xx, color_rgb, alpha):
        old = rgb_vol[zz, yy, xx].astype(np.float32)
        new = (1.0 - alpha) * old + alpha * color_rgb
        rgb_vol[zz, yy, xx] = np.clip(new, 0, 255).astype(np.uint8)

    def _draw_symbol_on_slice(
        rgb_vol,
        z0,
        y0,
        x0,
        zz,
        color_rgb,
        shape_name,
        alpha,
        center_slice=False,
    ):
        """
        Draw one 2D symbol on slice zz.
        Color indicates feature level.
        Shape indicates feature level.
        Center slice is emphasized by black outline and black center dot.
        """

        dz = abs(zz - z0)

        if rz == 0:
            z_scale = 1.0
        else:
            z_scale = np.sqrt(max(0.0, 1.0 - (dz / max(rz, 1)) ** 2))

        # Side slices are slightly smaller
        cur_ry = max(1, int(round(ry * z_scale)))
        cur_rx = max(1, int(round(rx * z_scale)))

        if center_slice:
            cur_ry = max(cur_ry, ry)
            cur_rx = max(cur_rx, rx)

        y_min = max(0, y0 - cur_ry)
        y_max = min(H - 1, y0 + cur_ry)
        x_min = max(0, x0 - cur_rx)
        x_max = min(W - 1, x0 + cur_rx)

        yy_grid, xx_grid = np.meshgrid(
            np.arange(y_min, y_max + 1),
            np.arange(x_min, x_max + 1),
            indexing='ij',
        )

        dy = (yy_grid - y0) / max(cur_ry, 1)
        dx = (xx_grid - x0) / max(cur_rx, 1)
        dist2 = dy * dy + dx * dx

        shape_name = shape_name.lower()

        if shape_name == 'filled_circle':
            mask = dist2 <= 1.0

        elif shape_name == 'hollow_circle':
            inner = 0.45 if min(cur_ry, cur_rx) <= 2 else 0.55
            mask = (dist2 <= 1.0) & (dist2 >= inner**2)

        elif shape_name == 'x_cross':
            line_width = 0.18 if center_slice else 0.14
            diag1 = np.abs(dy - dx) <= line_width
            diag2 = np.abs(dy + dx) <= line_width
            mask = (diag1 | diag2) & (dist2 <= 1.15)

        elif shape_name == 'plus':
            line_width = 0.18 if center_slice else 0.14
            vertical = np.abs(dx) <= line_width
            horizontal = np.abs(dy) <= line_width
            mask = (vertical | horizontal) & (dist2 <= 1.15)

        elif shape_name == 'diamond':
            mask = (np.abs(dy) + np.abs(dx)) <= 1.0

        else:
            mask = dist2 <= 1.0

        ys = yy_grid[mask]
        xs = xx_grid[mask]

        for yy, xx in zip(ys, xs):
            _blend_pixel(rgb_vol, zz, int(yy), int(xx), color_rgb, alpha)

        # Center slice special black outline
        if center_slice and center_outline:
            outline_mask = (dist2 <= 1.18) & (dist2 >= 0.86)
            ys_o = yy_grid[outline_mask]
            xs_o = xx_grid[outline_mask]

            black = np.array([0, 0, 0], dtype=np.float32)
            for yy, xx in zip(ys_o, xs_o):
                _blend_pixel(rgb_vol, zz, int(yy), int(xx), black, 0.90)

        # Center black dot
        if center_slice and center_dot:
            y_c = int(np.clip(y0, 0, H - 1))
            x_c = int(np.clip(x0, 0, W - 1))

            black = np.array([0, 0, 0], dtype=np.uint8)

            dot_r = 1
            for yy in range(max(0, y_c - dot_r), min(H - 1, y_c + dot_r) + 1):
                for xx in range(
                    max(0, x_c - dot_r), min(W - 1, x_c + dot_r) + 1
                ):
                    if (yy - y_c) ** 2 + (xx - x_c) ** 2 <= dot_r**2:
                        rgb_vol[z0, yy, xx] = black

    # ------------------------------------------------------------
    # 8. Draw 3D markers
    # ------------------------------------------------------------
    # Draw from low score to high score only to keep stable overlay order.
    # Color no longer means confidence; color means feature level.
    for idx in reversed(range(len(rows))):
        r = rows[idx]

        z0 = int(round(r['image_z']))
        y0 = int(round(r['image_y']))
        x0 = int(round(r['image_x']))
        level_id = int(r['level'])

        shape_name = level_shapes.get(level_id, 'filled_circle')
        color = _get_level_color(level_id)

        z_min = max(0, z0 - rz)
        z_max = min(D - 1, z0 + rz)

        for zz in range(z_min, z_max + 1):
            center_slice = zz == z0

            if center_slice:
                alpha_this_slice = marker_alpha
            else:
                alpha_this_slice = marker_alpha * side_alpha_scale

            _draw_symbol_on_slice(
                rgb_vol=rgb,
                z0=z0,
                y0=y0,
                x0=x0,
                zz=zz,
                color_rgb=color,
                shape_name=shape_name,
                alpha=alpha_this_slice,
                center_slice=center_slice,
            )

    # ------------------------------------------------------------
    # 9. Save 3D RGB TIFF
    # ------------------------------------------------------------
    tif_path = os.path.join(output_dir, f'{base}_query_init_3d_overlay.tif')

    tifffile.imwrite(
        tif_path,
        rgb,
        photometric='rgb',
        compression='zlib',
    )

    print(f'Saved 3D query-init overlay TIFF: {tif_path}')
    print(f'Overlay shape: {rgb.shape}, dtype={rgb.dtype}')

    # ------------------------------------------------------------
    # 10. Print summary
    # ------------------------------------------------------------
    print('\n=== Query initialization summary ===')
    print(f'image shape: D,H,W = {(D, H, W)}')
    print(f'topk visualized = {topk}')
    print(f'marker_radius = {marker_radius}')
    print('level marker shapes and colors:')
    for k, v in level_shapes.items():
        color = level_colors.get(
            k, fallback_level_colors[k % len(fallback_level_colors)]
        )
        print(f'  level {k}: shape={v}, color={color}')

    for meta in level_meta:
        print(
            f'level {meta["level"]}: '
            f'feature_shape={meta["shape"]}, '
            f'tokens={meta["num_tokens"]}, '
            f'global_token_range=[{meta["start"]}, {meta["end"]})'
        )

    print('\nTop initialized queries:')
    for r in rows[: min(20, len(rows))]:
        print(
            f'rank={r["rank"]:03d} | '
            f'prob={r["score_prob"]:.4f} | '
            f'logit={r["score_logit"]:.4f} | '
            f'level={r["level"]} | '
            f'shape={level_shapes.get(int(r["level"]), "filled_circle")} | '
            f'feat=({r["feature_z"]},{r["feature_y"]},{r["feature_x"]}) / '
            f'({r["feature_d"]},{r["feature_h"]},{r["feature_w"]}) | '
            f'image=({r["image_z"]:.1f},{r["image_y"]:.1f},{r["image_x"]:.1f})'
        )

    return {
        'tif_path': tif_path,
        'csv_path': csv_path,
        'rows': rows,
        'top_indices': top_indices.detach().cpu(),
        'top_scores': top_scores.detach().cpu(),
        'top_probs': top_probs.detach().cpu(),
        'level_meta': level_meta,
        'image_shape': (D, H, W),
        'level_shapes': level_shapes,
    }
