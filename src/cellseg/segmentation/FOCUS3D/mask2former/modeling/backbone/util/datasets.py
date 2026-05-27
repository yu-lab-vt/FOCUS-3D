# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# Add this after existing imports
try:
    from monai.data import ImageDataset
    from monai.transforms import (
        CenterSpatialCrop,
        Compose,
        EnsureChannelFirst,
        LoadImage,
        RandFlip,
        RandRotate,
        RandZoom,
        ScaleIntensity,
    )

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print('Warning: monai not installed, 3D dataset will not work.')

import glob

# class InstanceSegDataset3D(Dataset):
#     """
#     3D instance segmentation dataset.
#     Returns: image (tensor), instance_mask (tensor), distance_map (tensor, optional)
#     """
#     def __init__(self, images_dir, labels_dir, dist_dir=None, target_size=(64,128,128),
#                  augment=False, normalize=True, compute_dist_on_fly=False):
#         self.images_dir = images_dir
#         self.labels_dir = labels_dir
#         self.dist_dir = dist_dir
#         self.target_size = target_size
#         self.augment = augment
#         self.normalize = normalize
#         self.compute_dist_on_fly = compute_dist_on_fly
#         self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
#         self.label_paths = sorted(glob.glob(os.path.join(labels_dir, '*.tif')))
#         assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"
#         if dist_dir is not None and not compute_dist_on_fly:
#             self.dist_paths = sorted(glob.glob(os.path.join(dist_dir, '*.tif')))
#             assert len(self.dist_paths) == len(self.image_paths), "Mismatch for distance maps"
#     def __len__(self):
#         return len(self.image_paths)
#     def _load_tif(self, path):
#         img = tifffile.imread(path)
#         img = np.array(img, dtype=np.uint8, copy=True)
#         img = img.astype(np.float32)/255.0
#         # Add channel dimension
#         if img.ndim == 3:
#             img = img[np.newaxis, ...]  # (1, D, H, W)
#         return img
#     def _load_label(self, path):
#         img = tifffile.imread(path)
#         img = np.array(img, dtype=np.float32, copy=True)
#         # Add channel dimension
#         if img.ndim == 3:
#             img = img[np.newaxis, ...]  # (1, D, H, W)
#         return img
#     def _compute_distance_map(self,label):
#         """
#         Compute distance map for foreground only, with values scaled to [0.8, 1] (background = 0).
#         For each foreground pixel, value = 0.8 + 0.2 * normalized distance to nearest background.
#         Args:
#             label: 3D numpy array with instance IDs (0 background, >0 instances)
#         Returns:
#             dist_map: 3D numpy array, foreground pixels in [0.8, 1], background 0.
#         """
#         # Binary foreground mask
#         fg = (label > 0).astype(np.uint8)
#         # Euclidean distance transform for foreground pixels
#         dist_inside = distance_transform_edt(fg)  # shape (D, H, W)
#         # Normalize foreground distances to [0,1] using the maximum within foreground
#         fg_indices = fg > 0
#         if np.any(fg_indices):
#             max_dist_fg = dist_inside[fg_indices].max()
#             if max_dist_fg > 0:
#                 dist_norm = dist_inside / max_dist_fg  # [0,1]
#             else:
#                 # All foreground distances are zero (e.g., single pixel instances)
#                 dist_norm = np.zeros_like(dist_inside)
#         else:
#             # No foreground at all
#             return np.zeros_like(label, dtype=np.float32)
#         # Scale to [0.5, 1] for foreground
#         dist_tanh = 0.5 + 0.5 * np.tanh(5 * dist_norm)
#         # Apply mask to keep only foreground values
#         dist_map = dist_tanh * fg
#         return dist_map.astype(np.float32)
#     def __getitem__(self, idx):
#         # Load image
#         img = self._load_tif(self.image_paths[idx])
#         # Load label (no channel dimension)
#         label = self._load_label(self.label_paths[idx])[0]  # (D, H, W)
#         # Center crop to target size
#         _, D, H, W = img.shape
#         tD, tH, tW = self.target_size
#         if D < tD or H < tH or W < tW:
#             raise RuntimeError(f"Volume too small: {self.image_paths[idx]}")
#         start_d = (D - tD) // 2
#         start_h = (H - tH) // 2
#         start_w = (W - tW) // 2
#         img = img[:, start_d:start_d+tD, start_h:start_h+tH, start_w:start_w+tW]
#         label = label[start_d:start_d+tD, start_h:start_h+tH, start_w:start_w+tW]
#         # Convert to tensors
#         img = torch.from_numpy(img).float()
#         label = torch.from_numpy(label).long()  # instance IDs as integers
#         # --- Online augmentation (applied to both image and label) ---
#         if self.augment:
#             # Random flips along depth, height, width
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[1])
#                 label = torch.flip(label, dims=[0])
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[2])
#                 label = torch.flip(label, dims=[1])
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[3])
#                 label = torch.flip(label, dims=[2])
#             # Random intensity scaling (image only)
#             if random.random() > 0.5:
#                 scale = random.uniform(0.8, 1.2)
#                 img = img * scale
#                 img = torch.clamp(img, 0, 1)  # if normalized
#         # Normalize image to [0,1]
#         if self.normalize:
#             min_val = img.min()
#             max_val = img.max()
#             if max_val > min_val + 1e-6:
#                 img = (img - min_val) / (max_val - min_val)
#         # Get distance map
#         if self.compute_dist_on_fly:
#             dist = self._compute_distance_map(label.numpy())
#             dist = torch.from_numpy(dist).float().unsqueeze(0)  # add channel dim
#         elif self.dist_dir is not None:
#             dist = self._load_label(self.dist_paths[idx])  # precomputed
#         else:
#             dist = torch.zeros_like(img)  # dummy
#         return img, label, dist
# --------------------------------------------------------------------
# Load real data from folder
# --------------------------------------------------------------------
# class RealVolumeDataset(Dataset):
#     def __init__(self, root_dir, target_size=(128, 128, 64), normalize=True, augment=True):
#         """
#         Args:
#             root_dir: path to folder containing .tif files
#             target_size: tuple (D, H, W) for final crop size
#             normalize: if True, scale intensities to [0, 1] (assuming 16-bit)
#         """
#         self.root_dir = root_dir
#         self.target_size = target_size
#         self.normalize = normalize
#         self.augment = augment
#         self.file_list = glob.glob(os.path.join(root_dir, '*.tif')) + \
#                          glob.glob(os.path.join(root_dir, '*.tiff'))
#         if len(self.file_list) == 0:
#             raise RuntimeError(f"No .tif files found in {root_dir}")
#         print(f"Found {len(self.file_list)} volume files.")
#     def __len__(self):
#         return len(self.file_list)
#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         # Read with tifffile
#         img = tifffile.imread(img_path)  # may return numpy array or array-like
#         # Ensure it's a standard numpy array
#         img = np.array(img, dtype=np.uint8, copy=True)  # <-- this converts to standard np.ndarray
#         img = img.astype(np.float32) / 255.0
#         # Add channel dimension if needed
#         if img.ndim == 3:
#             img = img[np.newaxis, ...]  # (1, D, H, W)
#         elif img.ndim != 4:
#             raise ValueError(f"Unexpected image dimensions: {img.ndim} from {img_path}")
#         # Normalize to [0,1] (min-max per volume)
#         if self.normalize:
#             min_val = img.min()
#             max_val = img.max()
#             img = (img - min_val) / (max_val - min_val + 1e-8)
#         img = torch.from_numpy(img).float()
#         if self.augment:
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[1])
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[2])
#             if random.random() > 0.5:
#                 img = torch.flip(img, dims=[3])
#             if random.random() > 0.5:
#                 scale = random.uniform(0.9, 1.1)
#                 img = img * scale
#             if random.random() > 0.5:
#                 noise = torch.randn_like(img) * 0.01
#                 img = img + noise
#                 img = torch.clamp(img, 0, 1)
#         img = img.float()
#         return img, 0
# --------------------------------------------------------------------
# Load real data from folder
# --------------------------------------------------------------------
import math
import os
import random

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class RealVolumeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        target_size=(32, 96, 96),
        normalize=True,
        augment=True,
        scale_range=(0.75, 1.5),  # XY scale jitter range
        brightness_std=0.08,  # Gaussian brightness jitter std
        contrast_log_range=(-0.3, 0.3),  # log-contrast range, scale = exp(u)
        degradation_prob=0.5,  # probability to apply one degradation op
        poisson_peak_range=(20, 80),
        blur_sigma_range=(0.5, 1.5),
        downsample_scale_range=(0.5, 0.8),
        anisotropic_sigma_xy_range=(1.0, 2.5),
        anisotropic_sigma_z_range=(0.0, 0.3),
        scale_z=False,  # for anisotropic data, usually keep Z unchanged
    ):
        """
        Args:
            root_dir: path to folder containing .tif files
            target_size: kept for compatibility; not used here for cropping
            normalize: if True, normalize intensities to [0, 1] using min-max per volume
            augment: whether to apply data augmentation
            scale_range: random XY scaling range
            brightness_std: std of additive Gaussian brightness jitter
            contrast_log_range: log contrast range; actual scale = exp(uniform(a, b))
            degradation_prob: probability to apply one random degradation op
            poisson_peak_range: peak range for Poisson noise simulation
            blur_sigma_range: sigma range for isotropic Gaussian blur
            downsample_scale_range: XY downsample factor range
            anisotropic_sigma_xy_range: XY sigma range for anisotropic blur
            anisotropic_sigma_z_range: Z sigma range for anisotropic blur
            scale_z: whether random scaling also changes Z dimension
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        self.scale_range = scale_range
        self.brightness_std = brightness_std
        self.contrast_log_range = contrast_log_range
        self.degradation_prob = degradation_prob
        self.poisson_peak_range = poisson_peak_range
        self.blur_sigma_range = blur_sigma_range
        self.downsample_scale_range = downsample_scale_range
        self.anisotropic_sigma_xy_range = anisotropic_sigma_xy_range
        self.anisotropic_sigma_z_range = anisotropic_sigma_z_range
        self.scale_z = scale_z

        self.file_list = glob.glob(
            os.path.join(root_dir, '*.tif')
        ) + glob.glob(os.path.join(root_dir, '*.tiff'))
        if len(self.file_list) == 0:
            raise RuntimeError(f'No .tif files found in {root_dir}')
        print(f'Found {len(self.file_list)} volume files.')

    def __len__(self):
        return len(self.file_list)

    def _random_rotate_xy_90(self, img):
        """
        Randomly rotate the volume by 0/90/180/270 degrees in the XY plane.
        img: torch tensor of shape (C, D, H, W)
        """
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k=k, dims=[2, 3])
        return img

    def _resize_keep_shape(self, img, out_d, out_h, out_w):
        """
        Resize image to the given shape using trilinear interpolation.
        img: torch tensor of shape (C, D, H, W)
        """
        img = F.interpolate(
            img.unsqueeze(0),
            size=(out_d, out_h, out_w),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)
        return img

    def _center_crop_or_pad(
        self, img, target_d, target_h, target_w, pad_mode='replicate'
    ):
        """
        Center crop or pad image back to target shape.
        img: torch tensor of shape (C, D, H, W)

        pad_mode:
            - "replicate": repeat border values
            - "reflect": mirror padding
            - "constant": zero padding (old behavior)
        """
        c, d, h, w = img.shape

        # Crop first if needed
        if d > target_d:
            d0 = (d - target_d) // 2
            img = img[:, d0 : d0 + target_d, :, :]
        if h > target_h:
            h0 = (h - target_h) // 2
            img = img[:, :, h0 : h0 + target_h, :]
        if w > target_w:
            w0 = (w - target_w) // 2
            img = img[:, :, :, w0 : w0 + target_w]

        # Recompute shape after crop
        _, d, h, w = img.shape

        pad_front = max(0, (target_d - d) // 2)
        pad_back = max(0, target_d - d - pad_front)
        pad_top = max(0, (target_h - h) // 2)
        pad_bottom = max(0, target_h - h - pad_top)
        pad_left = max(0, (target_w - w) // 2)
        pad_right = max(0, target_w - w - pad_left)

        if (
            pad_front > 0
            or pad_back > 0
            or pad_top > 0
            or pad_bottom > 0
            or pad_left > 0
            or pad_right > 0
        ):
            # F.pad with replicate/reflect works more reliably on 5D tensors
            img = img.unsqueeze(0)  # (1, C, D, H, W)

            if pad_mode in ['replicate', 'reflect']:
                img = F.pad(
                    img,
                    (
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                        pad_front,
                        pad_back,
                    ),
                    mode=pad_mode,
                )
            else:
                img = F.pad(
                    img,
                    (
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                        pad_front,
                        pad_back,
                    ),
                    mode='constant',
                    value=0.0,
                )

            img = img.squeeze(0)

        return img

    def _random_scale(self, img, scale_range=(0.75, 1.5), scale_z=False):
        """
        Randomly scale the volume. For anisotropic data, usually only scale XY.
        The output shape is restored to the original shape by center crop/pad.
        img: torch tensor of shape (C, D, H, W)
        """
        c, d, h, w = img.shape
        scale = random.uniform(*scale_range)

        if scale_z:
            new_d = max(1, int(round(d * scale)))
        else:
            new_d = d
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        img_scaled = self._resize_keep_shape(img, new_d, new_h, new_w)
        img_scaled = self._center_crop_or_pad(
            img_scaled, d, h, w, pad_mode='replicate'
        )
        return img_scaled

    def _apply_brightness_jitter(self, img, std=0.08):
        """
        Add Gaussian brightness jitter to the whole volume.
        """
        delta = torch.randn(1, device=img.device).item() * std
        img = img + delta
        return img

    def _apply_contrast_jitter(self, img, log_range=(-0.3, 0.3)):
        """
        Apply multiplicative contrast scaling around the mean intensity.
        """
        log_scale = random.uniform(*log_range)
        scale = math.exp(log_scale)
        mean_val = img.mean()
        img = (img - mean_val) * scale + mean_val
        return img

    def _apply_poisson_noise(self, img, peak_range=(20, 80)):
        """
        Apply Poisson noise to a [0, 1] image.
        """
        peak = random.uniform(*peak_range)
        img_np = img.detach().cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        noisy = np.random.poisson(img_np * peak) / peak
        return torch.from_numpy(noisy).to(img.device).float()

    def _apply_gaussian_blur(self, img, sigma_range=(0.5, 1.5)):
        """
        Apply isotropic Gaussian blur in 3D.
        """
        sigma = random.uniform(*sigma_range)
        img_np = img.detach().cpu().numpy()
        # Do not blur the channel dimension
        blurred = gaussian_filter(img_np, sigma=(0, sigma, sigma, sigma))
        return torch.from_numpy(blurred).to(img.device).float()

    def _apply_downsample_upsample(
        self, img, scale_range=(0.5, 0.8), scale_z=False
    ):
        """
        Downsample and then upsample to simulate resolution loss.
        For anisotropic data, usually only downsample XY.
        """
        c, d, h, w = img.shape
        scale = random.uniform(*scale_range)

        if scale_z:
            small_d = max(1, int(round(d * scale)))
        else:
            small_d = d
        small_h = max(1, int(round(h * scale)))
        small_w = max(1, int(round(w * scale)))

        img_small = F.interpolate(
            img.unsqueeze(0),
            size=(small_d, small_h, small_w),
            mode='trilinear',
            align_corners=False,
        )
        img_back = F.interpolate(
            img_small,
            size=(d, h, w),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)

        return img_back

    def _apply_anisotropic_blur(
        self,
        img,
        sigma_xy_range=(1.0, 2.5),
        sigma_z_range=(0.0, 0.3),
    ):
        """
        Apply anisotropic Gaussian blur with stronger blur in XY than in Z.
        """
        sigma_xy = random.uniform(*sigma_xy_range)
        sigma_z = random.uniform(*sigma_z_range)
        img_np = img.detach().cpu().numpy()
        blurred = gaussian_filter(
            img_np, sigma=(0, sigma_z, sigma_xy, sigma_xy)
        )
        return torch.from_numpy(blurred).to(img.device).float()

    def _apply_random_degradation(self, img):
        """
        With a given probability, apply one randomly selected degradation.
        """
        if random.random() > self.degradation_prob:
            return img

        op = random.choice(
            [
                'poisson',
                'gaussian_blur',
                'downsample',
                'anisotropic_blur',
            ]
        )

        if op == 'poisson':
            img = self._apply_poisson_noise(
                img, peak_range=self.poisson_peak_range
            )
        elif op == 'gaussian_blur':
            img = self._apply_gaussian_blur(
                img, sigma_range=self.blur_sigma_range
            )
        elif op == 'downsample':
            img = self._apply_downsample_upsample(
                img,
                scale_range=self.downsample_scale_range,
                scale_z=self.scale_z,
            )
        elif op == 'anisotropic_blur':
            img = self._apply_anisotropic_blur(
                img,
                sigma_xy_range=self.anisotropic_sigma_xy_range,
                sigma_z_range=self.anisotropic_sigma_z_range,
            )

        return img

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        # Read volume with tifffile
        img = tifffile.imread(img_path)

        # Convert to float32 directly to preserve intensity information
        img = np.array(img, dtype=np.float32, copy=True)

        # Add channel dimension if needed
        if img.ndim == 3:
            img = img[np.newaxis, ...]  # (1, D, H, W)
        elif img.ndim != 4:
            raise ValueError(
                f'Unexpected image dimensions: {img.ndim} from {img_path}'
            )

        # Normalize to [0, 1] using per-volume min-max
        if self.normalize:
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val + 1e-8)

        img = torch.from_numpy(img).float()

        if self.augment:
            # Random flips along Z / Y / X
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[3])

            # Random 90-degree rotation in XY plane
            if random.random() > 0.5:
                img = self._random_rotate_xy_90(img)

            # Random scaling to simulate cell size variation
            if random.random() > 0.5:
                img = self._random_scale(
                    img,
                    scale_range=self.scale_range,
                    scale_z=self.scale_z,
                )

            # Brightness jitter
            if random.random() > 0.5:
                img = self._apply_brightness_jitter(
                    img,
                    std=self.brightness_std,
                )

            # Contrast jitter
            if random.random() > 0.5:
                img = self._apply_contrast_jitter(
                    img,
                    log_range=self.contrast_log_range,
                )

            # Random degradation
            img = self._apply_random_degradation(img)

            # Clamp to valid range
            img = torch.clamp(img, 0.0, 1.0)

        img = img.float()
        return img, 0
