import copy
import glob
import math
import os
import random

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Instances
from scipy.ndimage import gaussian_filter


def get_real_3d_dataset_from_metadata(name):
    meta = MetadataCatalog.get(name)
    image_dir = meta.image_dir
    label_dir = meta.label_dir

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    dataset_dicts = []
    for idx, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        label_path = os.path.join(label_dir, filename)
        if not os.path.exists(label_path):
            continue
        record = {
            'image_path': image_path,
            'label_path': label_path,
            'file_name': filename,
            'image_id': idx,
        }
        dataset_dicts.append(record)
    # random.shuffle(dataset_dicts)
    # for new_idx, record in enumerate(dataset_dicts):
    #     record['image_id'] = new_idx
    return dataset_dicts


def register_real_dataset(cfg):
    """Register the real dataset with Detectron2."""
    name = 'embryo_3d_train'
    image_dir = cfg.DATASETS.IMAGE_DIR
    label_dir = cfg.DATASETS.LABEL_DIR

    MetadataCatalog.get(name).set(
        stuff_classes=['cell'],
        image_dir=image_dir,
        label_dir=label_dir,
    )

    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    DatasetCatalog.register(
        name, lambda: get_real_3d_dataset_from_metadata(name)
    )
    return name


class MaskFormer3DInstanceDatasetMapper:
    """
    A mapper for 3D instance segmentation data.
    Expects dataset dicts to contain "image" and "instances" tensors.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augmentations = []

        # augmentation params
        self.scale_range = getattr(cfg.INPUT, 'SCALE_RANGE', (0.75, 1.5))
        self.brightness_std = getattr(cfg.INPUT, 'BRIGHTNESS_STD', 0.08)
        self.contrast_log_range = getattr(
            cfg.INPUT, 'CONTRAST_LOG_RANGE', (-0.3, 0.3)
        )
        self.degradation_prob = getattr(cfg.INPUT, 'DEGRADATION_PROB', 0.5)
        self.poisson_peak_range = getattr(
            cfg.INPUT, 'POISSON_PEAK_RANGE', (20, 80)
        )
        self.blur_sigma_range = getattr(
            cfg.INPUT, 'BLUR_SIGMA_RANGE', (0.5, 1.5)
        )
        self.downsample_scale_range = getattr(
            cfg.INPUT, 'DOWNSAMPLE_SCALE_RANGE', (0.5, 0.8)
        )
        self.anisotropic_sigma_xy_range = getattr(
            cfg.INPUT, 'ANISO_SIGMA_XY_RANGE', (1.0, 2.5)
        )
        self.anisotropic_sigma_z_range = getattr(
            cfg.INPUT, 'ANISO_SIGMA_Z_RANGE', (0.0, 0.3)
        )
        self.scale_z = getattr(cfg.INPUT, 'SCALE_Z', False)

    def _random_rotate_xy_90(self, image, label):
        """
        Randomly rotate image and label by 0/90/180/270 degrees in XY plane.
        image: (1, D, H, W)
        label: (D, H, W)
        """
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[2, 3])
            label = torch.rot90(label, k=k, dims=[1, 2])
        return image, label

    def _resize_keep_shape_image(self, image, out_d, out_h, out_w):
        """
        Resize image with trilinear interpolation.
        image: (1, D, H, W)
        """
        return F.interpolate(
            image.unsqueeze(0),
            size=(out_d, out_h, out_w),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)

    def _resize_keep_shape_label(self, label, out_d, out_h, out_w):
        """
        Resize label with nearest interpolation.
        label: (D, H, W)
        """
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(
            label,
            size=(out_d, out_h, out_w),
            mode='nearest',
        )
        return label.squeeze(0).squeeze(0).long()

    def _center_crop_or_pad_image(
        self, image, target_d, target_h, target_w, pad_mode='replicate'
    ):
        """
        Center crop or pad image back to target shape.
        image: (1, D, H, W)

        pad_mode:
            - "replicate": repeat border values
            - "reflect": mirror padding
            - "constant": old zero-padding behavior
        """
        c, d, h, w = image.shape

        # Crop first
        if d > target_d:
            d0 = (d - target_d) // 2
            image = image[:, d0 : d0 + target_d, :, :]
        if h > target_h:
            h0 = (h - target_h) // 2
            image = image[:, :, h0 : h0 + target_h, :]
        if w > target_w:
            w0 = (w - target_w) // 2
            image = image[:, :, :, w0 : w0 + target_w]

        # Recompute shape after crop
        _, d, h, w = image.shape

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
            image = image.unsqueeze(0)  # (1, 1, D, H, W)

            if pad_mode in ['replicate', 'reflect']:
                image = F.pad(
                    image,
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
                image = F.pad(
                    image,
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

            image = image.squeeze(0)

        return image

    def _center_crop_or_pad_label(self, label, target_d, target_h, target_w):
        """
        Center crop or pad label back to target shape.
        label: (D, H, W)
        """
        d, h, w = label.shape

        if d > target_d:
            d0 = (d - target_d) // 2
            label = label[d0 : d0 + target_d, :, :]
        elif d < target_d:
            pad_front = (target_d - d) // 2
            pad_back = target_d - d - pad_front
            label = F.pad(label, (0, 0, 0, 0, pad_front, pad_back), value=0)

        d, h, w = label.shape
        if h > target_h:
            h0 = (h - target_h) // 2
            label = label[:, h0 : h0 + target_h, :]
        elif h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            label = F.pad(label, (0, 0, pad_top, pad_bottom, 0, 0), value=0)

        d, h, w = label.shape
        if w > target_w:
            w0 = (w - target_w) // 2
            label = label[:, :, w0 : w0 + target_w]
        elif w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            label = F.pad(label, (pad_left, pad_right, 0, 0, 0, 0), value=0)

        return label

    def _random_scale(
        self, image, label, scale_range=(0.75, 1.5), scale_z=False
    ):
        """
        Randomly scale image and label together.
        For anisotropic data, usually only scale XY.
        Output shape is restored to the original shape by center crop/pad.
        """
        _, d, h, w = image.shape
        scale = random.uniform(*scale_range)

        if scale_z:
            new_d = max(1, int(round(d * scale)))
        else:
            new_d = d
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        image_scaled = self._resize_keep_shape_image(
            image, new_d, new_h, new_w
        )
        label_scaled = self._resize_keep_shape_label(
            label, new_d, new_h, new_w
        )

        image_scaled = self._center_crop_or_pad_image(
            image_scaled, d, h, w, pad_mode='replicate'
        )
        label_scaled = self._center_crop_or_pad_label(label_scaled, d, h, w)

        return image_scaled, label_scaled

    def _apply_brightness_jitter(self, image, std=0.08):
        delta = torch.randn(1, device=image.device).item() * std
        return image + delta

    def _apply_contrast_jitter(self, image, log_range=(-0.3, 0.3)):
        log_scale = random.uniform(*log_range)
        scale = math.exp(log_scale)
        mean_val = image.mean()
        return (image - mean_val) * scale + mean_val

    def _apply_poisson_noise(self, image, peak_range=(20, 80)):
        peak = random.uniform(*peak_range)
        image_np = image.detach().cpu().numpy()
        image_np = np.clip(image_np, 0.0, 1.0)
        noisy = np.random.poisson(image_np * peak) / peak
        return torch.from_numpy(noisy).to(image.device).float()

    def _apply_gaussian_blur(self, image, sigma_range=(0.5, 1.5)):
        sigma = random.uniform(*sigma_range)
        image_np = image.detach().cpu().numpy()
        blurred = gaussian_filter(image_np, sigma=(0, sigma, sigma, sigma))
        return torch.from_numpy(blurred).to(image.device).float()

    def _apply_downsample_upsample(
        self, image, scale_range=(0.5, 0.8), scale_z=False
    ):
        _, d, h, w = image.shape
        scale = random.uniform(*scale_range)

        if scale_z:
            small_d = max(1, int(round(d * scale)))
        else:
            small_d = d
        small_h = max(1, int(round(h * scale)))
        small_w = max(1, int(round(w * scale)))

        image_small = F.interpolate(
            image.unsqueeze(0),
            size=(small_d, small_h, small_w),
            mode='trilinear',
            align_corners=False,
        )
        image_back = F.interpolate(
            image_small,
            size=(d, h, w),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)

        return image_back

    def _apply_anisotropic_blur(
        self,
        image,
        sigma_xy_range=(1.0, 2.5),
        sigma_z_range=(0.0, 0.3),
    ):
        sigma_xy = random.uniform(*sigma_xy_range)
        sigma_z = random.uniform(*sigma_z_range)
        image_np = image.detach().cpu().numpy()
        blurred = gaussian_filter(
            image_np, sigma=(0, sigma_z, sigma_xy, sigma_xy)
        )
        return torch.from_numpy(blurred).to(image.device).float()

    def _apply_random_degradation(self, image):
        """
        Apply one random image-only degradation with a given probability.
        """
        if random.random() > self.degradation_prob:
            return image

        op = random.choice(
            [
                'poisson',
                'gaussian_blur',
                'downsample',
                'anisotropic_blur',
            ]
        )

        if op == 'poisson':
            image = self._apply_poisson_noise(
                image, peak_range=self.poisson_peak_range
            )
        elif op == 'gaussian_blur':
            image = self._apply_gaussian_blur(
                image, sigma_range=self.blur_sigma_range
            )
        elif op == 'downsample':
            image = self._apply_downsample_upsample(
                image,
                scale_range=self.downsample_scale_range,
                scale_z=self.scale_z,
            )
        elif op == 'anisotropic_blur':
            image = self._apply_anisotropic_blur(
                image,
                sigma_xy_range=self.anisotropic_sigma_xy_range,
                sigma_z_range=self.anisotropic_sigma_z_range,
            )

        return image

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_path = dataset_dict.pop('image_path')
        label_path = dataset_dict.pop('label_path')

        # Read image as float32 directly to preserve intensity information
        image = tifffile.imread(image_path)
        image = np.array(image, dtype=np.float32, copy=True)

        # # Per-volume min-max normalization to [0, 1]
        # min_val = image.min()
        # max_val = image.max()
        # image = (image - min_val) / (max_val - min_val + 1e-8)
        # image = np.array(image, dtype=np.uint8, copy=True)
        # image = image.astype(np.float32) / 255.0
        # (D, H, W) -> (1, D, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()

        # Read label
        label = tifffile.imread(label_path)
        label = np.array(label, dtype=np.int64, copy=True)
        label = torch.from_numpy(label).long()

        if self.is_train:
            # Random flips along Z / Y / X
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])
                label = torch.flip(label, dims=[0])

            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[1])

            if random.random() > 0.5:
                image = torch.flip(image, dims=[3])
                label = torch.flip(label, dims=[2])

            # Random 90-degree rotation in XY plane
            if random.random() > 0.5:
                image, label = self._random_rotate_xy_90(image, label)

            # Random scaling to simulate cell size variation
            # Geometric transform must be applied to both image and label
            # if random.random() > 0.5:
            #     image, label = self._random_scale(
            #         image,
            #         label,
            #         scale_range=self.scale_range,
            #         scale_z=self.scale_z,
            #     )

            # Brightness jitter (image only)
            if random.random() > 0.5:
                image = self._apply_brightness_jitter(
                    image,
                    std=self.brightness_std,
                )

            # Contrast jitter (image only)
            if random.random() > 0.5:
                image = self._apply_contrast_jitter(
                    image,
                    log_range=self.contrast_log_range,
                )

            # Random image degradation (image only)
            image = self._apply_random_degradation(image)

            # Keep image in valid range
            image = torch.clamp(image, 0.0, 1.0)

        # Build instance masks from label ids
        unique_ids = torch.unique(label)
        instance_ids = unique_ids[unique_ids != 0]

        masks = []
        classes = []
        for inst_id in instance_ids:
            mask = label == inst_id
            if mask.any():
                masks.append(mask.bool())
                classes.append(0)

        instances = Instances((label.shape[-2], label.shape[-1]))
        object.__setattr__(instances, 'image_depth', label.shape[0])

        if len(masks) > 0:
            instances.gt_masks = torch.stack(masks, dim=0)  # (N, D, H, W)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        else:
            instances.gt_masks = torch.zeros(
                (0, *label.shape), dtype=torch.bool
            )
            instances.gt_classes = torch.zeros((0,), dtype=torch.int64)

        dataset_dict['image'] = image.float()
        dataset_dict['instances'] = instances
        dataset_dict['height'] = image.shape[-2]
        dataset_dict['width'] = image.shape[-1]

        return dataset_dict
