import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, Union

import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShapeSpec:
    channels: int = None
    stride: int = None


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Lightweight replacement for fvcore.nn.c2_xavier_fill.

    This only affects initialization before checkpoint loading.
    It does not change checkpoint key names.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def get_norm(norm, out_channels: int):
    """
    Minimal Detectron2-free get_norm replacement for 3D modules.
    """
    if norm is None or norm == '':
        return None

    if callable(norm):
        return norm(out_channels)

    norm = str(norm)

    if norm == 'GN':
        # Keep GroupNorm behavior close to Detectron2.
        num_groups = 32
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        return nn.GroupNorm(num_groups, out_channels)

    if norm in {'BN', 'SyncBN'}:
        return nn.BatchNorm3d(out_channels)

    raise ValueError(f'Unsupported norm type for 3D pixel decoder: {norm}')


def build_pixel_decoder(cfg, input_shape):
    """
    Detectron2-free pixel decoder builder.
    """
    name = getattr(cfg.MODEL.SEM_SEG_HEAD, 'PIXEL_DECODER_NAME', None)
    if name is None:
        name = getattr(cfg.MODEL.SEM_SEG_HEAD, 'NAME', None)

    if name != 'BasePixelDecoder3D':
        raise ValueError(f'Unsupported pixel decoder in fpn3d_win.py: {name}')

    kwargs = BasePixelDecoder3D.from_config(cfg, input_shape)
    model = BasePixelDecoder3D(**kwargs)

    forward_features = getattr(model, 'forward_features', None)
    if not callable(forward_features):
        raise ValueError(
            f'Pixel decoder {name} must implement forward_features().'
        )
    return model


class Conv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# 3D-adapted FPN decoder
class BasePixelDecoder3D(nn.Module):
    """
    3D version of the BasePixelDecoder.
    Takes multi-scale features from a 3D backbone (e.g., MAE3D) and produces:
      - mask_features: high-resolution feature map (used for final mask prediction)
      - multi_scale_features: a list of feature maps at selected scales (for transformer decoder)
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Union[str, Callable] | None = None,
        num_classes: int,
    ):
        """
        Args:
            input_shape: dict mapping feature names to ShapeSpec (channels and stride).
            conv_dim: number of channels for intermediate convolution layers.
            mask_dim: number of channels for the final mask feature.
            norm: normalization specification (e.g., "GN", "BN", or a callable).
                  Note: For 3D data, the norm must support 5D inputs.
                  It is recommended to use "GN" (GroupNorm) as it is dimension-agnostic.
        """
        super().__init__()
        self.num_classes = num_classes
        # Sort features by stride (lowest stride = highest resolution last)
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ''
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # Top level (lowest resolution) has no lateral connection
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv3d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(output_conv)
                self.add_module(f'layer_{idx + 1}', output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv3d(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv3d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                c2_xavier_fill(lateral_conv)
                c2_xavier_fill(output_conv)
                self.add_module(f'adapter_{idx + 1}', lateral_conv)
                self.add_module(f'layer_{idx + 1}', output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

        # Reverse lists for top-down order (from low resolution to high resolution)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv3d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret['input_shape'] = {
            k: v
            for k, v in input_shape.items()
            if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret['conv_dim'] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret['mask_dim'] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret['norm'] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret['num_classes'] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        return ret

    def forward_features(self, features):
        """
        Args:
            features (dict): multi-scale features from backbone.
                Each value is a 5D tensor of shape (N, C, D, H, W).

        Returns:
            mask_features (Tensor): final mask feature map of shape (N, mask_dim, D, H, W).
            transformer_features (None): placeholder to match the interface of
                TransformerEncoderPixelDecoder (returns None).
            multi_scale_features (list[Tensor]): list of feature maps at selected scales,
                each of shape (N, conv_dim, D_i, H_i, W_i).
        """
        multi_scale_features = []
        num_cur_levels = 0

        # Process features in top-down order (from lowest resolution to highest)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]  # (N, C, D, H, W)
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]

            if lateral_conv is None:
                # Top level (lowest resolution) – no lateral connection
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Upsample the previous higher-level feature `y` to current resolution
                # and add element-wise.
                y = cur_fpn + F.interpolate(
                    y,
                    size=cur_fpn.shape[-3:],  # use depth, height, width
                    mode='trilinear',  # 3D interpolation
                    align_corners=False,
                )
                y = output_conv(y)

            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1

        return self.mask_features(y), None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            'Calling forward() may cause unpredicted behavior of PixelDecoder module.'
        )
        return self.forward_features(features)
