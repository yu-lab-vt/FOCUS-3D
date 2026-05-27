# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""

import math

import torch
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    """
    3D sinusoidal positional encoding.
    Generates embeddings for depth, height, and width dimensions.
    """

    def __init__(
        self, d_model, temperature=10000, normalize=False, scale=None
    ):
        """
        Args:
            num_pos_feats: number of position features (must be divisible by 3)
            temperature: temperature for the sinusoidal frequencies
            normalize: whether to normalize coordinates to [-scale/2, scale/2]
            scale: scaling factor for normalization (default 2π)
        """
        super().__init__()
        assert d_model % 6 == 0, 'd_model must be divisible by 6'
        self.num_pos_feats = (
            d_model // 3
        )  # features per dimension (depth, height, width)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input tensor of shape (N, C, D, H, W)
            mask: optional mask tensor of shape (N, D, H, W) indicating valid positions (True = masked)
        Returns:
            pos: positional encoding of shape (N, num_pos_feats*2, D, H, W)
        """
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3), x.size(4)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask  # (N, D, H, W)

        # Cumulative sums along each dimension
        d_embed = not_mask.cumsum(1, dtype=torch.float32)  # (N, D, H, W)
        h_embed = not_mask.cumsum(2, dtype=torch.float32)  # (N, D, H, W)
        w_embed = not_mask.cumsum(3, dtype=torch.float32)  # (N, D, H, W)

        if self.normalize:
            eps = 1e-6
            d_embed = d_embed / (d_embed[:, -1:, :, :] + eps) * self.scale
            h_embed = h_embed / (h_embed[:, :, -1:, :] + eps) * self.scale
            w_embed = w_embed / (w_embed[:, :, :, -1:] + eps) * self.scale

        # Create frequency bands
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (
            2 * (dim_t // 2) / self.num_pos_feats
        )  # (num_pos_feats)

        # Compute embeddings for each dimension
        pos_d = (
            d_embed[:, :, :, :, None] / dim_t
        )  # (N, D, H, W, num_pos_feats)
        pos_h = h_embed[:, :, :, :, None] / dim_t
        pos_w = w_embed[:, :, :, :, None] / dim_t

        # Apply sin/cos
        pos_d = torch.stack(
            (pos_d[..., 0::2].sin(), pos_d[..., 1::2].cos()), dim=-1
        ).flatten(-2)  # (N, D, H, W, num_pos_feats)
        pos_h = torch.stack(
            (pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_w = torch.stack(
            (pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        # Concatenate along channel dimension
        pos = torch.cat(
            (pos_d, pos_h, pos_w), dim=-1
        )  # (N, D, H, W, num_pos_feats*3)
        pos = pos.permute(
            0, 4, 1, 2, 3
        ).contiguous()  # (N, num_pos_feats*3, D, H, W)

        return pos

    def __repr__(self, _repr_indent=4):
        head = 'Positional encoding ' + self.__class__.__name__
        body = [
            f'num_pos_feats: {self.num_pos_feats}',
            f'temperature: {self.temperature}',
            f'normalize: {self.normalize}',
            f'scale: {self.scale}',
        ]
        lines = [head] + [' ' * _repr_indent + line for line in body]
        return '\n'.join(lines)
