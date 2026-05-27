# Copyright (c) Facebook, Inc. and its affiliates.
# PyTorch prototype implementation of true 3D multi-scale deformable attention.
# This version is designed for correctness and easy integration, not for maximum speed.

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.init import constant_, normal_, xavier_uniform_

from ..transformer_decoder.position_encoding_3d import PositionEmbeddingSine3D
from ..transformer_decoder.transformer_3d import (
    _get_activation_fn,
    _get_clones,
)


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


class MSDeformAttn3D_PyTorch(nn.Module):
    """
    A pure PyTorch prototype of true 3D multi-scale deformable attention.

    Input/Output semantics are intentionally kept close to the original 2D MSDeformAttn:
        query:              [B, Len_q, C]
        reference_points:   [B, Len_q, n_levels, 3] in normalized coordinates (x, y, z), each in [0, 1]
        input_flatten:      [B, Len_in, C]
        input_spatial_shapes:
                            [n_levels, 3], each row is (D_l, H_l, W_l)
        input_level_start_index:
                            [n_levels]
        input_padding_mask: [B, Len_in], True means padded (optional)

    This implementation uses grid_sample to perform trilinear sampling for each level.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f'd_model ({d_model}) must be divisible by n_heads ({n_heads}).'
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_per_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 3
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)

        # Initialize offsets in a roughly spherical 3D pattern.
        # This is only an initialization heuristic for the prototype version.
        grid_init = torch.zeros(self.n_heads, self.n_levels, self.n_points, 3)

        for h in range(self.n_heads):
            for l in range(self.n_levels):
                for p in range(self.n_points):
                    theta = 2.0 * np.pi * (p / max(self.n_points, 1))
                    z = -1.0 + 2.0 * (p + 0.5) / max(self.n_points, 1)
                    r_xy = max(1e-6, float(np.sqrt(max(0.0, 1.0 - z * z))))
                    x = r_xy * np.cos(theta)
                    y = r_xy * np.sin(theta)
                    vec = torch.tensor([x, y, z], dtype=torch.float32)
                    vec = vec / (vec.abs().max() + 1e-6)
                    grid_init[h, l, p] = vec * (p + 1)

        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        Args:
            query: [B, Len_q, C]
            reference_points: [B, Len_q, n_levels, 3], normalized in [0, 1], order (x, y, z)
            input_flatten: [B, Len_in, C]
            input_spatial_shapes: [n_levels, 3], rows are (D, H, W)
            input_level_start_index: [n_levels]
            input_padding_mask: [B, Len_in], True for padding
        Returns:
            output: [B, Len_q, C]
        """
        B, Len_q, _ = query.shape
        B_in, Len_in, _ = input_flatten.shape
        assert B_in == B, (
            'Batch size mismatch between query and input_flatten.'
        )
        assert input_spatial_shapes.shape[0] == self.n_levels, (
            'n_levels mismatch.'
        )
        assert reference_points.shape[:3] == (B, Len_q, self.n_levels), (
            'reference_points shape mismatch.'
        )

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)

        value = value.view(B, Len_in, self.n_heads, self.d_per_head)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 3
        )
        # attention_weights = self.attention_weights(query).view(
        #     B, Len_q, self.n_heads, self.n_levels, self.n_points
        # )
        # attention_weights = F.softmax(attention_weights, dim=-1)   # zqh0407

        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        output = query.new_zeros(B, Len_q, self.n_heads, self.d_per_head)

        for lvl in range(self.n_levels):
            D_l, H_l, W_l = input_spatial_shapes[lvl].tolist()
            start = input_level_start_index[lvl].item()
            end = start + D_l * H_l * W_l

            value_l = value[:, start:end]  # [B, D*H*W, n_heads, head_dim]
            value_l = value_l.view(
                B, D_l, H_l, W_l, self.n_heads, self.d_per_head
            )
            value_l = value_l.permute(
                0, 4, 5, 1, 2, 3
            ).contiguous()  # [B, n_heads, head_dim, D, H, W]
            value_l = value_l.view(
                B * self.n_heads, self.d_per_head, D_l, H_l, W_l
            )

            # reference_points: normalized (x, y, z) in [0, 1]
            ref_l = reference_points[:, :, lvl]  # [B, Len_q, 3]

            # Normalize offsets by spatial size so that offsets are scale-aware.
            normalizer = query.new_tensor([W_l, H_l, D_l]).view(1, 1, 1, 1, 3)
            sampling_locations = (
                ref_l[:, :, None, None, :]
                + sampling_offsets[:, :, :, lvl] / normalizer
            )

            # Convert to grid_sample format:
            # grid_sample for 5D expects grid [..., 3] in order (x, y, z), normalized to [-1, 1].
            grid = (
                2.0 * sampling_locations - 1.0
            )  # [B, Len_q, n_heads, n_points, 3]
            grid = grid.permute(
                0, 2, 1, 3, 4
            ).contiguous()  # [B, n_heads, Len_q, n_points, 3]
            grid = grid.view(B * self.n_heads, Len_q, self.n_points, 1, 3)

            sampled = F.grid_sample(
                value_l,
                grid,
                mode='bilinear',  # trilinear for 5D input
                padding_mode='zeros',
                align_corners=False,
            )
            # sampled: [B*n_heads, head_dim, Len_q, n_points, 1]
            sampled = sampled.squeeze(
                -1
            )  # [B*n_heads, head_dim, Len_q, n_points]
            sampled = sampled.view(
                B, self.n_heads, self.d_per_head, Len_q, self.n_points
            )
            sampled = sampled.permute(
                0, 3, 1, 4, 2
            ).contiguous()  # [B, Len_q, n_heads, n_points, head_dim]

            attn = attention_weights[:, :, :, lvl].unsqueeze(
                -1
            )  # [B, Len_q, n_heads, n_points, 1]
            output = output + (sampled * attn).sum(dim=3)

        output = output.view(B, Len_q, self.d_model)
        output = self.output_proj(output)
        return output


class MSDeformAttnTransformerEncoderOnly3D(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer3D(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder3D(
            encoder_layer, num_encoder_layers
        )
        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn3D_PyTorch):
                m._reset_parameters()
        normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        """
        Args:
            mask: [B, D, H, W], True means padded
        Returns:
            valid_ratio: [B, 3] in order (w_ratio, h_ratio, d_ratio)
        """
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W

        return torch.stack(
            [valid_ratio_w, valid_ratio_h, valid_ratio_d], dim=-1
        )

    def forward(self, srcs, pos_embeds):
        """
        Args:
            srcs: list of 3D tensors, each [B, C, D, H, W]
            pos_embeds: list of 3D tensors, each [B, C, D, H, W]
        Returns:
            memory: [B, sum(D_l*H_l*W_l), C]
            spatial_shapes: [n_levels, 3]
            level_start_index: [n_levels]
        """
        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3), x.size(4)),
                device=x.device,
                dtype=torch.bool,
            )
            for x in srcs
        ]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(
            zip(srcs, masks, pos_embeds)
        ):
            bs, c, d, h, w = src.shape
            spatial_shapes.append((d, h, w))

            src = src.flatten(2).transpose(1, 2)  # [B, D*H*W, C]
            mask = mask.flatten(1)  # [B, D*H*W]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, dim=1)  # [B, Len_in, C]
        mask_flatten = torch.cat(mask_flatten, dim=1)  # [B, Len_in]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)

        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )  # [n_levels, 3]

        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], dim=1
        )  # [B, n_levels, 3]

        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer3D(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation='relu',
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn3D_PyTorch(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class MSDeformAttnTransformerEncoder3D(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Args:
            spatial_shapes: [n_levels, 3] with (D, H, W)
            valid_ratios: [B, n_levels, 3] in order (w_ratio, h_ratio, d_ratio)
        Returns:
            reference_points: [B, Len_in, n_levels, 3]
        """
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, D_ - 0.5, D_, dtype=torch.float32, device=device
                ),
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device
                ),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
                ),
                indexing='ij',
            )

            ref_z = ref_z.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 2] * D_
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_
            )

            ref = torch.stack((ref_x, ref_y, ref_z), dim=-1)  # [B, D*H*W, 3]
            reference_points_list.append(ref)

        reference_points = torch.cat(
            reference_points_list, dim=1
        )  # [B, Len_in, 3]
        reference_points = (
            reference_points[:, :, None] * valid_ratios[:, None]
        )  # [B, Len_in, n_levels, 3]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )

        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder3D(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        transformer_dropout,
        transformer_nheads,
        transformer_dim_feedforward,
        transformer_enc_layers,
        conv_dim,
        mask_dim,
        norm=None,
        transformer_in_features,
        common_stride,
    ):
        super().__init__()

        transformer_input_shape = {
            k: v
            for k, v in input_shape.items()
            if k in transformer_in_features
        }
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        self.in_features = [k for k, _ in input_shape]
        self.feature_channels = [v.channels for _, v in input_shape]

        transformer_input_shape = sorted(
            transformer_input_shape.items(), key=lambda x: x[1].stride
        )
        self.transformer_in_features = [k for k, _ in transformer_input_shape]
        transformer_in_channels = [
            v.channels for _, v in transformer_input_shape
        ]
        self.transformer_feature_strides = [
            v.stride for _, v in transformer_input_shape
        ]
        self.transformer_num_feature_levels = len(self.transformer_in_features)

        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.common_stride = common_stride
        self.maskformer_num_feature_levels = 3

        input_proj_list = []
        for in_channels in transformer_in_channels[::-1]:
            proj = nn.Sequential(
                Conv3d(in_channels, conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            )
            input_proj_list.append(proj)

        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1.0)
            nn.init.constant_(proj[0].bias, 0.0)

        self.transformer = MSDeformAttnTransformerEncoderOnly3D(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            enc_n_points=4,
        )

        self.pe_layer = PositionEmbeddingSine3D(conv_dim, normalize=True)

        self.mask_features = Conv3d(
            conv_dim, mask_dim, kernel_size=1, stride=1, padding=0
        )
        weight_init.c2_xavier_fill(self.mask_features)

        # With anisotropic ViTAdapter outputs, do not infer FPN depth from isotropic stride logic.
        # We only add one extra top-down fusion level from the finest backbone feature.
        self.num_fpn_levels = 1

        lateral_convs = []
        output_convs = []
        use_bias = norm == ''

        for idx, in_channels in enumerate(
            self.feature_channels[: self.num_fpn_levels]
        ):
            lateral_conv = Conv3d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias
            )
            output_conv = Conv3d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )

            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

            if norm == 'GN':
                lateral_module = nn.Sequential(
                    lateral_conv, nn.GroupNorm(32, conv_dim)
                )
                output_module = nn.Sequential(
                    output_conv,
                    nn.GroupNorm(32, conv_dim),
                    nn.ReLU(),
                )
            else:
                lateral_module = lateral_conv
                output_module = nn.Sequential(output_conv, nn.ReLU())

            self.add_module(f'adapter_{idx + 1}', lateral_module)
            self.add_module(f'layer_{idx + 1}', output_module)

            lateral_convs.append(lateral_module)
            output_convs.append(output_module)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            'input_shape': {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            'conv_dim': cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            'mask_dim': cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            'norm': cfg.MODEL.SEM_SEG_HEAD.NORM,
            'transformer_dropout': cfg.MODEL.MASK_FORMER.DROPOUT,
            'transformer_nheads': cfg.MODEL.MASK_FORMER.NHEADS,
            'transformer_dim_feedforward': 1024,
            'transformer_enc_layers': cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            'transformer_in_features': cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES,
            'common_stride': cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        }
        return ret

    @autocast('cuda', enabled=False)
    def forward_features(self, features):
        """
        Args:
            features: dict of 3D features, each [B, C, D, H, W]
        Returns:
            mask_features: [B, mask_dim, D_mask, H_mask, W_mask]
            transformer_encoder_lowest: kept for compatibility
            multi_scale_features: list of 3 feature levels for the transformer decoder
        """
        srcs_3d = []
        pos_3d = []
        spatial_shapes_3d = []

        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # [B, C, D, H, W]
            bs, c_in, d, h, w = x.shape
            spatial_shapes_3d.append((d, h, w))

            x_proj = self.input_proj[idx](x)  # [B, conv_dim, D, H, W]
            pos_emb = self.pe_layer(x_proj)  # [B, conv_dim, D, H, W]

            srcs_3d.append(x_proj)
            pos_3d.append(pos_emb)

        # True 3D deformable transformer encoder
        y, spatial_shapes, level_start_index = self.transformer(
            srcs_3d, pos_3d
        )

        # Restore each level back to 3D
        split_sizes = [int(d * h * w) for d, h, w in spatial_shapes.tolist()]
        y_split = torch.split(y, split_sizes, dim=1)

        out = []
        for i, z in enumerate(y_split):
            d, h, w = spatial_shapes_3d[i]
            z_3d = (
                z.transpose(1, 2).contiguous().view(bs, self.conv_dim, d, h, w)
            )
            out.append(z_3d)

        # 3D FPN top-down fusion
        for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            cur_fpn = self.lateral_convs[idx](x)
            y_top = F.interpolate(
                out[-1],
                size=cur_fpn.shape[-3:],
                mode='trilinear',
                align_corners=False,
            )
            y = cur_fpn + y_top
            y = self.output_convs[idx](y)
            out.append(y)

        multi_scale_features = out[: self.maskformer_num_feature_levels]

        # Keep the same return signature as your current code:
        #   mask_features, out0, multi_scale_features
        return self.mask_features(out[-1]), out[0], multi_scale_features
