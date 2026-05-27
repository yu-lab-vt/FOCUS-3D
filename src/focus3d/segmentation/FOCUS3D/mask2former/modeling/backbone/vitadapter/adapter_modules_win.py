from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Detectron2/timm-free DropPath.

    Compatible with timm DropPath behavior.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


def get_reference_points_single_scale_3d(shape, device):
    """
    Build normalized reference points for a single 3D grid.

    Args:
        shape: (D, H, W)
    Returns:
        reference_points: [1, D*H*W, 3] in normalized coordinates (x, y, z)
    """
    D, H, W = shape
    ref_z, ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, D - 0.5, D, dtype=torch.float32, device=device),
        torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
        torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    ref_z = ref_z.reshape(-1)[None] / D
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref = torch.stack((ref_x, ref_y, ref_z), dim=-1)  # [1, L, 3]
    return ref


def get_reference_points_multi_scale_3d(shapes, device):
    """
    Build concatenated normalized reference points for multiple 3D grids.

    Args:
        shapes: list of (D, H, W)
    Returns:
        reference_points: [1, sum(L_i), 3]
    """
    refs = [get_reference_points_single_scale_3d(s, device) for s in shapes]
    return torch.cat(refs, dim=1)


def build_level_start_index(spatial_shapes):
    """
    Args:
        spatial_shapes: [n_levels, 3]
    Returns:
        level_start_index: [n_levels]
    """
    return torch.cat(
        (
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        )
    )


def build_deform_inputs_3d(batch_size, x_shape, c_shapes, device):
    """
    Build deformable-attention metadata for:
        1) injector: query = x tokens, feat = multi-scale c tokens
        2) extractor: query = multi-scale c tokens, feat = x tokens

    Args:
        batch_size: int
        x_shape: (D, H, W) for MAE token grid
        c_shapes: list of three shapes for c2/c3/c4, each (D, H, W)

    Returns:
        deform_inputs1 = [reference_points1, spatial_shapes1, level_start_index1]
        deform_inputs2 = [reference_points2, spatial_shapes2, level_start_index2]
    """
    # Injector:
    # query = x at stride-16
    # feat  = c2/c3/c4 at stride-8/16/32
    spatial_shapes1 = torch.as_tensor(
        c_shapes, dtype=torch.long, device=device
    )
    level_start_index1 = build_level_start_index(spatial_shapes1)

    ref_query_x = get_reference_points_single_scale_3d(
        x_shape, device
    )  # [1, Lx, 3]
    ref_query_x = ref_query_x[:, :, None, :].repeat(
        1, 1, len(c_shapes), 1
    )  # [1, Lx, 3_levels, 3]
    ref_query_x = ref_query_x.repeat(batch_size, 1, 1, 1)

    deform_inputs1 = [ref_query_x, spatial_shapes1, level_start_index1]

    # Extractor:
    # query = concatenated c2/c3/c4
    # feat  = x at stride-16
    spatial_shapes2 = torch.as_tensor(
        [x_shape], dtype=torch.long, device=device
    )
    level_start_index2 = build_level_start_index(spatial_shapes2)

    ref_query_c = get_reference_points_multi_scale_3d(
        c_shapes, device
    )  # [1, Lc, 3]
    ref_query_c = ref_query_c[:, :, None, :]  # [1, Lc, 1, 3]
    ref_query_c = ref_query_c.repeat(batch_size, 1, 1, 1)

    deform_inputs2 = [ref_query_c, spatial_shapes2, level_start_index2]

    return deform_inputs1, deform_inputs2


class MSDeformAttn3D_PyTorch(nn.Module):
    """
    Pure PyTorch prototype of true 3D multi-scale deformable attention.

    Input semantics:
        query:                [B, Len_q, C]
        reference_points:     [B, Len_q, n_levels, 3] or [B, Len_q, 1, 3]
        input_flatten:        [B, Len_in, C]
        input_spatial_shapes: [n_levels, 3], rows are (D, H, W)
        input_level_start_index: [n_levels]
        input_padding_mask:   [B, Len_in] or None

    Coordinates use normalized (x, y, z) in [0, 1].
    Sampling is done with trilinear grid_sample.
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
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Initialize offsets in a coarse 3D pattern.
        grid_init = torch.zeros(self.n_heads, self.n_levels, self.n_points, 3)
        for h in range(self.n_heads):
            for l in range(self.n_levels):
                for p in range(self.n_points):
                    theta = 2.0 * torch.pi * (p / max(self.n_points, 1))
                    z = -1.0 + 2.0 * (p + 0.5) / max(self.n_points, 1)
                    r_xy = max(1e-6, float((max(0.0, 1.0 - z * z)) ** 0.5))
                    x = r_xy * float(torch.cos(torch.tensor(theta)))
                    y = r_xy * float(torch.sin(torch.tensor(theta)))
                    vec = torch.tensor([x, y, z], dtype=torch.float32)
                    vec = vec / (vec.abs().max() + 1e-6)
                    grid_init[h, l, p] = vec * (p + 1)

        self.sampling_offsets.bias.data = grid_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

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
        Returns:
            output: [B, Len_q, C]
        """
        B, Len_q, _ = query.shape
        B_in, Len_in, _ = input_flatten.shape
        assert B_in == B, 'Batch size mismatch.'
        assert input_spatial_shapes.shape[0] == self.n_levels, (
            'n_levels mismatch.'
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
        # attention_weights = F.softmax(attention_weights, dim=-1)    # zqh0407

        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[2] == 1 and self.n_levels > 1:
            reference_points = reference_points.expand(
                -1, -1, self.n_levels, -1
            )
        assert reference_points.shape[:3] == (B, Len_q, self.n_levels), (
            f'reference_points shape mismatch: got {reference_points.shape}, '
            f'expected [B, Len_q, {self.n_levels}, 3]'
        )

        output = query.new_zeros(B, Len_q, self.n_heads, self.d_per_head)

        for lvl in range(self.n_levels):
            D_l, H_l, W_l = input_spatial_shapes[lvl].tolist()
            start = input_level_start_index[lvl].item()
            end = start + D_l * H_l * W_l

            # [B, D*H*W, n_heads, head_dim]
            value_l = value[:, start:end]
            value_l = value_l.view(
                B, D_l, H_l, W_l, self.n_heads, self.d_per_head
            )
            value_l = value_l.permute(
                0, 4, 5, 1, 2, 3
            ).contiguous()  # [B, n_heads, head_dim, D, H, W]
            value_l = value_l.view(
                B * self.n_heads, self.d_per_head, D_l, H_l, W_l
            )

            ref_l = reference_points[:, :, lvl]  # [B, Len_q, 3]
            normalizer = query.new_tensor([W_l, H_l, D_l]).view(1, 1, 1, 1, 3)
            sampling_locations = (
                ref_l[:, :, None, None, :]
                + sampling_offsets[:, :, :, lvl] / normalizer
            )

            grid = 2.0 * sampling_locations - 1.0
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
            # [B*n_heads, head_dim, Len_q, n_points, 1]
            sampled = sampled.squeeze(-1)
            sampled = sampled.view(
                B, self.n_heads, self.d_per_head, Len_q, self.n_points
            )
            sampled = sampled.permute(
                0, 3, 1, 4, 2
            ).contiguous()  # [B, Len_q, n_heads, n_points, head_dim]

            attn = attention_weights[:, :, :, lvl].unsqueeze(-1)
            output = output + (sampled * attn).sum(dim=3)

        output = output.view(B, Len_q, self.d_model)
        output = self.output_proj(output)
        return output


class DWConv3D(nn.Module):
    """
    Depthwise 3D convolution for token sequences.

    This version is truly 3D: kernel = 3x3x3.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
        )

    def forward(self, x, shape):
        """
        Args:
            x: [B, L, C]
            shape: (D, H, W)
        """
        B, L, C = x.shape
        D, H, W = shape
        assert L == D * H * W, f'Sequence length {L} != D*H*W ({D * H * W})'

        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN3D(nn.Module):
    """
    3D FFN for a single-scale token sequence.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv3D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, shape):
        x = self.fc1(x)
        x = self.dwconv(x, shape)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class MultiScaleConvFFN3D(nn.Module):
    """
    3D FFN for concatenated multi-scale token sequences.
    The same fc1/fc2/dwconv modules are shared across scales, while each scale
    is reshaped and convolved independently in true 3D.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv3D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, spatial_shapes):
        """
        Args:
            x: [B, L_total, C]
            spatial_shapes: list of (D, H, W)
        """
        x = self.fc1(x)

        chunks = []
        start = 0
        for shape in spatial_shapes:
            D, H, W = shape
            length = D * H * W
            chunk = x[:, start : start + length, :]
            chunk = self.dwconv(chunk, shape)
            chunks.append(chunk)
            start += length

        x = torch.cat(chunks, dim=1)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class Extractor3D(nn.Module):
    """
    Extract information from x tokens back into multi-scale c tokens.
    query = c
    feat  = x
    """

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        self.attn = MSDeformAttn3D_PyTorch(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
        )

        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = MultiScaleConvFFN3D(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop,
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(
        self,
        query,
        reference_points,
        feat,
        spatial_shapes,
        level_start_index,
        query_spatial_shapes,
    ):
        attn = self.attn(
            self.query_norm(query),
            reference_points,
            self.feat_norm(feat),
            spatial_shapes,
            level_start_index,
            None,
        )
        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(
                self.ffn(self.ffn_norm(query), query_spatial_shapes)
            )
        return query


class Injector3D(nn.Module):
    """
    Inject information from multi-scale c tokens into x tokens.
    query = x
    feat  = c
    """

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)

        self.attn = MSDeformAttn3D_PyTorch(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
        )
        self.gamma = nn.Parameter(
            init_values * torch.ones(dim), requires_grad=True
        )

    def forward(
        self, query, reference_points, feat, spatial_shapes, level_start_index
    ):
        attn = self.attn(
            self.query_norm(query),
            reference_points,
            self.feat_norm(feat),
            spatial_shapes,
            level_start_index,
            None,
        )
        return query + self.gamma * attn


class InteractionBlock(nn.Module):
    """
    One interaction stage:
        1) injector: c -> x
        2) MAE blocks
        3) extractor: x -> c
    """

    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
    ):
        super().__init__()
        self.injector = Injector3D(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
        )
        self.extractor = Extractor3D(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
        )

    def forward(
        self,
        x,
        c,
        blocks,
        deform_inputs1,
        deform_inputs2,
        x_shape,
        c_shapes,
    ):
        # Injector: query=x, feat=c
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        # MAE transformer blocks on x tokens
        for blk in blocks:
            x = blk(x)

        # Extractor: query=c, feat=x
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            query_spatial_shapes=c_shapes,
        )

        return x, c


class SpatialPriorModule(nn.Module):
    """
    3D spatial prior module.

    Outputs:
        c1: dense anisotropic feature map
            z stride = 1, xy stride = 4
        c2_seq: token sequence
            z stride = 1, xy stride = 8
        c3_seq: token sequence
            z stride = 2, xy stride = 16
        c4_seq: token sequence
            z stride = 2, xy stride = 32
        c_shapes: [(D2,H2,W2), (D3,H3,W3), (D4,H4,W4)]
    """

    def __init__(self, inplanes=64, embed_dim=768):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                inplanes,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inplanes,
                inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv3d(
        #         4 * inplanes,
        #         4 * inplanes,
        #         kernel_size=3,
        #         stride=(1, 2, 2),
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True),
        # )
        self.fc1 = nn.Conv3d(inplanes, embed_dim, kernel_size=1)
        self.fc2 = nn.Conv3d(2 * inplanes, embed_dim, kernel_size=1)
        self.fc3 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1)
        self.fc4 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1)

    @staticmethod
    def to_sequence(feat):
        """
        Convert [B, C, D, H, W] to [B, D*H*W, C].
        """
        B, C, D, H, W = feat.shape
        seq = feat.flatten(2).transpose(1, 2).contiguous()
        return seq, (D, H, W)

    def forward(self, x):
        c1 = self.stem(x)  # stride 4
        c2 = self.conv2(c1)  # stride 8
        c3 = self.conv3(c2)  # stride 16
        c4 = self.conv4(c3)  # stride 32

        # input x: [B, C, D, H, W]
        D, H, W = x.shape[-3:]

        assert c1.shape[-3] == D, (
            f'c1 z should stay unchanged, got {c1.shape[-3]} vs {D}'
        )
        assert c1.shape[-2] == H // 4 and c1.shape[-1] == W // 4, (
            f'c1 xy shape mismatch: got {c1.shape[-3:]}, expected ({D}, {H // 4}, {W // 4})'
        )

        assert c2.shape[-3] == D, (
            f'c2 z should stay unchanged, got {c2.shape[-3]} vs {D}'
        )
        assert c2.shape[-2] == H // 8 and c2.shape[-1] == W // 8, (
            f'c2 xy shape mismatch: got {c2.shape[-3:]}, expected ({D}, {H // 8}, {W // 8})'
        )

        assert c3.shape[-3] == D // 2, (
            f'c3 z mismatch: got {c3.shape[-3]} vs {D // 2}'
        )
        assert c3.shape[-2] == H // 16 and c3.shape[-1] == W // 16, (
            f'c3 shape mismatch: got {c3.shape[-3:]}, expected ({D // 2}, {H // 16}, {W // 16})'
        )

        assert c4.shape[-3] == D // 4, (
            f'c4 z mismatch: got {c4.shape[-3]} vs {D // 4}'
        )
        assert c4.shape[-2] == H // 32 and c4.shape[-1] == W // 32, (
            f'c4 shape mismatch: got {c4.shape[-3:]}, expected ({D // 4}, {H // 32}, {W // 32})'
        )
        # assert c4.shape[-3] == D // 2, f"c4 z mismatch: got {c4.shape[-3]} vs {D//2}"
        # assert c4.shape[-2] == H // 32 and c4.shape[-1] == W // 32, \
        #     f"c4 shape mismatch: got {c4.shape[-3:]}, expected ({D//2}, {H//32}, {W//32})"

        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        c2_seq, c2_shape = self.to_sequence(c2)
        c3_seq, c3_shape = self.to_sequence(c3)
        c4_seq, c4_shape = self.to_sequence(c4)

        c_shapes = [c2_shape, c3_shape, c4_shape]

        return c1, c2_seq, c3_seq, c4_seq, c_shapes
