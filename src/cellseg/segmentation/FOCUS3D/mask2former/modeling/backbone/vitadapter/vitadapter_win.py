import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.init import trunc_normal_
except ImportError:

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean=mean, std=std)


from collections import OrderedDict

from .adapter_modules_win import (
    InteractionBlock,
    SpatialPriorModule,
    build_deform_inputs_3d,
)


class ViTAdapter(nn.Module):
    """
    True 3D ViT-Adapter prototype.

    This module keeps the original high-level design:
        1) Build multi-scale spatial prior features from a 3D CNN branch
        2) Interact MAE patch tokens with multi-scale prior features
        3) Recover 4 output scales for downstream segmentation

    The key difference from the old implementation:
        - All interaction is now done in true 3D token space
        - No more collapsing depth into the batch dimension
        - No more pseudo-2D deformable attention
    """

    def __init__(
        self,
        mae_model,
        pretrain_size=256,
        num_heads=16,
        embed_dim=1056,
        iterations=4,
        interaction_indexes=None,
        add_vit_feature=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pretrain_size = pretrain_size
        self.mae = mae_model
        self.pos_drop = nn.Dropout(p=0.0)
        self.add_vit_feature = add_vit_feature

        # 1. Spatial prior module
        self.spm = SpatialPriorModule(inplanes=64, embed_dim=embed_dim)

        # 2. Interaction schedule
        if interaction_indexes is None:
            self.interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
        else:
            self.interaction_indexes = interaction_indexes

        assert iterations == len(self.interaction_indexes), (
            f'iterations ({iterations}) must equal len(interaction_indexes) '
            f'({len(self.interaction_indexes)})'
        )

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    n_points=4,
                    drop_path=0.1,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(len(self.interaction_indexes))
            ]
        )

        # 3. 3D upsampling from stride-8 to stride-4
        self.up = nn.ConvTranspose3d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )
        # self.up = nn.ConvTranspose3d(
        #     embed_dim,
        #     embed_dim,
        #     kernel_size=(1, 2, 2),
        #     stride=(1, 2, 2)
        # )
        # 4. Output norms
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.level_embed, std=0.02)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W]

        Returns:
            list of 4 feature maps:
                f1: stride 4
                f2: stride 8
                f3: stride 16
                f4: stride 32
        """
        B, _, D, H, W = x.shape

        # ------------------------------------------------------------------
        # Spatial prior branch
        # c1 is kept in dense 3D map form
        # c2_seq/c3_seq/c4_seq are returned as token sequences [B, L, C]
        # ------------------------------------------------------------------
        c1, c2_seq, c3_seq, c4_seq, c_shapes = self.spm(x)
        assert len(c_shapes) == 3, f'Expected 3 c_shapes, got {len(c_shapes)}'
        c2_shape, c3_shape, c4_shape = c_shapes

        assert c1.shape[-3:] == (D, H // 4, W // 4), (
            f'c1 shape mismatch: got {c1.shape[-3:]}, expected {(D, H // 4, W // 4)}'
        )
        assert c2_shape == (D, H // 8, W // 8), (
            f'c2_shape mismatch: got {c2_shape}, expected {(D, H // 8, W // 8)}'
        )
        assert c3_shape == (D // 2, H // 16, W // 16), (
            f'c3_shape mismatch: got {c3_shape}, expected {(D // 2, H // 16, W // 16)}'
        )
        assert c4_shape == (D // 4, H // 32, W // 32), (
            f'c4_shape mismatch: got {c4_shape}, expected {(D // 4, H // 32, W // 32)}'
        )
        # assert c4_shape == (D // 2, H // 32, W // 32), \
        #     f"c4_shape mismatch: got {c4_shape}, expected {(D//2, H//32, W//32)}"
        c2_seq, c3_seq, c4_seq = self._add_level_embed(c2_seq, c3_seq, c4_seq)
        c_all = torch.cat([c2_seq, c3_seq, c4_seq], dim=1)  # [B, Lc, C]

        # ------------------------------------------------------------------
        # MAE patch tokens at stride = patch_size
        # ------------------------------------------------------------------
        x_mae = self.mae.patch_embed(x)  # [B, Lx, C]

        patch_size = self.mae.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        xD = D // patch_size[0]
        xH = H // patch_size[1]
        xW = W // patch_size[2]
        x_shape = (xD, xH, xW)

        assert x_mae.shape[1] == xD * xH * xW, (
            f'x_mae token length mismatch: got {x_mae.shape[1]}, expected {xD * xH * xW}'
        )

        if self.mae.pos_embed is not None:
            x_mae = x_mae + self.mae.pos_embed[:, 1:, :]
        x_mae = self.pos_drop(x_mae)

        # Build 3D deformable-attention inputs for:
        #   - injector: query=x_mae, feat=c_all
        #   - extractor: query=c_all, feat=x_mae
        deform_inputs1, deform_inputs2 = build_deform_inputs_3d(
            batch_size=B,
            x_shape=x_shape,
            c_shapes=c_shapes,
            device=x.device,
        )

        outs = []
        for i, layer in enumerate(self.interactions):
            idx = self.interaction_indexes[i]
            blocks = self.mae.blocks[idx[0] : idx[-1] + 1]

            x_mae, c_all = layer(
                x=x_mae,
                c=c_all,
                blocks=blocks,
                deform_inputs1=deform_inputs1,
                deform_inputs2=deform_inputs2,
                x_shape=x_shape,
                c_shapes=c_shapes,
            )

            x_feat = (
                x_mae.transpose(1, 2)
                .contiguous()
                .view(B, self.embed_dim, xD, xH, xW)
            )
            outs.append(x_feat)

        # ------------------------------------------------------------------
        # Split c_all back into three scales: stride-8 / stride-16 / stride-32
        # ------------------------------------------------------------------
        c2_shape, c3_shape, c4_shape = c_shapes
        c2_len = c2_shape[0] * c2_shape[1] * c2_shape[2]
        c3_len = c3_shape[0] * c3_shape[1] * c3_shape[2]
        c4_len = c4_shape[0] * c4_shape[1] * c4_shape[2]

        c2_tok = c_all[:, 0:c2_len, :]
        c3_tok = c_all[:, c2_len : c2_len + c3_len, :]
        c4_tok = c_all[:, c2_len + c3_len : c2_len + c3_len + c4_len, :]

        f2 = (
            c2_tok.transpose(1, 2)
            .contiguous()
            .view(B, self.embed_dim, c2_shape[0], c2_shape[1], c2_shape[2])
        )
        f3 = (
            c3_tok.transpose(1, 2)
            .contiguous()
            .view(B, self.embed_dim, c3_shape[0], c3_shape[1], c3_shape[2])
        )
        f4 = (
            c4_tok.transpose(1, 2)
            .contiguous()
            .view(B, self.embed_dim, c4_shape[0], c4_shape[1], c4_shape[2])
        )
        assert f2.shape[-3:] == c2_shape, (
            f'f2 shape mismatch: got {f2.shape[-3:]}, expected {c2_shape}'
        )
        assert f3.shape[-3:] == c3_shape, (
            f'f3 shape mismatch: got {f3.shape[-3:]}, expected {c3_shape}'
        )
        assert f4.shape[-3:] == c4_shape, (
            f'f4 shape mismatch: got {f4.shape[-3:]}, expected {c4_shape}'
        )
        # Align exact target sizes
        target_f2 = c2_shape
        target_f3 = c3_shape
        target_f4 = c4_shape

        if f2.shape[-3:] != target_f2:
            f2 = F.interpolate(
                f2, size=target_f2, mode='trilinear', align_corners=False
            )
        if f3.shape[-3:] != target_f3:
            f3 = F.interpolate(
                f3, size=target_f3, mode='trilinear', align_corners=False
            )
        if f4.shape[-3:] != target_f4:
            f4 = F.interpolate(
                f4, size=target_f4, mode='trilinear', align_corners=False
            )

        # f1 is stride-4
        f1 = self.up(f2)
        if f1.shape[-3:] != c1.shape[-3:]:
            f1 = F.interpolate(
                f1, size=c1.shape[-3:], mode='trilinear', align_corners=False
            )
        assert f1.shape[-3:] == c1.shape[-3:], (
            f'f1 and c1 shape mismatch before fusion: f1={f1.shape[-3:]}, c1={c1.shape[-3:]}'
        )
        f1 = f1 + c1

        # ------------------------------------------------------------------
        # Optionally fuse intermediate MAE features from each interaction stage
        # All x1/x2/x3/x4 are stride-16 features, but we inject them into different scales
        # by trilinear interpolation, following the original design.
        # ------------------------------------------------------------------
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            f1 = f1 + F.interpolate(
                x1, size=f1.shape[-3:], mode='trilinear', align_corners=False
            )
            f2 = f2 + F.interpolate(
                x2, size=f2.shape[-3:], mode='trilinear', align_corners=False
            )
            f3 = f3 + F.interpolate(
                x3, size=f3.shape[-3:], mode='trilinear', align_corners=False
            )
            f4 = f4 + F.interpolate(
                x4, size=f4.shape[-3:], mode='trilinear', align_corners=False
            )

            assert x1.shape[1] == self.embed_dim
            assert x2.shape[1] == self.embed_dim
            assert x3.shape[1] == self.embed_dim
            assert x4.shape[1] == self.embed_dim

            assert (
                F.interpolate(
                    x1,
                    size=f1.shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                ).shape[-3:]
                == f1.shape[-3:]
            )
            assert (
                F.interpolate(
                    x2,
                    size=f2.shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                ).shape[-3:]
                == f2.shape[-3:]
            )
            assert (
                F.interpolate(
                    x3,
                    size=f3.shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                ).shape[-3:]
                == f3.shape[-3:]
            )
            assert (
                F.interpolate(
                    x4,
                    size=f4.shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                ).shape[-3:]
                == f4.shape[-3:]
            )

        return [self.norm1(f1), self.norm2(f2), self.norm3(f3), self.norm4(f4)]

    def load_mae_weights(self, checkpoint_path):
        """
        Load MAE encoder weights into self.mae.

        This helper keeps the old behavior:
            - only MAE encoder-related keys are loaded
            - keys are remapped with 'mae.' prefix before loading into the whole adapter
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if any(
                k.startswith(p)
                for p in ['patch_embed', 'pos_embed', 'blocks', 'norm']
            ):
                new_state_dict['mae.' + k] = v

        self.load_state_dict(new_state_dict, strict=False)
