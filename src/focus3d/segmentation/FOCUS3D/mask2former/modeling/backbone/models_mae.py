# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from .util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding: (B, C, D, H, W) -> (B, N, embed_dim)"""

    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        elif isinstance(img_size, list):
            img_size = tuple(img_size)
        # Convert patch_size to tuple if needed
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        self.img_size = (
            img_size
            if isinstance(img_size, tuple)
            else (img_size, img_size, img_size)
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, tuple)
            else (patch_size, patch_size, patch_size)
        )
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches = (
            (self.img_size[0] // self.patch_size[0])
            * (self.img_size[1] // self.patch_size[1])
            * (self.img_size[2] // self.patch_size[2])
        )

        # Use 3D convolution for patch projection
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, Dp, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()
        # Convert img_size to tuple if it's list
        if isinstance(img_size, (list, tuple)):
            img_size = tuple(img_size)
        else:
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
        else:
            patch_size = (patch_size, patch_size, patch_size)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.as_tensor(pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.as_tensor(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MaskedAutoencoderViT3D(nn.Module):
    """Masked Autoencoder with 3D VisionTransformer backbone (for volumes)"""

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics (3D)
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding (fixed sin-cos), will be initialized later
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Output dimension:
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size[0] * patch_size[1] * patch_size[2] * in_chans,
            bias=True,
        )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize position embeddings with 3D sin-cos
        # Compute grid size (number of patches per dimension)
        # Get grid size (number of patches per dimension)
        Dp = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        Hp = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        Wp = self.patch_embed.img_size[2] // self.patch_embed.patch_size[2]
        grid_size = (Dp, Hp, Wp)  # tuple for non-cubic volumes

        # Generate 3D position embeddings with grid_size tuple
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.as_tensor(pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.as_tensor(decoder_pos_embed).float().unsqueeze(0)
        )
        # Initialize patch_embed weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls_token and mask_token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear and layer norm layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert 3D volumes to patches.
        imgs: (N, C, D, H, W)
        Returns: (N, L, patch_size**3 * C)
        """
        p = self.patch_embed.patch_size  # assume tuple (pD, pH, pW) or int
        if isinstance(p, int):
            p = (p, p, p)
        C = imgs.shape[1]
        D, H, W = imgs.shape[2], imgs.shape[3], imgs.shape[4]
        assert D % p[0] == 0 and H % p[1] == 0 and W % p[2] == 0, (
            'Input dimensions must be divisible by patch size'
        )
        nD = D // p[0]
        nH = H // p[1]
        nW = W // p[2]
        # Reshape to (N, C, nD, pD, nH, pH, nW, pW)
        x = imgs.reshape(imgs.shape[0], C, nD, p[0], nH, p[1], nW, p[2])
        # Permute to bring patch dimensions together: (N, nD, nH, nW, pD, pH, pW, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # Flatten patch dimensions and channel
        x = x.reshape(imgs.shape[0], nD * nH * nW, p[0] * p[1] * p[2] * C)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to volumes.
        x: (N, L, pD * pH * pW * C)
        Returns: (N, C, D, H, W)
        """
        p = self.patch_embed.patch_size
        if isinstance(p, int):
            p = (p, p, p)
        C = self.patch_embed.in_chans
        # Get number of patches per dimension
        nD = self.patch_embed.img_size[0] // p[0]
        nH = self.patch_embed.img_size[1] // p[1]
        nW = self.patch_embed.img_size[2] // p[2]
        L = nD * nH * nW
        assert x.shape[1] == L, f'Expected {L} patches, got {x.shape[1]}'
        # Reshape to (N, nD, nH, nW, pD, pH, pW, C)
        x = x.reshape(x.shape[0], nD, nH, nW, p[0], p[1], p[2], C)
        # Permute to (N, C, nD, pD, nH, pH, nW, pW)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        # Reshape to final volume
        imgs = x.reshape(x.shape[0], C, nD * p[0], nH * p[1], nW * p[2])
        return imgs

    def random_masking(self, x, mask_ratio):
        """Same as original (sequence-level masking)."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x, mask_ratio, return_intermediate=False, return_layers=None
    ):
        # Embed patches
        x = self.patch_embed(x)

        # Add positional encoding without cls token
        x = x + self.pos_embed[:, 1:, :]
        if return_intermediate:
            intermediates = [x]  # layer0: after pos_embed, before masking
        # Masking
        if mask_ratio == 0:
            # No masking, keep order
            x_masked = x
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            ids_restore = (
                torch.arange(x.shape[1], device=x.device)
                .unsqueeze(0)
                .repeat(x.shape[0], 1)
            )
        else:
            x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)  # original design

        if return_intermediate:
            intermediates.append(x_masked)  # layer1: after masking
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)

        # Apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_intermediate:
                intermediates.append(x[:, 1:, :])
        x = self.norm(x)

        if return_intermediate:
            intermediates = intermediates[::-1]
            if return_layers is not None:
                intermediates = [
                    intermediates[j]
                    for j in range(len(intermediates))
                    if j in return_layers
                ]
            return x, intermediates, mask, ids_restore
        else:
            return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    # def forward_loss(self, imgs, pred, mask):
    #     """
    #     imgs: (N, 1, D, H, W)
    #     pred: (N, L, p**3 * 1)
    #     mask: (N, L), 0 is keep, 1 is remove
    #     """
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1e-6)**0.5

    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # (N, L)

    #     loss = (loss * mask).sum() / mask.sum()
    #     return loss

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)

        # do loss computation in float32 for numerical stability
        pred = pred.float()
        target = target.float()
        mask = mask.float()

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        mask_sum = mask.sum().clamp_min(1.0)
        loss = (loss * mask).sum() / mask_sum
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_3d(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_3d(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_3d(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = (
    mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)
mae_vit_large_patch16 = (
    mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)
mae_vit_huge_patch14 = (
    mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
)
