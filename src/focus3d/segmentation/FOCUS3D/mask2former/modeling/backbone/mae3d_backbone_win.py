# --------------------------------------------------------
# MAE-based 3D Backbone with ViT-Adapter for Multi-Scale Features
# --------------------------------------------------------

import torch
import torch.nn as nn

try:
    from torch.nn.init import trunc_normal_
except ImportError:

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        # Fallback for very old PyTorch versions.
        # This is only used for parameter initialization.
        with torch.no_grad():
            return tensor.normal_(mean=mean, std=std)


from dataclasses import dataclass
from functools import partial

from .vitadapter import ViTAdapter

# Import the 3D MAE model (adjust the import path as needed)
try:
    from .models_mae_win import MaskedAutoencoderViT3D
except ImportError:
    raise ImportError(
        'Could not import MaskedAutoencoderViT3D. Make sure models_mae.py is in your PYTHONPATH.'
    )


@dataclass
class ShapeSpec:
    channels: int = None
    stride: int = None


class D2MAE3DBackbone(nn.Module):
    """
    A Detectron2 compatible backbone that wraps a 3D MAE encoder
    and adds an adapter to produce multi-scale features.
    """

    def __init__(self, cfg, input_shape=None):
        super().__init__()

        # Extract MAE configuration from cfg (example structure, adapt to your config)
        img_size = cfg.MODEL.BACKBONE.IMG_SIZE
        patch_size = cfg.MODEL.BACKBONE.PATCH_SIZE
        in_chans = cfg.MODEL.BACKBONE.IN_CHANS
        depth = cfg.MODEL.BACKBONE.DEPTH
        decoder_embed_dim = cfg.MODEL.BACKBONE.DECODER_EMBED_DIM
        decoder_depth = cfg.MODEL.BACKBONE.DECODER_DEPTH
        decoder_num_heads = cfg.MODEL.BACKBONE.DECODER_NUM_HEADS
        mlp_ratio = cfg.MODEL.BACKBONE.MLP_RATIO
        norm_pix_loss = cfg.MODEL.BACKBONE.NORM_PIX_LOSS

        vit_adapter_cfg = cfg.MODEL.BACKBONE.VIT_ADAPTER
        pretrain_size = vit_adapter_cfg['PRETRAIN_SIZE']
        num_heads = vit_adapter_cfg['NUM_HEADS']
        iterations = vit_adapter_cfg['ITERATIONS']
        interaction_indexes = vit_adapter_cfg['INTERACTION_INDEXES']
        add_vit_feature = vit_adapter_cfg['ADD_VIT_FEATURE']
        embed_dim = vit_adapter_cfg['EMBEDED_DIM']

        # We will load pretrained weights later if specified.
        # Optionally, we can freeze parts of the encoder.
        # For simplicity, we keep all parameters trainable here.

        # Determine the patch grid size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = tuple(patch_size)
        self.img_size = img_size

        # Choose which intermediate layers to use for multi-scale features.
        # We take the last four blocks (deepest). After reversal in forward_encoder,
        # indices 0,1,2,3 correspond to the last, second-last, third-last, fourth-last blocks.
        self.selected_layers = [0, 1, 2, 3]  # after reversal, deepest first

        # Create adapter with appropriate scale factors.
        self.adapter = ViTAdapter(
            mae_model=MaskedAutoencoderViT3D(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                norm_pix_loss=norm_pix_loss,
            ),
            pretrain_size=pretrain_size,
            num_heads=num_heads,
            embed_dim=embed_dim,
            iterations=iterations,
            interaction_indexes=interaction_indexes,
            add_vit_feature=add_vit_feature,
        )

        # Output feature names and their strides
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        self._out_feature_channels = dict.fromkeys(
            self._out_features, embed_dim
        )

        # Initialize weights (optional, MAE already initialized)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, pretrained_path):
        """Load MAE pretrained weights (only encoder part)."""
        checkpoint = torch.load(
            pretrained_path, map_location='cpu', weights_only=False
        )
        # If checkpoint contains 'model' key, use that.
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Filter out decoder keys (we don't need them)
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('decoder_'):
                continue
            # Remove 'module.' prefix if present (DDP)
            if k.startswith('module.'):
                k = k[7:]
            encoder_dict[k] = v

        # Load encoder part into self.mae (the full model, but decoder keys missing)
        try:
            missing, unexpected = self.adapter.mae.load_state_dict(
                encoder_dict, strict=False
            )
            print(
                f'Loaded pretrained MAE from {pretrained_path}. Missing keys: {missing}, Unexpected keys: {unexpected}'
            )
        except Exception as e:
            print(f'[ERROR] Failed to load state_dict: {e}')
            import traceback

            traceback.print_exc()
            raise

    def forward(self, x):
        # x: (B, C, D, H, W)
        features = self.adapter(x)  # 返回 list of 4 tensors: [f1, f2, f3, f4]
        # 映射到 Detectron2 输出格式（res2=1/4, res3=1/8, res4=1/16, res5=1/32）
        return {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        # Ensure input dimensions are multiples of 32 (largest stride)
        return 32
