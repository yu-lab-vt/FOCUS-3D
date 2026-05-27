"""
Detectron2-free config system for Windows FOCUS3D inference/fine-tuning.

This file replaces:
    from detectron2.config import get_cfg
    from mask2former import add_maskformer2_config

It supports:
    cfg.A.B
    cfg.A["B"]
    cfg.merge_from_file(...)
    _BASE_ inheritance in yaml
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class CfgNode(dict):
    """
    Minimal Detectron2-free CfgNode.

    Supports both:
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    and:
        cfg.MODEL.SEM_SEG_HEAD["CONVS_DIM"]
    """

    def __init__(self, init_dict: Mapping[str, Any] | None = None):
        super().__init__()
        init_dict = init_dict or {}
        for k, v in init_dict.items():
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v):
        if isinstance(v, CfgNode):
            return v
        if isinstance(v, dict):
            return CfgNode(v)
        if isinstance(v, list):
            return [CfgNode._wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(CfgNode._wrap(x) for x in v)
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(
                f"'CfgNode' object has no attribute '{name}'"
            ) from e

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, self._wrap(value))

    def clone(self):
        return copy.deepcopy(self)

    def set_new_allowed(self, allowed: bool):
        # Detectron2 compatibility stub.
        # We always allow new keys in this lightweight config.
        return None

    def freeze(self):
        # Detectron2 compatibility stub.
        return None

    def defrost(self):
        # Detectron2 compatibility stub.
        return None

    def merge_from_dict(self, other: Mapping[str, Any]):
        _merge_dict_into_cfg(other, self)

    def merge_from_file(self, file_path):
        loaded = _load_yaml_with_base(file_path)
        self.merge_from_dict(loaded)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, CfgNode):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [
                    x.to_dict() if isinstance(x, CfgNode) else x for x in v
                ]
            elif isinstance(v, tuple):
                out[k] = tuple(
                    x.to_dict() if isinstance(x, CfgNode) else x for x in v
                )
            else:
                out[k] = v
        return out


def _merge_dict_into_cfg(src: Mapping[str, Any], dst: CfgNode):
    for k, v in src.items():
        if k == '_BASE_':
            continue

        if isinstance(v, dict) and k in dst and isinstance(dst[k], CfgNode):
            _merge_dict_into_cfg(v, dst[k])
        else:
            dst[k] = CfgNode._wrap(copy.deepcopy(v))


def _merge_plain_dict(src: Mapping[str, Any], dst: dict):
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            _merge_plain_dict(v, dst[k])
        else:
            dst[k] = copy.deepcopy(v)


def _load_yaml_with_base(file_path):
    file_path = Path(file_path).resolve()

    with open(file_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    base_key = cfg.get('_BASE_', None)
    if base_key is None:
        return cfg

    if isinstance(base_key, str):
        base_files = [base_key]
    else:
        base_files = list(base_key)

    merged = {}

    for base in base_files:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = file_path.parent / base_path

        base_cfg = _load_yaml_with_base(base_path)
        _merge_plain_dict(base_cfg, merged)

    cfg.pop('_BASE_', None)
    _merge_plain_dict(cfg, merged)

    return merged


def get_cfg():
    """
    Detectron2-free replacement for detectron2.config.get_cfg().

    This creates the base tree that add_maskformer2_config_win(cfg)
    will fill.
    """
    cfg = CfgNode()

    # ------------------------------------------------------------------
    # INPUT base nodes
    # ------------------------------------------------------------------
    cfg.INPUT = CfgNode()
    cfg.INPUT.CROP = CfgNode()

    # ------------------------------------------------------------------
    # SOLVER base nodes
    # ------------------------------------------------------------------
    cfg.SOLVER = CfgNode()

    # ------------------------------------------------------------------
    # DATASETS base nodes
    # ------------------------------------------------------------------
    cfg.DATASETS = CfgNode()
    cfg.DATASETS.TRAIN = ('dummy_train',)
    cfg.DATASETS.TEST = ('dummy_test',)

    # ------------------------------------------------------------------
    # TEST base nodes
    # ------------------------------------------------------------------
    cfg.TEST = CfgNode()
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    # ------------------------------------------------------------------
    # MODEL base nodes
    # ------------------------------------------------------------------
    cfg.MODEL = CfgNode()
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = ''

    # Single-channel 3D microscopy default.
    # Your yaml can override this.
    cfg.MODEL.PIXEL_MEAN = [0.0]
    cfg.MODEL.PIXEL_STD = [1.0]

    # Detectron2 normally creates this tree before maskformer config is added.
    cfg.MODEL.SEM_SEG_HEAD = CfgNode()

    # Common semantic head defaults.
    # These must be overwritten by your yaml if training used different values.
    cfg.MODEL.SEM_SEG_HEAD.NAME = 'MSDeformAttnPixelDecoder3D'
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.NORM = 'GN'
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ['res2', 'res3', 'res4', 'res5']

    # Your custom 3D backbone defaults.
    # These are fallbacks only. The real values should come from your yaml.
    cfg.MODEL.BACKBONE = CfgNode()
    cfg.MODEL.BACKBONE.NAME = 'D2MAE3DBackbone'
    cfg.MODEL.BACKBONE.IMG_SIZE = [32, 96, 96]
    cfg.MODEL.BACKBONE.PATCH_SIZE = [4, 4, 4]
    cfg.MODEL.BACKBONE.IN_CHANS = 1
    cfg.MODEL.BACKBONE.DEPTH = 12
    cfg.MODEL.BACKBONE.DECODER_EMBED_DIM = 512
    cfg.MODEL.BACKBONE.DECODER_DEPTH = 8
    cfg.MODEL.BACKBONE.DECODER_NUM_HEADS = 16
    cfg.MODEL.BACKBONE.MLP_RATIO = 4.0
    cfg.MODEL.BACKBONE.NORM_PIX_LOSS = False

    cfg.MODEL.BACKBONE.VIT_ADAPTER = CfgNode()
    cfg.MODEL.BACKBONE.VIT_ADAPTER.PRETRAIN_SIZE = [32, 96, 96]
    cfg.MODEL.BACKBONE.VIT_ADAPTER.NUM_HEADS = 12
    cfg.MODEL.BACKBONE.VIT_ADAPTER.ITERATIONS = 4
    cfg.MODEL.BACKBONE.VIT_ADAPTER.INTERACTION_INDEXES = [
        [0, 2],
        [3, 5],
        [6, 8],
        [9, 11],
    ]
    cfg.MODEL.BACKBONE.VIT_ADAPTER.ADD_VIT_FEATURE = True
    cfg.MODEL.BACKBONE.VIT_ADAPTER.EMBEDED_DIM = 768

    # Apply your original add_maskformer2_config contents.
    add_maskformer2_config_win(cfg)

    return cfg


def add_maskformer2_config_win(cfg):
    """
    Detectron2-free version of your original add_maskformer2_config(cfg).

    This is copied from your Linux config.py, replacing CN() with CfgNode().
    """
    # ------------------------------------------------------------------
    # data config
    # ------------------------------------------------------------------
    cfg.INPUT.DATASET_MAPPER_NAME = 'mask_former_semantic'
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # ------------------------------------------------------------------
    # solver config
    # ------------------------------------------------------------------
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = 'ADAMW'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.VITADAPTER_MULTIPLIER = 1.0

    # ------------------------------------------------------------------
    # MaskFormer model config
    # ------------------------------------------------------------------
    cfg.MODEL.MASK_FORMER = CfgNode()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = 'res5'
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # inference config
    cfg.MODEL.MASK_FORMER.TEST = CfgNode()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # ------------------------------------------------------------------
    # pixel decoder config
    # ------------------------------------------------------------------
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0

    # Original default is 2D. Keep it for compatibility,
    # but maskformer_model_win.py should map it to BasePixelDecoder3D.
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = 'BasePixelDecoder'

    # ------------------------------------------------------------------
    # swin transformer backbone
    # ------------------------------------------------------------------
    cfg.MODEL.SWIN = CfgNode()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # ------------------------------------------------------------------
    # MaskFormer2 extra configs
    # ------------------------------------------------------------------
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = (
        'MultiScaleMaskedTransformerDecoder'
    )

    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        'res3',
        'res4',
        'res5',
    ]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # ------------------------------------------------------------------
    # Your later custom options.
    # These are not in the original snippet, but your current model uses them.
    # They are safe defaults and can be overridden by yaml.
    # ------------------------------------------------------------------
    cfg.MODEL.MASK_FORMER.DN = 'no'
    cfg.MODEL.MASK_FORMER.DN_NOISE_SCALE = 0.4
    cfg.MODEL.MASK_FORMER.DN_NUM = 100

    cfg.MODEL.MASK_FORMER.ENC_LOSS = False
    cfg.MODEL.MASK_FORMER.FEATURE_QUERY_INIT = True
    cfg.MODEL.MASK_FORMER.FEATURE_QUERY_INIT_DETACH = True
    cfg.MODEL.MASK_FORMER.FEATURE_QUERY_INIT_ADD_LEARNED_CONTENT = False
    cfg.MODEL.MASK_FORMER.FEATURE_QUERY_INIT_ADD_LEARNED_POS = False

    # Important:
    # inference_win.py sets this to False.
    # train_win.py should set it to True for future Windows fine-tuning.
    cfg.MODEL.MASK_FORMER.BUILD_CRITERION = False
