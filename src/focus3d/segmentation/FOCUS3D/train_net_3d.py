# train_net_3d.py
# Training script for 3D cell segmentation using modified Mask2Former components.
import copy
import itertools
import logging
import os
import weakref
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import TrainerBase
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from pathlib import Path
import sys

FOCUS3D_ROOT = Path(__file__).resolve().parent
if str(FOCUS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(FOCUS3D_ROOT))
# Import the modified components (ensure they are registered)
from mask2former import add_maskformer2_config
from mask2former.data.data_mapper import (
    MaskFormer3DInstanceDatasetMapper,
    register_real_dataset,
)
from mask2former.modeling.backbone.mae3d_backbone import D2MAE3DBackbone


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        """
        Same as Detectron2 DefaultTrainer, but enable
        find_unused_parameters=True for DDP.
        """
        TrainerBase.__init__(self)
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.model = model
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = AMPTrainer(model, data_loader, optimizer)
        else:
            self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormer3DInstanceDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        # Optional: load MAE pretrained weights if specified
        backbone = model.backbone
        if isinstance(backbone, D2MAE3DBackbone):
            pretrained_path = cfg.MODEL.BACKBONE.get('PRETRAINED', '')

            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    backbone.load_pretrained(pretrained_path)
                    freeze_unused_mae_decoder(model)

                    name = 'adapter.mae.blocks.0.attn.qkv.weight'
                    for n, p in model.backbone.named_parameters():
                        if n == name:
                            if comm.is_main_process():
                                print(
                                    '[Check loaded encoder weight]:',
                                    p.mean().item(),
                                    p.std().item(),
                                    flush=True,
                                )
                            break
                except Exception as e:
                    print(f'[ERROR] Exception in load_pretrained: {e}')
                    import traceback

                    traceback.print_exc()
                    raise
                logger = logging.getLogger(__name__)
                logger.info(
                    f'Loaded MAE pretrained weights from {pretrained_path}'
                )
        return model

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults['lr'] = cfg.SOLVER.BASE_LR
        defaults['weight_decay'] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(
                recurse=False
            ):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                # 1) MAE inside ViTAdapter: 0.1x
                if 'backbone.adapter.mae' in module_name:
                    hyperparams['lr'] = (
                        hyperparams['lr'] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )

                # 2) other ViTAdapter params: 1.0x
                elif 'backbone.adapter' in module_name:
                    hyperparams['lr'] = (
                        hyperparams['lr'] * cfg.SOLVER.VITADAPTER_MULTIPLIER
                    )

                # 3) other backbone params: 0.1x
                elif 'backbone' in module_name:
                    hyperparams['lr'] = (
                        hyperparams['lr'] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    'relative_position_bias_table' in module_param_name
                    or 'absolute_pos_embed' in module_param_name
                ):
                    print(module_param_name)
                    hyperparams['weight_decay'] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams['weight_decay'] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams['weight_decay'] = weight_decay_embed
                params.append({'params': [value], **hyperparams})
                # if "backbone" in module_name:
                #     print(f"[LR GROUP] {module_name}.{module_param_name}: lr={hyperparams['lr']}")

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model'
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x['params'] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == 'SGD':
            optimizer = maybe_add_full_model_gradient_clipping(
                torch.optim.SGD
            )(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == 'ADAMW':
            optimizer = maybe_add_full_model_gradient_clipping(
                torch.optim.AdamW
            )(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f'no optimizer type {optimizer_type}')
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model':
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


# -----------------------------------------------------------------------------
# 4. Setup and main
# -----------------------------------------------------------------------------
def freeze_unused_mae_decoder(model):
    """
    Freeze MAE reconstruction decoder parameters.
    These parameters are not used by downstream segmentation,
    so they should not be optimized or synchronized by DDP.
    """
    decoder_keywords = (
        'adapter.mae.decoder_pos_embed',
        'adapter.mae.decoder_embed',
        'adapter.mae.decoder_blocks',
        'adapter.mae.decoder_norm',
        'adapter.mae.decoder_pred',
        'adapter.mae.mask_token',
    )

    frozen = []
    for name, param in model.backbone.named_parameters():
        if any(key in name for key in decoder_keywords):
            param.requires_grad = False
            frozen.append(name)

    if comm.is_main_process():
        print(f'[Freeze] MAE decoder params: {len(frozen)}', flush=True)
        for name in frozen[:10]:
            print(f'  - {name}', flush=True)
        if len(frozen) > 10:
            print(f'  ... {len(frozen) - 10} more', flush=True)


def build_cfg(config_file=None, opts=None):
    """Build and return configuration object."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Make sure this field exists before merging yaml
    cfg.INPUT.IMAGE_SIZE = [32, 256, 256]

    cfg.set_new_allowed(True)

    if config_file:
        cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)

    return cfg


def setup(args):
    """
    Create cfg and perform basic setup.
    This function is called once in every distributed process.
    """
    cfg = build_cfg(args.config_file, args.opts)

    cfg.defrost()
    train_dataset_name = register_real_dataset(cfg)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name='mask2former',
    )
    return cfg


def main(args):
    print(f'[Rank {comm.get_rank()}] enter main', flush=True)
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        return {}

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Only print model structure on rank 0
    if comm.is_main_process():
        raw_model = trainer.model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        print('[Model]', type(raw_model))
        print('[Backbone]', type(raw_model.backbone))
        if hasattr(raw_model.backbone, 'adapter'):
            print('[Adapter]', type(raw_model.backbone.adapter))
            if hasattr(raw_model.backbone.adapter, 'mae'):
                print('[MAE]', type(raw_model.backbone.adapter.mae))

    return trainer.train()


def run_train(cfg, resume=False):
    """
    Single-process training entrance.
    Keep this for notebook / quick single-GPU debugging.
    For multi-GPU training, use command line with launch below.
    """
    cfg = cfg.clone()
    cfg.defrost()

    train_dataset_name = register_real_dataset(cfg)
    cfg.DATASETS.TRAIN = (train_dataset_name,)

    cfg.freeze()

    class Args:
        config_file = ''
        opts = []
        eval_only = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = 'tcp://127.0.0.1:49152'

    args = Args()
    args.resume = resume

    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name='mask2former',
    )

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer.train()


def run_train_distributed(
    config_file,
    num_gpus=2,
    resume=False,
    opts=None,
    num_machines=1,
    machine_rank=0,
    dist_url='auto',
):
    """
    Optional Python entrance for launching distributed training.
    Prefer command-line training on servers / H100 nodes.
    """
    from types import SimpleNamespace

    args = SimpleNamespace(
        config_file=config_file,
        resume=resume,
        eval_only=False,
        num_gpus=num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        opts=[] if opts is None else opts,
    )

    return launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print('Command Line Args:', args)
    print('[Main Process] before launch', flush=True)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    print('[Main Process] after launch', flush=True)
