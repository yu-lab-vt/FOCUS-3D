# Copyright (c) Facebook, Inc. and its affiliates.
import os
from types import SimpleNamespace
from typing import Tuple
import math
import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F

from .modeling.backbone.mae3d_backbone_win import D2MAE3DBackbone
from .modeling.pixel_decoder.fpn3d_win import BasePixelDecoder3D
from .modeling.pixel_decoder.msdeformattn3d_win import (
    MSDeformAttnPixelDecoder3D,
)
from .modeling.transformer_decoder.mask2former_transformer_decoder_3d_win import (
    MultiScaleMaskedTransformerDecoder,
)


def mem(tag, device='cuda'):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated(device) / 1024**3
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f'[MEM] {tag:<30} current={cur:.3f} GB, peak={peak:.3f} GB')


def _make_empty_metadata():
    return SimpleNamespace(thing_dataset_id_to_contiguous_id={0: 0})


class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        criterion=None,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = (
            sem_seg_postprocess_before_inference
        )
        self.register_buffer(
            'pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1, 1), False
        )
        self.register_buffer(
            'pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1, 1), False
        )
        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.fast_score_thresh = None
        self.fast_mask_thresh = None

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = D2MAE3DBackbone(cfg, input_shape=None)
        # Build pixel decoder without Detectron2 configurable / registry
        pixel_decoder_name = getattr(cfg.MODEL.SEM_SEG_HEAD, 'NAME', None)
        if pixel_decoder_name is None:
            pixel_decoder_name = getattr(
                cfg.MODEL.SEM_SEG_HEAD, 'PIXEL_DECODER_NAME', None
            )

        if pixel_decoder_name == 'BasePixelDecoder3D':
            pixel_decoder_kwargs = BasePixelDecoder3D.from_config(
                cfg,
                backbone.output_shape(),
            )
            pixel_decoder = BasePixelDecoder3D(**pixel_decoder_kwargs)

        elif pixel_decoder_name == 'MSDeformAttnPixelDecoder3D':
            pixel_decoder_kwargs = MSDeformAttnPixelDecoder3D.from_config(
                cfg,
                backbone.output_shape(),
            )
            pixel_decoder = MSDeformAttnPixelDecoder3D(**pixel_decoder_kwargs)

        else:
            raise ValueError(
                f'Unsupported pixel decoder: {pixel_decoder_name}'
            )

        # Build transformer decoder
        decoder_kwargs = MultiScaleMaskedTransformerDecoder.from_config(
            cfg,
            in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_classification=True,
        )
        transformer_decoder = MultiScaleMaskedTransformerDecoder(
            **decoder_kwargs
        )

        build_criterion = bool(
            getattr(cfg.MODEL.MASK_FORMER, 'BUILD_CRITERION', False)
        )

        if build_criterion:
            criterion = build_criterion_from_cfg(cfg)
        else:
            criterion = None

        return {
            'backbone': backbone,
            'pixel_decoder': pixel_decoder,
            'transformer_decoder': transformer_decoder,
            'criterion': criterion,
            'num_queries': cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            'object_mask_threshold': cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            'overlap_threshold': cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            'metadata': _make_empty_metadata(),
            'size_divisibility': cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            'sem_seg_postprocess_before_inference': (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            # inference
            'semantic_on': cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            'instance_on': cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            'panoptic_on': cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            'test_topk_per_image': cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _to_uint8_grayscale_volume(self, vol: np.ndarray) -> np.ndarray:
        vol = np.asarray(vol, dtype=np.float32)
        vmin = float(vol.min())
        vmax = float(vol.max())
        if vmax <= vmin:
            return np.zeros(vol.shape, dtype=np.uint8)
        vol = (vol - vmin) / (vmax - vmin)
        vol = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
        return vol

    def _build_overlay_volume(
        self,
        gray_vol_uint8: np.ndarray,  # [D,H,W], uint8
        mask_prob_vol: np.ndarray,  # [D,H,W], float32 in [0,1]
        alpha: float = 0.45,
        threshold: float = 0.5,
    ) -> np.ndarray:
        gray = gray_vol_uint8.astype(np.float32) / 255.0
        rgb = np.stack([gray, gray, gray], axis=-1)  # [D,H,W,3]

        mask_alpha = (mask_prob_vol >= threshold).astype(np.float32) * alpha
        mask_alpha = mask_alpha[..., None]  # [D,H,W,1]

        red = np.zeros_like(rgb, dtype=np.float32)
        red[..., 0] = 1.0

        overlay = rgb * (1.0 - mask_alpha) + red * mask_alpha
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        return overlay


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)
        images = torch.stack(images, dim=0)
        features = self.backbone(images)
        mask_features, _, multi_scale_features = (
            self.pixel_decoder.forward_features(features)
        )

        if self.training:
            if self.criterion is None:
                raise RuntimeError(
                    'This Windows MaskFormer model was built without criterion. '
                    'For Windows fine-tuning, set cfg.MODEL.MASK_FORMER.BUILD_CRITERION = True.'
                )
            if 'instances' in batched_inputs[0]:
                gt_instances = [
                    x['instances'].to(self.device) for x in batched_inputs
                ]
                targets = self.prepare_targets(gt_instances, images)

                outputs, mask_dict = self.transformer_decoder(
                    multi_scale_features,
                    mask_features,
                    targets=targets,
                )


                losses = self.criterion(outputs, targets, mask_dict)
                # Build a zero tensor on the correct device.
                # Use an existing loss if possible, so dtype/device are consistent.
                if len(losses) > 0:
                    zero_loss = next(iter(losses.values())).sum() * 0.0
                else:
                    zero_loss = next(self.parameters()).sum() * 0.0

                # Remove losses that are not registered in weight_dict.
                for k in list(losses.keys()):
                    if k not in self.criterion.weight_dict:
                        losses.pop(k)

                # Fill missing expected loss keys.
                for k in self.criterion.weight_dict.keys():
                    if k not in losses:
                        losses[k] = zero_loss

                # Apply loss weights.
                for k in list(losses.keys()):
                    losses[k] = losses[k] * self.criterion.weight_dict[k]

                return losses

            else:
                device = next(self.parameters()).device
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                return {'loss_dummy': loss}

        else:
            outputs, _ = self.transformer_decoder(
                multi_scale_features,
                mask_features,
                targets=None,
            )

            mask_cls_results = outputs['pred_logits']
            mask_pred_results = outputs['pred_masks']

            del outputs
            processed_results = []
            for b, (
                mask_cls_result,
                mask_pred_result,
                input_per_image,
            ) in enumerate(
                zip(mask_cls_results, mask_pred_results, batched_inputs)
            ):
                processed_results.append({})

                instance_r = self.instance_inference(
                    mask_cls_result,
                    mask_pred_result,
                )

                pred_masks = instance_r['pred_masks']

                if pred_masks.shape[0] > 0:

                    pred_masks = F.interpolate(
                        pred_masks.unsqueeze(0),
                        size=images.shape[-3:],
                        mode='trilinear',
                        align_corners=False,
                    ).squeeze(0)

                else:

                    pred_masks = pred_masks.new_empty(
                        (
                            0,
                            int(images.shape[-3]),
                            int(images.shape[-2]),
                            int(images.shape[-1]),
                        )
                    )


                instance_r['pred_masks'] = pred_masks

                if self.instance_on:
                    processed_results[-1].update(instance_r)

            return processed_results

    def forward_one_click(self, batched_inputs):
        """
        Fast inference path for one-click segmentation.

        Differences from normal forward():
        - Runs the same backbone, pixel decoder, and transformer decoder.
        - Does NOT call instance_inference().
        - Does NOT upsample all query masks to the input patch size.
        - Returns all queries at the transformer's native mask resolution.
        - Query selection and single-mask upsampling are handled later by
        click_inference.select_clicked_query_instance().

        Returns:
            list[dict], one dict per input sample:
                pred_scores:  [Q]
                pred_classes: [Q]
                pred_masks:   [Q, d, h, w] raw mask logits
        """
        if self.training:
            raise RuntimeError(
                'forward_one_click() is inference-only. '
                'Call model.eval() before one-click inference.'
            )

        # ------------------------------------------------------------
        # 1. Same input preprocessing as normal forward()
        # ------------------------------------------------------------
        images = [
            x['image'].to(self.device)
            for x in batched_inputs
        ]

        images = [
            (x - self.pixel_mean) / self.pixel_std
            for x in images
        ]

        images = torch.stack(images, dim=0)

        # ------------------------------------------------------------
        # 2. Same backbone and pixel decoder
        # ------------------------------------------------------------
        features = self.backbone(images)

        mask_features, _, multi_scale_features = (
            self.pixel_decoder.forward_features(features)
        )

        # ------------------------------------------------------------
        # 3. Same transformer decoder
        # ------------------------------------------------------------

        outputs, _ = self.transformer_decoder(
            multi_scale_features,
            mask_features,
            targets=None,
        )

        mask_cls_results = outputs['pred_logits']   # [B, Q, C+1]
        mask_pred_results = outputs['pred_masks']  # [B, Q, d, h, w]

        # ------------------------------------------------------------
        # 4. Build one score per query
        # ------------------------------------------------------------
        # Remove the final "no-object" class.
        class_prob = F.softmax(
            mask_cls_results,
            dim=-1,
        )[..., :-1]  # [B, Q, C]

        pred_scores, pred_classes = class_prob.max(
            dim=-1
        )  # both [B, Q]

        # ------------------------------------------------------------
        # 5. IMPORTANT:
        # Return native-resolution masks directly.
        # No instance_inference().
        # No F.interpolate() here.
        # ------------------------------------------------------------
        processed_results = []

        for b in range(images.shape[0]):
            processed_results.append(
                {
                    'pred_scores': pred_scores[b],
                    'pred_classes': pred_classes[b],
                    'pred_masks': mask_pred_results[b],
                }
            )

        return processed_results

    def prepare_targets(self, targets, images):
        d_pad, h_pad, w_pad = images.shape[-3:]  # get depth, height, width
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks  # (N, D, H, W)
            padded_masks = torch.zeros(
                (gt_masks.shape[0], d_pad, h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[
                :,
                : gt_masks.shape[1],
                : gt_masks.shape[2],
                : gt_masks.shape[3],
            ] = gt_masks
            new_targets.append(
                {
                    'labels': targets_per_image.gt_classes,
                    'masks': padded_masks,
                }
            )
        return new_targets


    def instance_inference(self, mask_cls, mask_pred):
        """
        Lightweight instance candidate selection for inference.

        Input
        -----
        mask_cls:
            [Q, C+1]

        mask_pred:
            [Q, d, h, w]

            IMPORTANT:
            These are LOW-RESOLUTION raw mask logits from the
            transformer decoder. Full-resolution interpolation is
            performed later in forward().

        Output
        ------
        {
            "pred_scores":  [K],
            "pred_classes": [K],
            "pred_masks":   [K, d, h, w]
        }

        Notes
        -----
        This function performs only mathematically safe filtering
        before full-resolution mask interpolation.

        It does NOT:
            - compute mask quality score
            - sigmoid full-resolution masks
            - perform conflict resolution
            - perform final instance filtering

        Those operations remain in the external postprocessing.
        """

        # ============================================================
        # 1. Classification probability
        # ============================================================

        # [Q, C]
        scores = F.softmax(
            mask_cls,
            dim=-1,
        )[:, :-1]

        num_queries, num_classes = scores.shape

        labels = (
            torch.arange(
                num_classes,
                device=mask_cls.device,
            )
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        # ============================================================
        # 2. Original classification top-k
        # ============================================================

        flat_scores = scores.flatten(0, 1)

        k = min(
            int(self.test_topk_per_image),
            int(flat_scores.numel()),
        )

        scores_per_image, topk_indices = (
            flat_scores.topk(
                k,
                sorted=False,
            )
        )

        labels_per_image = labels[
            topk_indices
        ]

        # flattened (query, class) index -> query index
        query_indices = (
            topk_indices // num_classes
        )

        # ============================================================
        # 3. Exact classification-score filtering
        #
        # Downstream:
        #
        #     final_score = class_score * mask_score
        #
        # and:
        #
        #     0 <= mask_score <= 1
        #
        # therefore:
        #
        #     final_score <= class_score
        #
        # So any class_score <= score_thresh can NEVER pass the
        # downstream final_score > score_thresh condition.
        # ============================================================

        fast_score_thresh = getattr(
            self,
            'fast_score_thresh',
            None,
        )

        if fast_score_thresh is not None:

            keep_cls = (
                scores_per_image
                > float(fast_score_thresh)
            )

            scores_per_image = (
                scores_per_image[keep_cls]
            )

            labels_per_image = (
                labels_per_image[keep_cls]
            )

            query_indices = (
                query_indices[keep_cls]
            )

        # ============================================================
        # 4. Gather ONLY surviving low-resolution masks
        # ============================================================

        mask_pred = mask_pred[
            query_indices
        ]

        # ============================================================
        # 5. Exact low-resolution foreground-possibility filtering
        #
        # External postprocessing eventually evaluates:
        #
        #     sigmoid(mask_logit) > mask_thresh
        #
        # Let:
        #
        #     logit_thresh = log(t / (1-t))
        #
        # Trilinear interpolation is a weighted combination of
        # neighboring values and therefore cannot create a value
        # greater than the maximum input value.
        #
        # Thus, if:
        #
        #     max(lowres mask logits) <= logit_thresh
        #
        # the upsampled mask can never contain foreground.
        # ============================================================

        fast_mask_thresh = getattr(
            self,
            'fast_mask_thresh',
            None,
        )

        if (
            fast_mask_thresh is not None
            and mask_pred.shape[0] > 0
        ):
            t = float(fast_mask_thresh)

            if t >= 1.0:
                # sigmoid(logit) > 1 can never happen
                keep_mask = torch.zeros(
                    mask_pred.shape[0],
                    dtype=torch.bool,
                    device=mask_pred.device,
                )

            elif t <= 0.0:
                # sigmoid(logit) > 0 for finite logits.
                # No useful filtering can be performed.
                keep_mask = torch.ones(
                    mask_pred.shape[0],
                    dtype=torch.bool,
                    device=mask_pred.device,
                )

            else:
                logit_thresh = math.log(
                    t / (1.0 - t)
                )

                lowres_max = (
                    mask_pred
                    .flatten(1)
                    .amax(dim=1)
                )

                keep_mask = (
                    lowres_max
                    > logit_thresh
                )

            scores_per_image = (
                scores_per_image[keep_mask]
            )

            labels_per_image = (
                labels_per_image[keep_mask]
            )

            mask_pred = (
                mask_pred[keep_mask]
            )

        # ============================================================
        # 6. Existing panoptic filtering
        # ============================================================

        if self.panoptic_on:

            keep = torch.zeros_like(
                scores_per_image,
                dtype=torch.bool,
            )

            thing_ids = set(
                self.metadata
                .thing_dataset_id_to_contiguous_id
                .values()
            )

            for i, lab in enumerate(
                labels_per_image
            ):
                keep[i] = (
                    int(lab.item())
                    in thing_ids
                )

            scores_per_image = (
                scores_per_image[keep]
            )

            labels_per_image = (
                labels_per_image[keep]
            )

            mask_pred = (
                mask_pred[keep]
            )

        return {
            'pred_scores': scores_per_image,
            'pred_classes': labels_per_image,
            'pred_masks': mask_pred,
        }



def build_maskformer_model_from_cfg(cfg):
    """
    Build MaskFormer without Detectron2.

    Equivalent to Detectron2 build_model(cfg) for the Windows inference path.
    """
    kwargs = MaskFormer.from_config(cfg)
    model = MaskFormer(**kwargs)

    # Same behavior as your original inference.py
    if getattr(model, 'sem_seg_postprocess_before_inference', False):
        model.sem_seg_postprocess_before_inference = False

    return model


def build_criterion_from_cfg(cfg):
    """
    Build training criterion without Detectron2.
    Used for future Windows fine-tuning.
    """
    class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
    dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
    mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
    no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
    from .modeling.criterion_win import SetCriterion
    from .modeling.matcher import HungarianMatcher

    matcher = HungarianMatcher(
        cost_class=class_weight,
        cost_mask=mask_weight,
        cost_dice=dice_weight,
        num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
    )

    weight_dict = {
        'loss_ce': class_weight,
        'loss_mask': mask_weight,
        'loss_dice': dice_weight,
    }

    if cfg.MODEL.MASK_FORMER.DN != 'no':
        weight_dict.update(
            {
                'loss_ce_dn': class_weight,
                'loss_mask_dn': mask_weight,
                'loss_dice_dn': dice_weight,
            }
        )

    enc_loss = getattr(cfg.MODEL.MASK_FORMER, 'ENC_LOSS', False)
    if enc_loss:
        weight_dict.update(
            {
                'loss_ce_enc': class_weight,
                'loss_mask_enc': mask_weight,
                'loss_dice_enc': dice_weight,
            }
        )

    deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
    if deep_supervision:
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=no_object_weight,
        losses=['labels', 'masks'],
        num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
        importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        dn=cfg.MODEL.MASK_FORMER.DN,
        dn_losses=['labels', 'masks'],
    )

    return criterion
