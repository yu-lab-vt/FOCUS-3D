# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.misc_win import is_dist_avail_and_initialized
from .point_features_3d import (
    get_uncertain_point_coords_with_randomness_3d as get_uncertain_point_coords_with_randomness,
)
from .point_features_3d import (
    point_sample_3d as point_sample,
)


def get_world_size():
    """

    For normal single-process Windows training / inference, world size is 1.
    If you later use torch.distributed on Windows, this still works.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        dn='no',
        dn_losses=None,
        enc_losses=None,
        use_enc_loss=None,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses if dn_losses is not None else losses
        self.enc_losses = enc_losses if enc_losses is not None else losses

        if use_enc_loss is None:
            self.use_enc_loss = any(
                k.endswith('_enc') for k in weight_dict.keys()
            )
        else:
            self.use_enc_loss = bool(use_enc_loss)

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']

        src_idx = (
            src_idx[0].to(src_masks.device),
            src_idx[1].to(src_masks.device),
        )
        tgt_idx = (
            tgt_idx[0].to(src_masks.device),
            tgt_idx[1].to(src_masks.device),
        )

        if src_idx[1].numel() > 0:
            max_idx = src_idx[1].max().item()
            min_idx = src_idx[1].min().item()
            assert max_idx < src_masks.shape[1], (
                f'src_idx max {max_idx} >= src_masks.shape[1] {src_masks.shape[1]}'
            )
        else:
            print('[DEBUG] src_idx is empty')

        src_masks = src_masks[src_idx]
        if src_masks.shape[0] == 0:
            return {
                'loss_mask': src_masks.sum() * 0.0,
                'loss_dice': src_masks.sum() * 0.0,
            }
        # torch.cuda.synchronize()

        masks = [t['masks'] for t in targets]
        target_masks = torch.cat(masks, dim=0)
        target_masks = target_masks.to(src_masks.device)

        if tgt_idx[1].numel() > 0:
            max_tgt = tgt_idx[1].max().item()
            min_tgt = tgt_idx[1].min().item()
            assert max_tgt < target_masks.shape[0], (
                f'tgt_idx max {max_tgt} >= target_masks.shape[0] {target_masks.shape[0]}'
            )

        batch_idx, per_img_idx = tgt_idx
        cumulative_sizes = [0]
        for m in masks:
            cumulative_sizes.append(cumulative_sizes[-1] + m.shape[0])
        cumulative_sizes = cumulative_sizes[:-1]
        cumulative_sizes = torch.tensor(
            cumulative_sizes, device=target_masks.device
        )
        global_idx = cumulative_sizes[batch_idx] + per_img_idx

        assert (global_idx < target_masks.shape[0]).all(), (
            'global_idx out of range'
        )

        target_masks = target_masks[global_idx]
        # torch.cuda.synchronize()

        src_masks = src_masks[:, None]  # (num_matched, 1, D', H', W')
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = point_sample(
                target_masks.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            'loss_mask': sigmoid_ce_loss_jit(
                point_logits, point_labels, num_masks
            ),
            'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        scalar = mask_dict['scalar']
        pad_size = mask_dict['pad_size']
        assert pad_size % scalar == 0

        single_pad = pad_size // scalar
        return output_known_lbs_bboxes, single_pad, scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k not in ['aux_outputs', 'enc_outputs']
        }
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks)
            )
        if (
            self.use_enc_loss
            and 'enc_outputs' in outputs
            and outputs['enc_outputs'] is not None
        ):
            enc_outputs = outputs['enc_outputs']

            # enc_outputs should contain:
            #   pred_logits: [B, Q, C+1]
            #   pred_masks:  [B, Q, D, H, W]
            enc_indices = self.matcher(enc_outputs, targets)

            for loss in self.enc_losses:
                l_dict = self.get_loss(
                    loss,
                    enc_outputs,
                    targets,
                    enc_indices,
                    num_masks,
                )
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if self.dn != 'no' and mask_dict is not None:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                mask_dict
            )

            exc_idx = []
            default_device = next(iter(outputs.values())).device

            for i in range(len(targets)):
                num_tgt = len(targets[i]['labels'])

                if num_tgt > 0:
                    device = targets[i]['labels'].device

                    t = torch.arange(num_tgt, device=device).long()
                    t = t.unsqueeze(0).repeat(scalar, 1)

                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.arange(scalar, device=device).long().unsqueeze(1)
                        * single_pad
                        + t
                    ).flatten()

                    exc_idx.append((output_idx, tgt_idx))
                else:
                    output_idx = torch.tensor([], device=default_device).long()
                    tgt_idx = torch.tensor([], device=default_device).long()
                    exc_idx.append((output_idx, tgt_idx))

            for loss in self.dn_losses:
                l_dict = self.get_loss(
                    loss,
                    output_known_lbs_bboxes,
                    targets,
                    exc_idx,
                    num_masks * scalar,
                )
                l_dict = {k + '_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.dn != 'no' and mask_dict is not None:
                    if i < len(output_known_lbs_bboxes['aux_outputs']):
                        dn_aux_outputs = output_known_lbs_bboxes[
                            'aux_outputs'
                        ][i]

                        for loss in self.dn_losses:
                            l_dict = self.get_loss(
                                loss,
                                dn_aux_outputs,
                                targets,
                                exc_idx,
                                num_masks * scalar,
                            )
                            l_dict = {
                                k + f'_dn_{i}': v for k, v in l_dict.items()
                            }
                            losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = [
            f'matcher: {self.matcher.__repr__(_repr_indent=8)}',
            f'losses: {self.losses}',
            f'weight_dict: {self.weight_dict}',
            f'num_classes: {self.num_classes}',
            f'eos_coef: {self.eos_coef}',
            f'num_points: {self.num_points}',
            f'oversample_ratio: {self.oversample_ratio}',
            f'importance_sample_ratio: {self.importance_sample_ratio}',
        ]
        _repr_indent = 4
        lines = [head] + [' ' * _repr_indent + line for line in body]
        return '\n'.join(lines)
