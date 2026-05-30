# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F

"""
Shape shorthand in this module:
    N: minibatch dimension size.
    P: number of points.
    D, H, W: Depth, Height, Width of the 3D volume.
"""


def point_sample_3d(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(1).unsqueeze(1)

    grid = 2.0 * point_coords - 1.0

    output = F.grid_sample(input, grid, **kwargs)

    if add_dim:
        output = output.squeeze(2).squeeze(2)

    return output


def get_uncertain_point_coords_with_randomness_3d(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):

    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    point_coords = torch.rand(
        num_boxes, num_sampled, 3, device=coarse_logits.device
    )

    point_logits = point_sample_3d(
        coarse_logits, point_coords, align_corners=False
    )

    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    _, idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1
    )

    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]

    point_coords = point_coords.view(-1, 3)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )

    if num_random_points > 0:
        random_coords = torch.rand(
            num_boxes, num_random_points, 3, device=coarse_logits.device
        )
        point_coords = torch.cat([point_coords, random_coords], dim=1)

    return point_coords
