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
    """
    针对 3D point_coords 的 F.grid_sample 包装器。
    对齐 Detectron2 官方 2D point_sample 的逻辑，确保 3D 采样的高效与准确。

    Args:
        input (Tensor): 形状为 (N, C, D, H, W) 的 5D Tensor。
        point_coords (Tensor): 形状为 (N, P, 3) 或 (N, Dgrid, Hgrid, Wgrid, 3) 的坐标。
                               坐标范围应在 [0, 1] 之间。
        **kwargs: 传递给 grid_sample 的参数（推荐 align_corners=False, mode='bilinear'）。

    Returns:
        output (Tensor): 形状为 (N, C, P) 或 (N, C, Dgrid, Hgrid, Wgrid)。
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        # 修改点：(N, P, 3) -> (N, 1, 1, P, 3)
        # 将 P 放在 Width 维度（最后一维），这在 grid_sample 内部通常对应最快的内存访问
        # 同时也最符合官方对“点集采样”在空间维度排布的惯例
        point_coords = point_coords.unsqueeze(1).unsqueeze(1)

    # 将 [0, 1] 映射到 [-1, 1]
    # 重要提醒：grid_sample 的 3D 坐标顺序要求为 (x, y, z)，
    # 物理意义严格对应输入 Tensor 的 (Width, Height, Depth)
    grid = 2.0 * point_coords - 1.0

    # 执行 5D 采样
    output = F.grid_sample(input, grid, **kwargs)

    if add_dim:
        # 对应地，挤压掉 Dgrid 和 Hgrid 维度 (即 index 2 和 3)
        # (N, C, 1, 1, P) -> (N, C, P)
        output = output.squeeze(2).squeeze(2)

    return output


def get_uncertain_point_coords_with_randomness_3d(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    在 3D 空间中基于不确定性进行点采样（PointRend 3D 逻辑）。

    Args:
        coarse_logits (Tensor): (N, C, D, H, W) 或 (N, 1, D, H, W)。
        uncertainty_func: 输入 (N, C, P) 返回 (N, 1, P) 的不确定性计算函数。
        num_points (int): 最终采样的点数 P。
        oversample_ratio (int): 过采样倍数。
        importance_sample_ratio (float): 重要性采样的比例。

    Returns:
        point_coords (Tensor): (N, P, 3) 的 [0, 1] 规范化坐标。
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    # 1. 在 3D 空间均匀随机撒点 (N, P_oversample, 3)
    point_coords = torch.rand(
        num_boxes, num_sampled, 3, device=coarse_logits.device
    )

    # 2. 采样预测值以计算不确定性
    point_logits = point_sample_3d(
        coarse_logits, point_coords, align_corners=False
    )

    # 3. 计算不确定性并筛选最不确定的点
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    # 取前 K 个最不确定的点
    _, idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1
    )

    # 索引对齐
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]

    # 提取坐标
    point_coords = point_coords.view(-1, 3)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )

    # 4. 补充剩余的随机点
    if num_random_points > 0:
        random_coords = torch.rand(
            num_boxes, num_random_points, 3, device=coarse_logits.device
        )
        point_coords = torch.cat([point_coords, random_coords], dim=1)

    return point_coords
