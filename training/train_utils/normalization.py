# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, rows, cols = extrinsics.shape
    device = extrinsics.device

    # Convert extrinsics to homogeneous form: (B, S, 4, 4)
    if (rows, cols) == (4, 4):
        extrinsics_homog = extrinsics
    elif (rows, cols) == (3, 4):
        pad_row = torch.zeros((B, S, 1, 4), device=device, dtype=extrinsics.dtype)
        extrinsics_homog = torch.cat([extrinsics, pad_row], dim=-2)
        extrinsics_homog[:, :, -1, -1] = 1.0
    else:
        raise ValueError(
            f"Expected extrinsics of shape (B,S,3,4) or (B,S,4,4), got {extrinsics.shape}"
        )

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    valid_mask = None
    if point_masks is not None:
        valid_mask = point_masks.to(torch.bool)

    safe_world_points = world_points
    if world_points is not None and valid_mask is not None:
        safe_world_points = torch.where(
            valid_mask.unsqueeze(-1),
            world_points,
            torch.zeros_like(world_points),
        )

    if safe_world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (
            safe_world_points
            @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
        ) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    if valid_mask is None and scale_by_points:
        logging.warning(
            "scale_by_points=True but point_masks is None; skipping scale normalization."
        )
        scale_by_points = False

    new_cam_points = cam_points
    if cam_points is not None and valid_mask is not None:
        new_cam_points = torch.where(
            valid_mask.unsqueeze(-1),
            cam_points,
            torch.zeros_like(cam_points),
        )

    new_depths = depths
    if depths is not None and valid_mask is not None:
        new_depths = torch.where(
            valid_mask,
            depths,
            torch.zeros_like(depths),
        )

    if scale_by_points and new_world_points is not None and valid_mask is not None:
        dist = new_world_points.norm(dim=-1)
        dist = torch.where(valid_mask, dist, torch.zeros_like(dist))
        dist_sum = dist.sum(dim=[1, 2, 3])
        valid_count = valid_mask.sum(dim=[1, 2, 3]).to(dist_sum.dtype)

        avg_scale = torch.where(
            valid_count > 0,
            dist_sum / torch.clamp(valid_count, min=1.0),
            torch.ones_like(dist_sum),
        )
        avg_scale = torch.clamp(avg_scale, min=1e-4, max=1e4)
        avg_scale = torch.where(
            torch.isfinite(avg_scale),
            avg_scale,
            torch.ones_like(avg_scale),
        )

        scale_world = avg_scale.view(-1, 1, 1, 1, 1)
        scale_extr = avg_scale.view(-1, 1, 1)
        scale_depth = avg_scale.view(-1, 1, 1, 1)

        new_world_points = torch.where(
            valid_mask.unsqueeze(-1),
            new_world_points / scale_world,
            new_world_points,
        )

        if new_cam_points is not None:
            new_cam_points = torch.where(
                valid_mask.unsqueeze(-1),
                new_cam_points / scale_world,
                new_cam_points,
            )

        if new_depths is not None:
            new_depths = torch.where(
                valid_mask,
                new_depths / scale_depth,
                new_depths,
            )

        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / scale_extr
    elif not scale_by_points:
        new_depths = depths
        new_cam_points = cam_points

    new_extrinsics = new_extrinsics[:, :, :3]  # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    if new_cam_points is not None:
        new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    if new_world_points is not None:
        new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    if new_depths is not None:
        new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_depths



