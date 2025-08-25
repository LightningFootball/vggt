# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        """
        VGGT 模型的前向传播。

        参数 (Args):
            images (torch.Tensor): 输入图像，形状为 [S, 3, H, W] 或 [B, S, 3, H, W]，数值范围在 [0, 1]。
                B: 批大小（batch size），S: 序列长度（sequence length），3: RGB 通道，H: 高度，W: 宽度
            query_points (torch.Tensor, 可选): 用于跟踪的查询点，像素坐标形式。
                形状为 [N, 2] 或 [B, N, 2]，其中 N 表示查询点的数量。
                默认值: None

        返回 (Returns):
            dict: 包含以下预测结果的字典：
                - pose_enc (torch.Tensor): 相机位姿编码，形状为 [B, S, 9]（来自最后一次迭代）
                - depth (torch.Tensor): 预测的深度图，形状为 [B, S, H, W, 1]
                - depth_conf (torch.Tensor): 深度预测的置信度分数，形状为 [B, S, H, W]
                - world_points (torch.Tensor): 每个像素的三维世界坐标，形状为 [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): 世界坐标的置信度分数，形状为 [B, S, H, W]
                - images (torch.Tensor): 原始输入图像，保留下来用于可视化

                如果提供了 query_points，还会包含：
                - track (torch.Tensor): 点的跟踪结果，形状为 [B, S, N, 2]（来自最后一次迭代），像素坐标形式
                - vis (torch.Tensor): 跟踪点的可见性分数，形状为 [B, S, N]
                - conf (torch.Tensor): 跟踪点的置信度分数，形状为 [B, S, N]
        """

        # If without batch dimension, add it
        # 如果输入图像没有批次维度，则添加一个批次维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        # 如果提供了 query_points，但没有批次维度，则添加一个批次维度
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # 调用聚合器，获取聚合后的 tokens 列表和每个 patch 的起始索引
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        # 初始化一个空字典，用于存储预测结果
        predictions = {}

        # 禁用自动混合精度
        with torch.cuda.amp.autocast(enabled=False):
            # 如果启用了相机头，则调用相机头，获取相机位姿编码
            if self.camera_head is not None:
                # 调用相机头，获取相机位姿编码
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                # 将最后一次迭代的相机位姿编码存储到 predictions 字典中
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                # 将所有迭代的相机位姿编码存储到 predictions 字典中
                predictions["pose_enc_list"] = pose_enc_list
                
            # 如果启用了深度头，则调用深度头，获取深度图和深度预测的置信度分数
            if self.depth_head is not None:
                # 调用深度头，获取深度图和深度预测的置信度分数
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # 如果启用了点头，则调用点头，获取三维点云和三维点云的置信度分数
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # 如果启用了跟踪头，并且提供了查询点，则调用跟踪头，获取跟踪结果、可见性和置信度分数
        if self.track_head is not None and query_points is not None:
            # 调用跟踪头，获取跟踪结果、可见性和置信度分数
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # 如果当前不是训练模式，则将原始输入图像存储到 predictions 字典中，用于可视化
        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

