#!/usr/bin/env python3
"""
Evaluate VGGT base and LoRA checkpoints on KITTI-360 with building-focused metrics.

This script compares the original VGGT model against every checkpoint stored under a
LoRA training run directory. It computes depth/normal/planarity/photometric/pose metrics
for the full image as well as building-masked regions, and writes results under
`evaluate/kitti360_b`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Extend sys.path so imports work when running as a script
import sys
import importlib
import importlib.machinery
import types

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Backward compatibility: training.dataset modules expect `data` package
data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(REPO_ROOT / "training" / "data")]
data_pkg.__spec__ = importlib.machinery.ModuleSpec("data", loader=None, is_package=True)
sys.modules["data"] = data_pkg
for submodule in ("dataset_util", "base_dataset", "dynamic_dataloader", "worker_fn"):
    full_name = f"training.data.{submodule}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError:
        continue
    sys.modules[f"data.{submodule}"] = module

from training.data.datasets.kitti360 import (  # noqa: E402
    KITTI360Dataset,
    SEMANTIC_CLASSES,
)
from training.data.dataset_util import (  # noqa: E402
    crop_image_depth_and_intrinsic_by_pp,
    read_image_cv2,
    resize_image_depth_and_intrinsic,
)
from training.lora_utils import apply_lora_to_model  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


logger = logging.getLogger("evaluate_kitti360_buildings")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# --------------------------------------------------------------------------------------
# Utility containers
# --------------------------------------------------------------------------------------


@dataclass
class SequenceSample:
    seq_name: str
    frame_indices: List[int]
    images: torch.Tensor  # [S, 3, H, W], float32 in [0, 1]
    gt_depths: torch.Tensor  # [S, H, W]
    gt_masks: torch.Tensor  # [S, H, W] bool
    building_masks: torch.Tensor  # [S, H, W] bool
    non_building_masks: torch.Tensor  # [S, H, W] bool
    semantics: torch.Tensor  # [S, H, W] int16
    gt_extrinsics: torch.Tensor  # [S, 3, 4]
    gt_intrinsics: torch.Tensor  # [S, 3, 3]

    def to_device(self, device: torch.device) -> "SequenceSample":
        return SequenceSample(
            seq_name=self.seq_name,
            frame_indices=self.frame_indices,
            images=self.images.to(device, non_blocking=True),
            gt_depths=self.gt_depths.to(device, non_blocking=True),
            gt_masks=self.gt_masks.to(device, non_blocking=True),
            building_masks=self.building_masks.to(device, non_blocking=True),
            non_building_masks=self.non_building_masks.to(device, non_blocking=True),
            semantics=self.semantics.to(device, non_blocking=True),
            gt_extrinsics=self.gt_extrinsics.to(device, non_blocking=True),
            gt_intrinsics=self.gt_intrinsics.to(device, non_blocking=True),
        )


@dataclass
class AggregatedMetrics:
    values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, key: str, value: float) -> None:
        if math.isnan(value) or math.isinf(value):
            return
        self.values[key].append(float(value))

    def extend(self, prefix: str, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.add(f"{prefix}.{key}", value)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for key, vals in self.values.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }
        return summary


# --------------------------------------------------------------------------------------
# Geometric helpers
# --------------------------------------------------------------------------------------


def crop_by_principal_point(
    image: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    target_shape: np.ndarray,
    filepath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop arrays around the principal point to match target_shape."""
    target_shape = np.asarray(target_shape, dtype=np.int32)
    intrinsic_before = np.copy(intrinsic)

    image_c, depth_c, intrinsic_after, _ = crop_image_depth_and_intrinsic_by_pp(
        np.copy(image),
        np.copy(depth),
        np.copy(intrinsic),
        target_shape,
        track=None,
        filepath=filepath,
        strict=True,
    )

    start_x = int(round(intrinsic_before[1, 2] - intrinsic_after[1, 2]))
    start_y = int(round(intrinsic_before[0, 2] - intrinsic_after[0, 2]))
    end_x = start_x + image_c.shape[0]
    end_y = start_y + image_c.shape[1]

    semantic_c = semantic[start_x:end_x, start_y:end_y]
    mask_c = mask[start_x:end_x, start_y:end_y]

    pad_h = target_shape[0] - image_c.shape[0]
    pad_w = target_shape[1] - image_c.shape[1]
    if pad_h > 0 or pad_w > 0:
        image_c = np.pad(image_c, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        depth_c = np.pad(depth_c, ((0, pad_h), (0, pad_w)), mode="constant")
        semantic_c = np.pad(semantic_c, ((0, pad_h), (0, pad_w)), mode="constant")
        mask_c = np.pad(mask_c, ((0, pad_h), (0, pad_w)), mode="constant")

    return image_c, depth_c, semantic_c, mask_c, intrinsic_after


def compute_building_masks(
    semantic: np.ndarray,
    valid_mask: np.ndarray,
    building_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    building_mask = (semantic == building_id) & valid_mask
    non_building_mask = (~building_mask) & valid_mask
    return building_mask.astype(np.bool_), non_building_mask.astype(np.bool_)


def depth_to_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Convert depth map to 3D points in camera coordinates."""
    device = depth.device
    h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing="ij",
    )

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points = torch.stack((x, y, z), dim=-1)
    return points


def compute_normals_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Estimate surface normals from depth."""
    # Compute 3D points then finite differences
    points = depth_to_points(depth, intrinsics)
    dzdx = F.pad(points[1:, :, :] - points[:-1, :, :], (0, 0, 0, 0, 1, 0))
    dzdy = F.pad(points[:, 1:, :] - points[:, :-1, :], (0, 0, 1, 0, 0, 0))
    normals = torch.cross(dzdx, dzdy, dim=-1)
    norms = torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-8
    normals = normals / norms
    return normals


def plane_fit(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit plane using SVD. Returns (centroid, normal)."""
    centroid = points.mean(dim=0, keepdim=True)
    demeaned = points - centroid
    _, _, vh = torch.linalg.svd(demeaned, full_matrices=False)
    normal = vh[-1]
    normal = normal / (torch.linalg.norm(normal) + 1e-8)
    return centroid.squeeze(0), normal


def plane_residuals(
    points: torch.Tensor,
    centroid: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    distances = torch.abs((points - centroid) @ normal)
    return distances


def photometric_reprojection_error(
    img_src: torch.Tensor,
    img_tgt: torch.Tensor,
    depth_src: torch.Tensor,
    extr_src: torch.Tensor,
    extr_tgt: torch.Tensor,
    intr_src: torch.Tensor,
    intr_tgt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Compute photometric error by warping source image into target view."""
    device = img_src.device
    h, w = depth_src.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth_src.dtype),
        torch.arange(w, device=device, dtype=depth_src.dtype),
        indexing="ij",
    )

    fx, fy = intr_src[0, 0], intr_src[1, 1]
    cx, cy = intr_src[0, 2], intr_src[1, 2]

    z = depth_src
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    ones = torch.ones_like(z)
    pts_cam = torch.stack((x, y, z, ones), dim=-1)  # [H, W, 4]

    # Camera to world for source: extrinsics are world->cam, so invert
    extr_src_4x4 = torch.eye(4, device=device, dtype=z.dtype)
    extr_src_4x4[:3, :] = extr_src
    extr_tgt_4x4 = torch.eye(4, device=device, dtype=z.dtype)
    extr_tgt_4x4[:3, :] = extr_tgt

    world = torch.linalg.solve(extr_src_4x4, pts_cam.reshape(-1, 4).T).T
    pts_tgt = (extr_tgt_4x4 @ world.T).T
    pts_tgt = pts_tgt[:, :3]

    x_t = pts_tgt[:, 0]
    y_t = pts_tgt[:, 1]
    z_t = pts_tgt[:, 2].clamp(min=1e-4)

    fx_t, fy_t = intr_tgt[0, 0], intr_tgt[1, 1]
    cx_t, cy_t = intr_tgt[0, 2], intr_tgt[1, 2]
    u_proj = fx_t * x_t / z_t + cx_t
    v_proj = fy_t * y_t / z_t + cy_t

    u_norm = (u_proj / (w - 1)) * 2 - 1
    v_norm = (v_proj / (h - 1)) * 2 - 1
    grid = torch.stack((u_norm, v_norm), dim=-1).reshape(h, w, 2)

    warped = F.grid_sample(
        img_tgt.unsqueeze(0),
        grid.unsqueeze(0),
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    ).squeeze(0)

    diff = torch.abs(img_src - warped)
    valid = (
        (grid[..., 0] >= -1)
        & (grid[..., 0] <= 1)
        & (grid[..., 1] >= -1)
        & (grid[..., 1] <= 1)
        & mask
    )

    if valid.sum() == 0:
        return float("nan"), float("nan")

    photometric = diff[:, valid].mean().item()
    # Inlier ratio: pixels with error < 0.05 (roughly 12/255)
    inlier = (diff[:, valid].mean(dim=0) < 0.05).float().mean().item()
    return photometric, inlier


def rotation_error(pred_R: torch.Tensor, gt_R: torch.Tensor) -> float:
    """Angular error in degrees between two rotation matrices."""
    rel = pred_R @ gt_R.transpose(-1, -2)
    trace = rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    trace = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    angle = torch.rad2deg(torch.arccos(trace))
    return angle.item()


def translation_error(pred_t: torch.Tensor, gt_t: torch.Tensor) -> float:
    """Euclidean translation error."""
    return torch.linalg.norm(pred_t - gt_t).item()


def relative_pose_error(
    pred_extrinsics: torch.Tensor,
    gt_extrinsics: torch.Tensor,
) -> Tuple[float, float]:
    """
    Relative pose error between consecutive frames.
    Returns (translational, rotational) errors averaged over pairs.
    """
    def to_hom(extr: torch.Tensor) -> torch.Tensor:
        mat = torch.eye(4, device=extr.device, dtype=extr.dtype)
        mat[:3, :3] = extr[:, :3]
        mat[:3, 3] = extr[:, 3]
        return mat

    trans_errors = []
    rot_errors = []
    for i in range(pred_extrinsics.shape[0] - 1):
        pred_rel = torch.linalg.solve(to_hom(pred_extrinsics[i]), to_hom(pred_extrinsics[i + 1]))
        gt_rel = torch.linalg.solve(to_hom(gt_extrinsics[i]), to_hom(gt_extrinsics[i + 1]))

        t_err = torch.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3]).item()
        r_err = rotation_error(pred_rel[:3, :3], gt_rel[:3, :3])
        trans_errors.append(t_err)
        rot_errors.append(r_err)

    if not trans_errors:
        return float("nan"), float("nan")

    return float(np.mean(trans_errors)), float(np.mean(rot_errors))


# --------------------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------------------


def load_yaml_config(config_name: str) -> Dict:
    cfg_path = Path("training/config") / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_common_config(img_size: int, patch_size: int) -> object:
    """Create a minimal namespace mimicking Hydra common_config."""
    from types import SimpleNamespace

    augs = SimpleNamespace(
        scales=None,
        aspects=[1.0, 1.0],
        cojitter=False,
        cojitter_ratio=0.0,
        color_jitter=None,
        gray_scale=False,
        gau_blur=False,
    )
    common = SimpleNamespace(
        img_size=img_size,
        patch_size=patch_size,
        augs=augs,
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        training=False,
        inside_random=False,
        load_depth=True,
        get_nearby=False,
        debug=False,
        repeat_batch=False,
        allow_duplicate_img=False,
        fix_img_num=-1,
        fix_aspect_ratio=1.0,
        track_num=0,
        load_track=False,
        max_img_per_gpu=48,
        img_nums=[8, 8],
    )
    return common


def prepare_sequence_indices(
    frames: List[Dict],
    seq_len: int,
    stride: int,
) -> Dict[str, List[List[int]]]:
    sequences: Dict[str, List[List[int]]] = defaultdict(list)
    by_sequence: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)

    for idx, frame in enumerate(frames):
        by_sequence[frame["sequence"]].append((idx, frame))

    for seq_name, items in by_sequence.items():
        items.sort(key=lambda tup: tup[1]["frame_idx"])
        indices = [idx for idx, _ in items]
        for start in range(0, len(indices) - seq_len + 1, stride):
            window = indices[start : start + seq_len]
            if len(window) == seq_len:
                sequences[seq_name].append(window)

    return sequences


def load_depth_and_semantic(
    dataset: KITTI360Dataset,
    frame: Dict,
    frame_idx: int,
    target_shape: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load RGB/depth/semantic/mask with resizing and cropping applied.

    This function follows the same preprocessing pipeline as training (BaseDataset.process_one_image):
    1. First crop by principal point (non-strict) to center the principal point
    2. Resize to target_shape + safe_bound (slightly larger than target)
    3. Second crop by principal point (strict) to exact target_shape

    This ensures evaluation data matches training distribution and avoids padding artifacts.
    """
    image = read_image_cv2(frame["image_path"])
    if image is None:
        raise FileNotFoundError(f"Failed to read image {frame['image_path']}")

    original_size = np.array(image.shape[:2])

    depth_result = dataset._load_precomputed_depth(
        frame["sequence"],
        frame["frame_idx"],
        tuple(original_size),
    )
    if depth_result is None:
        depth_map, depth_mask = dataset._project_lidar_to_depth(frame, frame_idx, original_size)
    else:
        depth_map, depth_mask = depth_result

    if frame["semantic_path"] and os.path.exists(frame["semantic_path"]):
        semantic = cv2.imread(frame["semantic_path"], cv2.IMREAD_GRAYSCALE)
        if semantic is None:
            semantic = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        semantic = np.zeros_like(depth_map, dtype=np.uint8)

    intrinsics = dataset.calib["K"].astype(np.float32)
    extrinsics = np.linalg.inv(frame["pose"]).astype(np.float32)

    # Step 1: First crop by principal point (non-strict mode)
    # This centers the principal point in the image
    image, depth_map, intrinsics, _ = crop_image_depth_and_intrinsic_by_pp(
        np.copy(image),
        np.copy(depth_map),
        np.copy(intrinsics),
        original_size,
        track=None,
        filepath=frame.get("image_path"),
        strict=False,
    )

    # Also crop semantic and mask
    cx_before = dataset.calib["K"][1, 2]
    cy_before = dataset.calib["K"][0, 2]
    cx_after = intrinsics[1, 2]
    cy_after = intrinsics[0, 2]
    start_x = int(round(cx_before - cx_after))
    start_y = int(round(cy_before - cy_after))
    end_x = start_x + image.shape[0]
    end_y = start_y + image.shape[1]
    semantic = semantic[start_x:end_x, start_y:end_y]
    depth_mask = depth_mask[start_x:end_x, start_y:end_y]

    # Update original size after first crop
    original_size = np.array(image.shape[:2])

    # Step 2: Resize to slightly larger than target (target_shape + safe_bound)
    # This ensures we have enough pixels for the final crop without padding
    if dataset.rescale:
        image, depth_map, intrinsics, _ = resize_image_depth_and_intrinsic(
            image=image,
            depth_map=depth_map,
            intrinsic=intrinsics,
            target_shape=target_shape,
            original_size=original_size,
            rescale_aug=False,  # No random augmentation during evaluation
            safe_bound=4,  # Same as training default
        )
        depth_mask = cv2.resize(
            depth_mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        semantic = cv2.resize(
            semantic,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Step 3: Final crop by principal point (strict mode) to exact target_shape
    # Since image is now larger than target_shape, this should not require padding
    image, depth_map, semantic, depth_mask, intrinsics = crop_by_principal_point(
        image,
        depth_map,
        semantic,
        depth_mask,
        intrinsics,
        target_shape,
        filepath=frame.get("image_path"),
    )

    # Convert to tensors
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    depth_tensor = torch.from_numpy(depth_map).float()
    mask_tensor = torch.from_numpy(depth_mask).bool()
    semantic_tensor = torch.from_numpy(semantic).to(torch.int16)
    intrinsics_tensor = torch.from_numpy(intrinsics).float()
    extrinsics_tensor = torch.from_numpy(extrinsics[:3, :]).float()

    building_mask, non_building_mask = compute_building_masks(
        semantic,
        depth_mask,
        SEMANTIC_CLASSES["building"],
    )

    building_mask_tensor = torch.from_numpy(building_mask)
    non_building_mask_tensor = torch.from_numpy(non_building_mask)

    return (
        image_tensor,
        depth_tensor,
        mask_tensor,
        building_mask_tensor,
        non_building_mask_tensor,
        semantic_tensor,
        extrinsics_tensor,
        intrinsics_tensor,
    )


def build_samples(
    dataset: KITTI360Dataset,
    seq_len: int,
    stride: int,
    max_sequences: Optional[int],
) -> List[SequenceSample]:
    target_shape = dataset.get_target_shape(1.0)
    sequences = prepare_sequence_indices(dataset.frames, seq_len, stride)
    samples: List[SequenceSample] = []

    for seq_name, windows in sequences.items():
        for frame_indices in windows:
            tensors = [
                load_depth_and_semantic(dataset, dataset.frames[idx], idx, target_shape)
                for idx in frame_indices
            ]
            imgs, depths, masks, b_masks, nb_masks, semantics, extrinsics, intrinsics = zip(*tensors)  # type: ignore[arg-type]

            sample = SequenceSample(
                seq_name=f"{seq_name}",
                frame_indices=[dataset.frames[idx]["frame_idx"] for idx in frame_indices],
                images=torch.stack(list(imgs)),
                gt_depths=torch.stack(list(depths)),
                gt_masks=torch.stack(list(masks)),
                building_masks=torch.stack(list(b_masks)),
                non_building_masks=torch.stack(list(nb_masks)),
                semantics=torch.stack(list(semantics)),
                gt_extrinsics=torch.stack(list(extrinsics)),
                gt_intrinsics=torch.stack(list(intrinsics)),
            )
            samples.append(sample)

            if max_sequences is not None and len(samples) >= max_sequences:
                return samples

    return samples


def sample_generator(
    dataset: KITTI360Dataset,
    seq_len: int,
    stride: int,
    max_sequences: Optional[int] = None,
    stratified_sampling: bool = False,
    num_workers: int = 8,
    prefetch_factor: int = 2,
):
    """
    Memory-efficient generator that yields samples on-demand with parallel data loading.

    Args:
        dataset: KITTI360Dataset instance
        seq_len: Number of frames per sequence
        stride: Stride between sequences
        max_sequences: Maximum number of sequences to generate
        stratified_sampling: If True, sample proportionally from rich/mixed/road buckets
        num_workers: Number of parallel data loading threads (default: 4)
        prefetch_factor: Number of samples to prefetch per worker (default: 2)
    """
    target_shape = dataset.get_target_shape(1.0)
    sequences = prepare_sequence_indices(dataset.frames, seq_len, stride)

    # Build list of all (seq_name, window) pairs
    all_windows = []
    for seq_name, windows in sequences.items():
        for frame_indices in windows:
            all_windows.append((seq_name, frame_indices))

    # If stratified sampling is enabled and we have bucket info
    if stratified_sampling and max_sequences is not None and hasattr(dataset, 'bucket_building_rich'):
        # Classify each window by its center frame's bucket
        rich_windows = []
        mixed_windows = []
        road_windows = []

        for seq_name, frame_indices in all_windows:
            center_idx = frame_indices[len(frame_indices) // 2]
            if center_idx in dataset.bucket_building_rich:
                rich_windows.append((seq_name, frame_indices))
            elif center_idx in dataset.bucket_mixed:
                mixed_windows.append((seq_name, frame_indices))
            elif center_idx in dataset.bucket_road:
                road_windows.append((seq_name, frame_indices))

        # Sample proportionally from each bucket
        # Use same weights as training: rich=0.5, mixed=0.3, road=0.2
        n_rich = int(max_sequences * 0.5)
        n_mixed = int(max_sequences * 0.3)
        n_road = max_sequences - n_rich - n_mixed

        # Random sampling from each bucket
        import random
        sampled_windows = []
        if len(rich_windows) > 0:
            sampled_windows.extend(random.sample(rich_windows, min(n_rich, len(rich_windows))))
        if len(mixed_windows) > 0:
            sampled_windows.extend(random.sample(mixed_windows, min(n_mixed, len(mixed_windows))))
        if len(road_windows) > 0:
            sampled_windows.extend(random.sample(road_windows, min(n_road, len(road_windows))))

        # Shuffle to mix buckets
        random.shuffle(sampled_windows)

        logger.info(f"Stratified sampling: {len(sampled_windows)} sequences "
                   f"(rich={min(n_rich, len(rich_windows))}, "
                   f"mixed={min(n_mixed, len(mixed_windows))}, "
                   f"road={min(n_road, len(road_windows))})")

        all_windows = sampled_windows
    elif max_sequences is not None and len(all_windows) > max_sequences:
        # Simple random sampling without stratification
        import random
        all_windows = random.sample(all_windows, max_sequences)
        logger.info(f"Random sampling: {len(all_windows)} sequences from total")

    # Parallel data loading function
    def load_window(args):
        seq_name, frame_indices = args
        tensors = [
            load_depth_and_semantic(dataset, dataset.frames[idx], idx, target_shape)
            for idx in frame_indices
        ]
        imgs, depths, masks, b_masks, nb_masks, semantics, extrinsics, intrinsics = zip(*tensors)  # type: ignore[arg-type]

        return SequenceSample(
            seq_name=f"{seq_name}",
            frame_indices=[dataset.frames[idx]["frame_idx"] for idx in frame_indices],
            images=torch.stack(list(imgs)),
            gt_depths=torch.stack(list(depths)),
            gt_masks=torch.stack(list(masks)),
            building_masks=torch.stack(list(b_masks)),
            non_building_masks=torch.stack(list(nb_masks)),
            semantics=torch.stack(list(semantics)),
            gt_extrinsics=torch.stack(list(extrinsics)),
            gt_intrinsics=torch.stack(list(intrinsics)),
        )

    # Use ThreadPoolExecutor for parallel data loading with prefetching
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit prefetch_factor * num_workers tasks ahead
            futures = []
            window_iter = iter(all_windows)

            # Initial prefetch
            for _ in range(min(prefetch_factor * num_workers, len(all_windows))):
                try:
                    window = next(window_iter)
                    futures.append(executor.submit(load_window, window))
                except StopIteration:
                    break

            count = 0
            while futures:
                # Get next completed sample
                future = futures.pop(0)
                sample = future.result()
                yield sample

                count += 1
                if max_sequences is not None and count >= max_sequences:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return

                # Submit next task to maintain prefetch queue
                try:
                    window = next(window_iter)
                    futures.append(executor.submit(load_window, window))
                except StopIteration:
                    pass
    else:
        # Fallback to sequential loading
        count = 0
        for seq_name, frame_indices in all_windows:
            sample = load_window((seq_name, frame_indices))
            yield sample

            count += 1
            if max_sequences is not None and count >= max_sequences:
                return


# --------------------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------------------


def align_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mask.sum() == 0:
        return pred

    if mode in ("scale", "median"):
        scale = torch.median(gt[mask]) / (torch.median(pred[mask]) + 1e-6)
        return pred * scale
    if mode == "scale_shift":
        p = pred[mask]
        g = gt[mask]
        pm = p.mean()
        gm = g.mean()
        var_p = ((p - pm) ** 2).mean()
        if var_p <= 1e-6:
            return pred
        cov = ((p - pm) * (g - gm)).mean()
        s = cov / var_p
        b = gm - s * pm
        return pred * s + b
    return pred


def depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    if mask.sum() == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
            "rmse_log": float("nan"),
            "si_log": float("nan"),
        }
    p = pred[mask]
    g = gt[mask]
    diff = p - g
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    abs_rel = (diff.abs() / (g + 1e-6)).mean().item()
    sq_rel = ((diff ** 2) / (g + 1e-6)).mean().item()
    thresh = torch.maximum(p / (g + 1e-6), g / (p + 1e-6))
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()
    rmse_log = torch.sqrt(((torch.log(p + 1e-6) - torch.log(g + 1e-6)) ** 2).mean()).item()
    si_log = torch.sqrt(
        ((torch.log(p + 1e-6) - torch.log(g + 1e-6)) ** 2).mean()
        - ((torch.log(p + 1e-6) - torch.log(g + 1e-6)).mean()) ** 2
    ).item()
    return {
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "rmse_log": rmse_log,
        "si_log": si_log,
    }


def normals_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Dict[str, float]:
    if mask.sum() == 0:
        return {
            "angular_mean": float("nan"),
            "angular_median": float("nan"),
            "inlier_11_25": float("nan"),
            "inlier_22_5": float("nan"),
            "inlier_30": float("nan"),
        }
    pred_normals = compute_normals_from_depth(pred_depth, intrinsics)
    gt_normals = compute_normals_from_depth(gt_depth, intrinsics)

    pred_normals = pred_normals[mask]
    gt_normals = gt_normals[mask]

    dot = (pred_normals * gt_normals).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(dot))

    angular_mean = angles.mean().item()
    angular_median = angles.median().item()
    inlier_11 = (angles < 11.25).float().mean().item()
    inlier_22 = (angles < 22.5).float().mean().item()
    inlier_30 = (angles < 30.0).float().mean().item()

    return {
        "angular_mean": angular_mean,
        "angular_median": angular_median,
        "inlier_11_25": inlier_11,
        "inlier_22_5": inlier_22,
        "inlier_30": inlier_30,
    }


def planarity_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    if mask.sum() < 50:
        return {
            "residual_mean": float("nan"),
            "residual_median": float("nan"),
            "inlier_ratio": float("nan"),
        }

    gt_points = depth_to_points(gt_depth, intrinsics)[mask]
    pred_points = depth_to_points(pred_depth, intrinsics)[mask]

    centroid, normal = plane_fit(gt_points)
    residuals = plane_residuals(pred_points, centroid, normal)
    residual_mean = residuals.mean().item()
    residual_median = residuals.median().item()
    inlier_ratio = (residuals < 0.1).float().mean().item()

    return {
        "residual_mean": residual_mean,
        "residual_median": residual_median,
        "inlier_ratio": inlier_ratio,
    }


# --------------------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------------------


class Kitti360Evaluator:
    def __init__(
        self,
        data_root: str,
        img_size: int,
        patch_size: int,
        seq_len: int,
        stride: int,
        device: str,
        split: str = "val",
        max_sequences: Optional[int] = None,
        use_generator: bool = True,
        stratified_sampling: bool = False,
        use_bf16: bool = False,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ) -> None:
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.stride = stride
        self.max_sequences = max_sequences
        self.use_generator = use_generator
        self.split = split
        self.stratified_sampling = stratified_sampling
        self.use_bf16 = use_bf16
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        common = build_common_config(img_size, patch_size)
        self.dataset = KITTI360Dataset(
            root_dir=data_root,
            split=split,
            common_conf=common,
            sampling_strategy="adaptive",
            building_sampling_weights=(0.5, 0.3, 0.2),
            accumulation_frames=4,
            min_valid_points=2000,
            semantic_weight_enabled=True,
            filter_buildings_only=False,
            depth_range=(0.1, 80.0),
            len_train=50000,
            len_test=5000,
            use_precomputed=True,
        )

        # Count total sequences for progress bar
        sequences = prepare_sequence_indices(self.dataset.frames, seq_len, stride)
        total_count = sum(len(windows) for windows in sequences.values())
        if max_sequences is not None:
            total_count = min(total_count, max_sequences)
        self.total_sequences = total_count

        if not use_generator:
            # Old behavior: load all samples at once (high memory usage)
            self.samples = build_samples(
                self.dataset,
                seq_len=seq_len,
                stride=stride,
                max_sequences=max_sequences,
            )
            logger.info("Prepared %d evaluation sequences (all loaded in memory)", len(self.samples))
        else:
            # New behavior: use generator (low memory usage)
            self.samples = None
            logger.info("Using generator mode for %d evaluation sequences (memory efficient)", total_count)

        if stratified_sampling:
            logger.info("Stratified sampling enabled (proportional sampling from rich/mixed/road buckets)")
        if use_bf16:
            logger.info("BF16 inference enabled for faster computation")

        logger.info(
            f"Dataset: split={split}, seq_len={seq_len}, stride={stride}, "
            f"total_sequences={total_count}, frames={len(self.dataset.frames)}, "
            f"num_workers={num_workers}, prefetch_factor={prefetch_factor}"
        )

    def evaluate_model(
        self,
        model: VGGT,
        align_mode: str,
        tag: str,
        output_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        logger.info(f"Loading model '{tag}' to device: {self.device}")
        model = model.to(self.device).eval()

        # Verify model is on correct device
        if self.device.type == "cuda":
            param_device = next(model.parameters()).device
            logger.info(f"Model loaded on device: {param_device}")
            logger.info(f"GPU Memory before evaluation: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

        agg = AggregatedMetrics()
        pose_trans_errors: List[float] = []
        pose_rot_errors: List[float] = []
        rpe_trans_errors: List[float] = []
        rpe_rot_errors: List[float] = []

        torch.cuda.reset_peak_memory_stats(self.device) if self.device.type == "cuda" else None
        start_time = time.perf_counter()

        # Use generator if enabled, otherwise use preloaded samples
        if self.use_generator:
            sample_iter = sample_generator(
                self.dataset,
                seq_len=self.seq_len,
                stride=self.stride,
                max_sequences=self.max_sequences,
                stratified_sampling=self.stratified_sampling,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )
            total = self.total_sequences
        else:
            sample_iter = self.samples
            total = len(self.samples)

        for sample_idx, sample in enumerate(tqdm(sample_iter, desc=f"{tag} evaluation", total=total)):
            # Log GPU usage for first batch to confirm GPU is being used
            if sample_idx == 0 and self.device.type == "cuda":
                logger.info(f"Processing first batch on GPU...")
            data = sample.to_device(self.device)

            # Use autocast for BF16 inference if enabled
            with torch.no_grad():
                if self.use_bf16 and self.device.type == "cuda":
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(images=data.images.unsqueeze(0))
                else:
                    outputs = model(images=data.images.unsqueeze(0))

            pred_depths = outputs["depth"].squeeze(0).squeeze(-1)  # [S, H, W]
            pose_enc = outputs.get("pose_enc")
            if pose_enc is not None:
                # pose_enc is already [B, S, ...] from model output, no need to unsqueeze
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_enc,
                    image_size_hw=data.images.shape[-2:],
                )
                pred_extrinsics = pred_extrinsics.squeeze(0)
                pred_intrinsics = pred_intrinsics.squeeze(0)
            else:
                pred_extrinsics = None
                pred_intrinsics = None

            for idx in range(pred_depths.shape[0]):
                pred_depth = pred_depths[idx]
                gt_depth = data.gt_depths[idx]
                valid_mask = data.gt_masks[idx]
                build_mask = data.building_masks[idx]
                nb_mask = data.non_building_masks[idx]
                intr = data.gt_intrinsics[idx]

                pred_depth_aligned = align_depth(pred_depth, gt_depth, valid_mask, align_mode)

                metrics_full = depth_metrics(pred_depth_aligned, gt_depth, valid_mask)
                metrics_building = depth_metrics(pred_depth_aligned, gt_depth, build_mask)
                metrics_non_building = depth_metrics(pred_depth_aligned, gt_depth, nb_mask)

                agg.extend("depth.full", metrics_full)
                agg.extend("depth.building", metrics_building)
                agg.extend("depth.non_building", metrics_non_building)

                normal_full = normals_metrics(pred_depth_aligned, gt_depth, valid_mask, intr)
                normal_building = normals_metrics(pred_depth_aligned, gt_depth, build_mask, intr)
                agg.extend("normals.full", normal_full)
                agg.extend("normals.building", normal_building)

                plane_building = planarity_metrics(pred_depth_aligned, gt_depth, intr, build_mask)
                agg.extend("planarity.building", plane_building)

                if idx < pred_depths.shape[0] - 1 and pred_extrinsics is not None and pred_intrinsics is not None:
                    photo_err, inlier = photometric_reprojection_error(
                        data.images[idx],
                        data.images[idx + 1],
                        pred_depth_aligned,
                        pred_extrinsics[idx],
                        pred_extrinsics[idx + 1],
                        pred_intrinsics[idx],
                        pred_intrinsics[idx + 1],
                        build_mask,
                    )
                    agg.add("photometric.building.error", photo_err)
                    agg.add("photometric.building.inlier_ratio", inlier)

            if pred_extrinsics is not None:
                gt_extr = data.gt_extrinsics
                for idx in range(pred_extrinsics.shape[0]):
                    pose_trans_errors.append(
                        translation_error(pred_extrinsics[idx, :, 3], gt_extr[idx, :, 3])
                    )
                    pose_rot_errors.append(
                        rotation_error(pred_extrinsics[idx, :, :3], gt_extr[idx, :, :3])
                    )
                rpe_t, rpe_r = relative_pose_error(pred_extrinsics, gt_extr)
                rpe_trans_errors.append(rpe_t)
                rpe_rot_errors.append(rpe_r)

            # Clean up GPU memory after each sample to prevent accumulation
            if self.device.type == "cuda":
                del data, outputs, pred_depths
                if pose_enc is not None:
                    del pose_enc
                if pred_extrinsics is not None:
                    del pred_extrinsics, pred_intrinsics
                torch.cuda.empty_cache()

            # Log GPU usage after first batch
            if sample_idx == 0 and self.device.type == "cuda":
                logger.info(f"First batch complete. GPU Memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
                logger.info(f"GPU is actively being used for inference ✓")

        elapsed = time.perf_counter() - start_time
        fps = total / elapsed if elapsed > 0 else float("nan")
        latency = elapsed / total if total > 0 else float("nan")
        agg.add("performance.fps", fps)
        agg.add("performance.latency_sec", latency)
        if self.device.type == "cuda":
            vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            agg.add("performance.vram_mb", vram)

        if pose_trans_errors:
            agg.add("pose.ate_trans", float(np.mean(pose_trans_errors)))
            agg.add("pose.ate_rot_deg", float(np.mean(pose_rot_errors)))
        if rpe_trans_errors:
            agg.add("pose.rpe_trans", float(np.nanmean(rpe_trans_errors)))
            agg.add("pose.rpe_rot_deg", float(np.nanmean(rpe_rot_errors)))

        summary = agg.summarize()
        out_file = output_dir / f"{tag}_metrics.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return summary


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------


def load_base_model(
    ckpt_path: Path,
    img_size: int,
    patch_size: int,
    enable_camera: bool,
) -> VGGT:
    model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1024,
        enable_camera=enable_camera,
        enable_point=False,
        enable_depth=True,
        enable_track=False,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model


def load_lora_model(
    ckpt_path: Path,
    base_ckpt: Path,
    lora_cfg: Dict,
    img_size: int,
    patch_size: int,
    cached_base_state: Optional[Dict] = None,
) -> VGGT:
    """
    Load LoRA checkpoint on top of base model.

    Args:
        cached_base_state: Pre-loaded base model state dict to avoid repeated disk I/O.
                          If None, will load from base_ckpt (slower).
    """
    model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=1024,
        enable_camera=True,
        enable_point=False,
        enable_depth=True,
        enable_track=False,
    )
    if lora_cfg and lora_cfg.get("enabled", False):
        model = apply_lora_to_model(model, lora_cfg, verbose=False)

    # Use cached base state if available to avoid disk I/O
    if cached_base_state is not None:
        base_state = cached_base_state
    else:
        base_state = torch.load(base_ckpt, map_location="cpu")
        if isinstance(base_state, dict) and "model" in base_state:
            base_state = base_state["model"]

    model.load_state_dict(base_state, strict=False)

    ckpt_state = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt_state:
        ckpt_state = ckpt_state["model"]
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        logger.debug("Missing keys while loading %s: %s", ckpt_path.name, missing[:5])
    if unexpected:
        logger.debug("Unexpected keys while loading %s: %s", ckpt_path.name, unexpected[:5])
    return model


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------


def flatten_metrics(
    tag: str,
    summary: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    flat: Dict[str, float] = {"checkpoint": tag}
    for key, stats in summary.items():
        flat[f"{key}.mean"] = stats.get("mean", float("nan"))
    return flat


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_completed_checkpoints(output_dir: Path) -> set:
    """
    Detect which checkpoints have already been evaluated by checking for existing metric files.

    Returns:
        Set of checkpoint tags that have completed evaluation (e.g., {"checkpoint_0", "checkpoint_1"})
    """
    completed = set()

    # Look for existing metric JSON files
    for metric_file in output_dir.glob("*_metrics.json"):
        # Extract checkpoint tag from filename (e.g., "checkpoint_8_metrics.json" -> "checkpoint_8")
        tag = metric_file.stem.replace("_metrics", "")
        completed.add(tag)

    return completed


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGGT checkpoints on KITTI-360 buildings.")
    parser.add_argument("--config", type=str, default="lora_kitti360_strategy_b", help="Training config name.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to KITTI-360 root directory.")
    parser.add_argument("--log-dir", type=str, default="logs/lora_kitti360_strategy_b_r16", help="Training log dir.")
    parser.add_argument("--output-dir", type=str, default="evaluate/kitti360_b", help="Output directory.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to use: 'test' for test set (default), 'val' for validation, 'train' for training set")
    parser.add_argument("--align", type=str, default="median", choices=["none", "scale", "median", "scale_shift"])
    parser.add_argument("--seq-len", type=int, default=8, help="Frames per evaluation sequence.")
    parser.add_argument("--seq-stride", type=int, default=8,
                        help="Stride between evaluation sequences (default: 8 for non-overlapping windows, use 4 for 50%% overlap)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use (e.g., 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, 'cpu')")
    parser.add_argument("--max-seqs", type=int, default=None, help="Optional limit on sequences.")
    parser.add_argument("--no-generator", action="store_true",
                        help="Disable generator mode (loads all samples in memory, high memory usage)")
    parser.add_argument("--parallel-models", type=int, default=1,
                        help="Number of models to evaluate in parallel (default: 1). Set to 2 to evaluate 2 checkpoints simultaneously to maximize GPU utilization.")
    parser.add_argument("--stratified-sampling", action="store_true",
                        help="Enable stratified sampling: sample proportionally from rich/mixed/road buckets instead of random sampling. Recommended when using --max-seqs for more representative evaluation.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 (bfloat16) mixed precision for faster inference. Requires Ampere GPU or newer (RTX 30xx/40xx, A100, etc.). Provides ~20-30%% speedup with negligible accuracy loss.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel data loading threads (default: 4). Increase for faster I/O if you have many CPU cores. Set to 0 to disable parallel loading.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Number of samples to prefetch per worker (default: 2). Total prefetched samples = num_workers * prefetch_factor.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Set CUDA device explicitly if using GPU
    # Note: In some Docker environments, torch.cuda.is_available() may return False
    # even though GPUs are accessible. We follow the same approach as training code
    # which sets the device directly without checking availability first.
    if args.device.startswith("cuda"):
        # Parse device ID (e.g., "cuda:0" -> 0, "cuda" -> 0)
        if ":" in args.device:
            device_id = int(args.device.split(":")[1])
        else:
            device_id = 0

        try:
            torch.cuda.set_device(device_id)
            logger.info(f"Set CUDA device to: {device_id}")
            logger.info(f"CUDA available (torch check): {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"Device name: {torch.cuda.get_device_name(device_id)}")
        except Exception as e:
            logger.warning(f"Could not set CUDA device: {e}")
            logger.warning("Proceeding anyway - trainer.py uses similar approach")

    cfg = load_yaml_config(args.config)
    img_size = cfg.get("img_size", 518)
    patch_size = cfg.get("patch_size", 14)
    checkpoint_cfg = cfg.get("checkpoint", {})
    base_checkpoint = checkpoint_cfg.get(
        "resume_checkpoint_path",
        str(REPO_ROOT / "pretrained" / "vggt-1b" / "model.pt"),
    )
    lora_cfg = cfg.get("lora", {})

    log_dir = Path(args.log_dir)
    ckpt_dir = log_dir / "ckpts"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect already-completed checkpoints for resumption
    completed_checkpoints = get_completed_checkpoints(output_dir)
    if completed_checkpoints:
        logger.info(f"Found {len(completed_checkpoints)} already-evaluated checkpoints. Will skip: {sorted(completed_checkpoints)}")
    else:
        logger.info("No existing evaluation results found. Starting fresh evaluation.")

    logger.info("Loading evaluation dataset (this may take a few minutes on CPU)...")
    evaluator = Kitti360Evaluator(
        data_root=args.data_root,
        img_size=img_size,
        patch_size=patch_size,
        seq_len=args.seq_len,
        stride=args.seq_stride,
        device=args.device,
        split=args.split,
        max_sequences=args.max_seqs,
        use_generator=not args.no_generator,
        stratified_sampling=args.stratified_sampling,
        use_bf16=args.bf16,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    logger.info(f"Dataset loaded. Starting evaluation on {args.device} using '{args.split}' split...")

    summaries: List[Dict[str, float]] = []

    # Evaluate base model (skip if already completed)
    if "baseline" in completed_checkpoints:
        logger.info("Baseline already evaluated. Loading existing results...")
        baseline_metrics_path = output_dir / "baseline_metrics.json"
        with baseline_metrics_path.open("r", encoding="utf-8") as f:
            base_summary = json.load(f)
        summaries.append(flatten_metrics("baseline", base_summary))
    else:
        base_model = load_base_model(Path(base_checkpoint), img_size, patch_size, enable_camera=True)
        base_summary = evaluator.evaluate_model(
            base_model,
            align_mode=args.align,
            tag="baseline",
            output_dir=output_dir,
        )
        summaries.append(flatten_metrics("baseline", base_summary))
        logger.info("Baseline evaluation complete.")

        # Clean up base model to free memory
        del base_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            logger.info(f"Freed base model memory. GPU Memory: {torch.cuda.memory_allocated(args.device) / 1024**3:.2f} GB")

    ckpt_files = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )

    # Filter out already-completed checkpoints
    pending_ckpt_files = []
    for ckpt_path in ckpt_files:
        tag = ckpt_path.stem
        if tag in completed_checkpoints:
            logger.info(f"Skipping already-evaluated checkpoint: {tag}")
            # Load existing results for CSV summary
            existing_metrics_path = output_dir / f"{tag}_metrics.json"
            if existing_metrics_path.exists():
                with existing_metrics_path.open("r", encoding="utf-8") as f:
                    existing_summary = json.load(f)
                summaries.append(flatten_metrics(tag, existing_summary))
        else:
            pending_ckpt_files.append(ckpt_path)

    if not pending_ckpt_files:
        logger.info("All checkpoints already evaluated. Nothing to do.")
        csv_path = output_dir / "metrics_overall.csv"
        write_csv(csv_path, summaries)
        logger.info("Saved summary metrics to %s", csv_path)
        return

    logger.info(f"Found {len(pending_ckpt_files)} pending checkpoints to evaluate: {[p.stem for p in pending_ckpt_files]}")

    # Cache base model state to avoid repeated disk I/O (major speedup)
    logger.info("Loading base model state into memory for reuse...")
    base_state = torch.load(Path(base_checkpoint), map_location="cpu")
    if isinstance(base_state, dict) and "model" in base_state:
        base_state = base_state["model"]
    logger.info("Base model state cached in RAM (avoiding disk I/O for each checkpoint)")

    if args.parallel_models <= 1:
        # Sequential evaluation (original behavior)
        for ckpt_path in pending_ckpt_files:
            model = load_lora_model(
                ckpt_path,
                Path(base_checkpoint),
                lora_cfg,
                img_size=img_size,
                patch_size=patch_size,
                cached_base_state=base_state,
            )
            tag = ckpt_path.stem
            summary = evaluator.evaluate_model(model, args.align, tag, output_dir)
            summaries.append(flatten_metrics(tag, summary))
            logger.info("Completed evaluation for %s", ckpt_path.name)

            # Clean up model to free memory before loading next checkpoint
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                logger.info(f"Freed model memory. GPU Memory: {torch.cuda.memory_allocated(args.device) / 1024**3:.2f} GB")
    else:
        # Parallel evaluation with multiple models
        logger.info(f"Evaluating {args.parallel_models} checkpoints in parallel...")

        # Thread-safe lock for accessing summaries list
        summaries_lock = threading.Lock()

        def evaluate_checkpoint(ckpt_path: Path) -> None:
            """Worker function to evaluate a single checkpoint."""
            model = load_lora_model(
                ckpt_path,
                Path(base_checkpoint),
                lora_cfg,
                img_size=img_size,
                patch_size=patch_size,
                cached_base_state=base_state,
            )
            tag = ckpt_path.stem
            summary = evaluator.evaluate_model(model, args.align, tag, output_dir)

            # Thread-safe append to summaries
            with summaries_lock:
                summaries.append(flatten_metrics(tag, summary))

            logger.info("Completed evaluation for %s", ckpt_path.name)

            # Clean up model to free memory
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                logger.info(f"[{tag}] Freed model memory. GPU Memory: {torch.cuda.memory_allocated(args.device) / 1024**3:.2f} GB")

        # Use ThreadPoolExecutor to run multiple evaluations in parallel
        with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
            # Submit all pending checkpoint evaluation tasks
            futures = {executor.submit(evaluate_checkpoint, ckpt_path): ckpt_path for ckpt_path in pending_ckpt_files}

            # Wait for all to complete and handle any exceptions
            for future in as_completed(futures):
                ckpt_path = futures[future]
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    logger.error(f"Error evaluating {ckpt_path.name}: {e}", exc_info=True)

    csv_path = output_dir / "metrics_overall.csv"
    write_csv(csv_path, summaries)
    logger.info("Saved summary metrics to %s", csv_path)


if __name__ == "__main__":
    main()
