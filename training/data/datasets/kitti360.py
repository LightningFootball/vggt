# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for KITTI-360 dataset support with adaptive sampling

import os
import numpy as np
import cv2
import random
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image

from data.dataset_util import *
from data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# KITTI-360 Semantic Segmentation Class IDs (Cityscapes standard)
# Reference: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
SEMANTIC_CLASSES = {
    'road': 7,
    'sidewalk': 8,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'pole': 17,
    'traffic_light': 19,
    'traffic_sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
}

# Semantic weighting for building-focused training
DEFAULT_SEMANTIC_WEIGHTS = {
    11: 2.0,   # building (重点)
    12: 2.0,   # wall (建筑物附属)
    13: 1.5,   # fence (次要结构)
    7:  0.7,   # road (保留基本监督)
    8:  0.7,   # sidewalk
    26: 0.5,   # car (移动物体降权)
    23: 0.2,   # sky (背景降权)
}


class KITTI360Dataset(BaseDataset):
    """
    KITTI-360 Dataset for LoRA fine-tuning of VGGT.

    Features:
    - Adaptive sampling based on building ratio (建筑丰富/混合场景/道路为主)
    - Multi-frame LiDAR accumulation for dense depth
    - Semantic-based pixel weighting for building-focused training
    - Compatible with VGGT BaseDataset interface

    Dataset structure:
        root_dir/
        ├── data_2d_raw/{sequence}/image_00/data_rect/*.png
        ├── data_3d_raw/{sequence}/velodyne_points/data/*.bin
        ├── data_poses/{sequence}/poses.txt
        ├── data_2d_semantics/train/{sequence}/image_00/semantic/*.png
        └── calibration/
            ├── perspective.txt
            └── calib_cam_to_velo.txt

    Args:
        root_dir: Path to KITTI-360 dataset root
        split: 'train' or 'val'
        common_conf: Common configuration object from training config
        sequences: List of sequence names (if None, use all train sequences)
        sampling_strategy: 'adaptive' (建筑分类采样) or 'uniform'
        building_sampling_weights: Tuple of (rich, mixed, road) sampling weights
        accumulation_frames: Number of frames to accumulate for LiDAR (±N frames)
        min_valid_points: Minimum valid depth points per frame
        semantic_weight_enabled: Whether to compute semantic weights
        semantic_weights: Custom semantic weight dict
        filter_buildings_only: If True, mask out non-building pixels
        depth_range: (min_depth, max_depth) in meters
        len_train: Dataset length for training
        len_test: Dataset length for testing
    """

    # All KITTI-360 training sequences
    ALL_SEQUENCES = [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync',
        '2013_05_28_drive_0004_sync',
        '2013_05_28_drive_0005_sync',
        '2013_05_28_drive_0006_sync',
        '2013_05_28_drive_0007_sync',
        '2013_05_28_drive_0009_sync',
        '2013_05_28_drive_0010_sync',
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        common_conf=None,
        sequences: Optional[List[str]] = None,
        sampling_strategy: str = 'adaptive',
        building_sampling_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        accumulation_frames: int = 4,
        min_valid_points: int = 2000,
        semantic_weight_enabled: bool = True,
        semantic_weights: Optional[Dict[int, float]] = None,
        filter_buildings_only: bool = False,
        depth_range: Tuple[float, float] = (0.1, 80.0),
        len_train: int = 50000,
        len_test: int = 5000,
        camera_id: int = 0,  # 0 for image_00 (left), 1 for image_01 (right)
        dataset_configs=None,  # Ignored, for compatibility with DynamicTorchDataset
        **kwargs,  # Accept and ignore any additional parameters
    ):
        # Handle common_conf being None (shouldn't happen but be defensive)
        if common_conf is None:
            raise ValueError("common_conf is required but was None. Check your training config structure.")

        super().__init__(common_conf=common_conf)

        self.root_dir = root_dir
        self.split = split
        self.camera_id = camera_id
        self.camera_name = f'image_{camera_id:02d}'

        # Set training flag (required by BaseDataset)
        self.training = (split == 'train')
        self.debug = False
        self.get_nearby = False
        self.load_depth = True
        self.inside_random = False
        self.allow_duplicate_img = False

        # Sampling configuration
        self.sampling_strategy = sampling_strategy
        self.building_sampling_weights = building_sampling_weights
        self.accumulation_frames = accumulation_frames
        self.min_valid_points = min_valid_points

        # Semantic weighting configuration
        self.semantic_weight_enabled = semantic_weight_enabled
        self.semantic_weights = semantic_weights if semantic_weights is not None else DEFAULT_SEMANTIC_WEIGHTS
        self.filter_buildings_only = filter_buildings_only

        # Depth configuration
        self.min_depth, self.max_depth = depth_range

        # Dataset length
        self.len_train = len_train if split == 'train' else len_test

        # Select sequences
        self.sequences = sequences if sequences is not None else self.ALL_SEQUENCES

        # Load calibration (shared across all sequences)
        self.calib = self._load_calibration()

        # Load all frames and metadata
        self.frames = []
        self.sequence_metadata = {}
        for seq_name in self.sequences:
            seq_frames = self._load_sequence(seq_name)
            self.frames.extend(seq_frames)
            self.sequence_metadata[seq_name] = {
                'start_idx': len(self.frames) - len(seq_frames),
                'end_idx': len(self.frames),
                'num_frames': len(seq_frames),
            }

        # Load train/val split from official annotation
        self.train_frame_list, self.val_frame_list = self._load_official_split()

        # Filter frames by split
        if split == 'train':
            self.frames = [f for f in self.frames if self._get_frame_key(f) in self.train_frame_list]
        else:
            self.frames = [f for f in self.frames if self._get_frame_key(f) in self.val_frame_list]

        # Rebuild sequence metadata after filtering
        self.sequence_metadata = {}
        current_idx = 0
        for seq_name in self.sequences:
            seq_frames = [f for f in self.frames if f['sequence'] == seq_name]
            if len(seq_frames) > 0:
                self.sequence_metadata[seq_name] = {
                    'start_idx': current_idx,
                    'end_idx': current_idx + len(seq_frames),
                    'num_frames': len(seq_frames),
                }
                current_idx += len(seq_frames)

        # Build sampling buckets (for adaptive sampling)
        if sampling_strategy == 'adaptive':
            self._build_sampling_buckets()

        logger.info(f"KITTI-360 {split}: Loaded {len(self.frames)} frames from {len(self.sequences)} sequences")
        logger.info(f"Sampling strategy: {sampling_strategy}")
        if sampling_strategy == 'adaptive':
            logger.info(f"Building sampling weights: rich={building_sampling_weights[0]:.2f}, "
                       f"mixed={building_sampling_weights[1]:.2f}, road={building_sampling_weights[2]:.2f}")
            logger.info(f"Bucket sizes: rich={len(self.bucket_building_rich)}, "
                       f"mixed={len(self.bucket_mixed)}, road={len(self.bucket_road)}")

    def _load_calibration(self) -> Dict:
        """Load camera calibration from perspective.txt and cam_to_velo.txt"""
        calib_dir = os.path.join(self.root_dir, 'calibration')

        # Load perspective camera calibration
        perspective_file = os.path.join(calib_dir, 'perspective.txt')
        calib = {}

        with open(perspective_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(':')
                if len(parts) != 2:
                    continue

                key = parts[0].strip()
                values = list(map(float, parts[1].strip().split()))

                # Parse camera 00 and 01 parameters
                if key.startswith(f'P_rect_{self.camera_id:02d}'):
                    # P_rect is 3x4 projection matrix
                    calib['P_rect'] = np.array(values).reshape(3, 4)
                    # Extract intrinsics: K = P_rect[:, :3]
                    calib['K'] = calib['P_rect'][:, :3]
                elif key.startswith(f'R_rect_{self.camera_id:02d}'):
                    calib['R_rect'] = np.array(values).reshape(3, 3)

        # Load camera to velodyne calibration
        # KITTI-360 format: single line with 12 values representing 3x4 matrix
        # [r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3]
        cam_to_velo_file = os.path.join(calib_dir, 'calib_cam_to_velo.txt')
        with open(cam_to_velo_file, 'r') as f:
            line = f.readline().strip()
            values = list(map(float, line.split()))

            if len(values) == 12:
                # Parse 3x4 matrix
                T_cam_to_velo = np.array(values).reshape(3, 4)

                # Build 4x4 transformation matrix: cam0 -> velo
                calib['T_cam_to_velo_4x4'] = np.eye(4)
                calib['T_cam_to_velo_4x4'][:3, :] = T_cam_to_velo

                # Extract R and t for backward compatibility
                calib['R_cam_to_velo'] = T_cam_to_velo[:, :3]
                calib['T_cam_to_velo'] = T_cam_to_velo[:, 3:4]
            else:
                raise ValueError(f"Expected 12 values in calib_cam_to_velo.txt, got {len(values)}")

        # Inverse: velo -> cam0
        calib['T_velo_to_cam_4x4'] = np.linalg.inv(calib['T_cam_to_velo_4x4'])

        return calib

    def _load_sequence(self, seq_name: str) -> List[Dict]:
        """Load all frames from a sequence"""
        seq_dir = os.path.join(self.root_dir, 'data_2d_raw', seq_name)
        image_dir = os.path.join(seq_dir, self.camera_name, 'data_rect')

        if not os.path.exists(image_dir):
            logger.warning(f"Sequence {seq_name} not found at {image_dir}")
            return []

        # Load poses
        pose_file = os.path.join(self.root_dir, 'data_poses', seq_name, 'poses.txt')
        poses = self._parse_poses(pose_file)

        # Get all image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        frames = []
        for img_file in image_files:
            frame_idx = int(os.path.splitext(img_file)[0])

            # Check if pose exists for this frame
            if frame_idx not in poses:
                continue

            # Build paths
            img_path = os.path.join(image_dir, img_file)
            lidar_path = os.path.join(self.root_dir, 'data_3d_raw', seq_name,
                                     'velodyne_points', 'data', f'{frame_idx:010d}.bin')
            semantic_path = os.path.join(self.root_dir, 'data_2d_semantics', 'train',
                                        seq_name, self.camera_name, 'semantic', f'{frame_idx:010d}.png')

            # Check if required files exist
            if not os.path.exists(img_path) or not os.path.exists(lidar_path):
                continue

            frames.append({
                'sequence': seq_name,
                'frame_idx': frame_idx,
                'image_path': img_path,
                'lidar_path': lidar_path,
                'semantic_path': semantic_path if os.path.exists(semantic_path) else None,
                'pose': poses[frame_idx],  # 3x4 camera pose matrix [R|t]
            })

        return frames

    def _parse_poses(self, pose_file: str) -> Dict[int, np.ndarray]:
        """
        Parse KITTI-360 poses.txt
        Format: frame_id r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
        Returns: dict mapping frame_id -> 4x4 transformation matrix (cam0 -> world)
        """
        poses = {}
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 13:
                    continue

                frame_id = int(parts[0])
                values = list(map(float, parts[1:]))

                # Build 3x4 matrix
                pose_3x4 = np.array(values).reshape(3, 4)

                # Convert to 4x4
                pose = np.eye(4)
                pose[:3, :] = pose_3x4

                poses[frame_id] = pose

        return poses

    def _load_official_split(self) -> Tuple[set, set]:
        """Load official train/val split from annotation files"""
        train_file = os.path.join(self.root_dir, 'data_2d_semantics', 'train',
                                  '2013_05_28_drive_train_frames.txt')
        val_file = os.path.join(self.root_dir, 'data_2d_semantics', 'train',
                               '2013_05_28_drive_val_frames.txt')

        train_set = set()
        val_set = set()

        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        # Extract sequence and frame from path like:
                        # data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png
                        img_path = parts[0]
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 3:
                            seq = path_parts[1]
                            frame = int(os.path.splitext(os.path.basename(img_path))[0])
                            camera = path_parts[2]
                            train_set.add((seq, camera, frame))

        if os.path.exists(val_file):
            with open(val_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        img_path = parts[0]
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 3:
                            seq = path_parts[1]
                            frame = int(os.path.splitext(os.path.basename(img_path))[0])
                            camera = path_parts[2]
                            val_set.add((seq, camera, frame))

        logger.info(f"Official split: {len(train_set)} train frames, {len(val_set)} val frames")
        return train_set, val_set

    def _get_frame_key(self, frame: Dict) -> Tuple[str, str, int]:
        """Get unique key for frame in official split"""
        return (frame['sequence'], self.camera_name, frame['frame_idx'])

    def _build_sampling_buckets(self):
        """Build sampling buckets based on building ratio"""
        self.bucket_building_rich = []  # building_ratio >= 0.30
        self.bucket_mixed = []          # 0.10 <= building_ratio < 0.30
        self.bucket_road = []           # building_ratio < 0.10

        logger.info("Building sampling buckets (analyzing building ratios)...")

        for idx, frame in enumerate(self.frames):
            if frame['semantic_path'] is None:
                # If no semantic annotation, assume mixed
                self.bucket_mixed.append(idx)
                continue

            # Load semantic mask and compute building ratio
            semantic = cv2.imread(frame['semantic_path'], cv2.IMREAD_GRAYSCALE)
            if semantic is None:
                self.bucket_mixed.append(idx)
                continue

            total_pixels = semantic.size
            building_pixels = (semantic == SEMANTIC_CLASSES['building']).sum()
            building_ratio = building_pixels / total_pixels

            # Classify into buckets
            if building_ratio >= 0.30:
                self.bucket_building_rich.append(idx)
            elif building_ratio >= 0.10:
                self.bucket_mixed.append(idx)
            else:
                self.bucket_road.append(idx)

            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.frames)} frames")

        logger.info(f"Bucket distribution: "
                   f"rich={len(self.bucket_building_rich)} ({len(self.bucket_building_rich)/len(self.frames)*100:.1f}%), "
                   f"mixed={len(self.bucket_mixed)} ({len(self.bucket_mixed)/len(self.frames)*100:.1f}%), "
                   f"road={len(self.bucket_road)} ({len(self.bucket_road)/len(self.frames)*100:.1f}%)")

    def _sample_frame_index(self) -> int:
        """Sample a frame index based on sampling strategy"""
        if self.sampling_strategy == 'adaptive':
            # Sample from buckets with specified weights
            bucket_choice = random.choices(
                ['rich', 'mixed', 'road'],
                weights=self.building_sampling_weights,
                k=1
            )[0]

            if bucket_choice == 'rich' and len(self.bucket_building_rich) > 0:
                return random.choice(self.bucket_building_rich)
            elif bucket_choice == 'mixed' and len(self.bucket_mixed) > 0:
                return random.choice(self.bucket_mixed)
            elif bucket_choice == 'road' and len(self.bucket_road) > 0:
                return random.choice(self.bucket_road)
            else:
                # Fallback to uniform sampling if bucket is empty
                return random.randint(0, len(self.frames) - 1)
        else:
            # Uniform sampling
            return random.randint(0, len(self.frames) - 1)

    def _load_lidar_points(self, lidar_path: str) -> np.ndarray:
        """
        Load LiDAR point cloud from .bin file
        Returns: Nx4 array [x, y, z, reflectance]
        """
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        return points

    def _project_lidar_to_depth(
        self,
        frame: Dict,
        center_frame_idx: int,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project multi-frame accumulated LiDAR points to depth map

        Args:
            frame: Current frame dict
            center_frame_idx: Index of center frame in self.frames
            image_shape: (height, width) of target image

        Returns:
            depth_map: HxW depth map
            depth_mask: HxW bool mask of valid depth pixels
        """
        H, W = image_shape
        depth_map = np.zeros((H, W), dtype=np.float32)
        depth_count = np.zeros((H, W), dtype=np.int32)

        current_pose = frame['pose']
        current_pose_inv = np.linalg.inv(current_pose)

        # Get sequence bounds for this frame
        seq_name = frame['sequence']
        seq_meta = self.sequence_metadata.get(seq_name)
        if seq_meta is None:
            # Sequence not found, only use current frame
            frame_indices = [center_frame_idx]
        else:
            seq_start = seq_meta['start_idx']
            seq_end = seq_meta['end_idx']

            # Generate nearby indices within sequence bounds
            start_idx = max(seq_start, center_frame_idx - self.accumulation_frames)
            end_idx = min(seq_end, center_frame_idx + self.accumulation_frames + 1)
            frame_indices = list(range(start_idx, end_idx))

        for frame_idx in frame_indices:
            # Get frame data
            if frame_idx < 0 or frame_idx >= len(self.frames):
                continue

            other_frame = self.frames[frame_idx]
            if other_frame['sequence'] != frame['sequence']:
                continue  # Only accumulate within same sequence

            # Load LiDAR points
            if not os.path.exists(other_frame['lidar_path']):
                continue

            points_velo = self._load_lidar_points(other_frame['lidar_path'])[:, :3]  # Nx3

            # Transform: velo -> cam0 (of other frame)
            points_cam_other = self._transform_velo_to_cam(points_velo)

            # Transform: cam0 (other frame) -> world -> cam0 (current frame)
            if frame_idx != frame_indices[len(frame_indices)//2]:  # Not current frame
                # other_cam -> world
                points_world = self._transform_points(points_cam_other, other_frame['pose'])
                # world -> current_cam
                points_cam = self._transform_points(points_world, current_pose_inv)
            else:
                points_cam = points_cam_other

            # Filter points behind camera
            valid_mask = points_cam[:, 2] > self.min_depth
            points_cam = points_cam[valid_mask]

            if len(points_cam) == 0:
                continue

            # Project to image
            points_2d, depths = self._project_points_to_image(points_cam)

            # Filter valid projections
            valid_mask = (
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) &
                (depths > self.min_depth) & (depths < self.max_depth)
            )

            points_2d = points_2d[valid_mask].astype(np.int32)
            depths = depths[valid_mask]

            # Accumulate depths (average) using vectorized indexed adds
            us = points_2d[:, 0].astype(np.intp, copy=False)
            vs = points_2d[:, 1].astype(np.intp, copy=False)
            depth_vals = depths.astype(depth_map.dtype, copy=False)
            np.add.at(depth_map, (vs, us), depth_vals)
            np.add.at(depth_count, (vs, us), np.ones_like(vs, dtype=depth_count.dtype))

        # Average accumulated depths
        valid_mask = depth_count > 0
        depth_map[valid_mask] /= depth_count[valid_mask]

        return depth_map, valid_mask

    def _transform_velo_to_cam(self, points_velo: np.ndarray) -> np.ndarray:
        """Transform points from velodyne to camera coordinates"""
        # points_velo: Nx3
        points_hom = np.hstack([points_velo, np.ones((len(points_velo), 1))])  # Nx4
        points_cam_hom = (self.calib['T_velo_to_cam_4x4'] @ points_hom.T).T  # Nx4
        return points_cam_hom[:, :3]

    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply 4x4 transformation to 3D points"""
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        points_transformed = (transform @ points_hom.T).T
        return points_transformed[:, :3]

    def _project_points_to_image(self, points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D camera points to 2D image
        Returns: (points_2d, depths)
        """
        # points_cam: Nx3
        K = self.calib['K']

        # Project: [u, v, 1]^T = K @ [x, y, z]^T / z
        points_2d_hom = (K @ points_cam.T).T  # Nx3
        depths = points_2d_hom[:, 2]

        # Avoid division by zero
        valid_mask = depths > 1e-6
        points_2d = np.zeros((len(points_cam), 2))
        points_2d[valid_mask, 0] = points_2d_hom[valid_mask, 0] / depths[valid_mask]  # u
        points_2d[valid_mask, 1] = points_2d_hom[valid_mask, 1] / depths[valid_mask]  # v

        return points_2d, depths

    def _compute_semantic_weights(
        self,
        semantic_mask: np.ndarray,
        depth_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pixel-wise semantic weights

        Args:
            semantic_mask: HxW semantic class IDs
            depth_mask: HxW bool mask of valid depth

        Returns:
            weights: HxW float32 weights
        """
        H, W = semantic_mask.shape
        weights = np.ones((H, W), dtype=np.float32)

        # Apply semantic class weights
        for class_id, weight in self.semantic_weights.items():
            weights[semantic_mask == class_id] = weight

        # If filter_buildings_only, zero out non-building pixels
        if self.filter_buildings_only:
            building_mask = (semantic_mask == SEMANTIC_CLASSES['building'])
            weights[~building_mask] = 0.0

        # Ensure zero weight for invalid depth
        weights[~depth_mask] = 0.0

        return weights

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        aspect_ratio: float = 1.0,
        seq_name: str = None,
        ids: list = None,
    ) -> dict:
        """
        Retrieve data for a sequence (compatible with BaseDataset interface)

        Args:
            seq_index: Sequence index (not used, sampling is frame-based)
            img_per_seq: Number of images per sequence
            aspect_ratio: Target aspect ratio
            seq_name: Sequence name (not used)
            ids: Frame IDs (not used)

        Returns:
            Batch dict compatible with VGGT training
        """
        # Sample anchor frame
        anchor_idx = self._sample_frame_index()
        anchor_frame = self.frames[anchor_idx]

        # Sample nearby frames in the same sequence
        # Find all frames in the same sequence
        seq_name = anchor_frame['sequence']
        seq_meta = self.sequence_metadata[seq_name]
        seq_start = seq_meta['start_idx']
        seq_end = seq_meta['end_idx']
        seq_len = seq_meta['num_frames']

        # Compute anchor position within sequence
        anchor_pos = anchor_idx - seq_start

        # Sample frames around anchor (temporal window)
        # Retry sampling if depth coverage is too low (max 5 attempts)
        max_retries = 5
        for retry in range(max_retries):
            frame_indices = []
            for _ in range(img_per_seq):
                # Sample within ±50 frames of anchor
                offset = random.randint(-50, 50)
                sampled_pos = np.clip(anchor_pos + offset, 0, seq_len - 1)
                frame_indices.append(seq_start + sampled_pos)

            # Ensure anchor is included
            if anchor_idx not in frame_indices:
                frame_indices[0] = anchor_idx

            # Quick check: verify anchor frame has sufficient depth
            # (full check happens later during processing)
            anchor_depth_map, anchor_depth_mask = self._project_lidar_to_depth(
                anchor_frame, anchor_idx, np.array([1080, 1920])  # Use original size for check
            )
            anchor_valid_count = anchor_depth_mask.sum()

            if anchor_valid_count >= self.min_valid_points:
                break  # Good anchor frame

            if retry < max_retries - 1:
                # Try different anchor
                anchor_idx = self._sample_frame_index()
                anchor_frame = self.frames[anchor_idx]
                seq_name = anchor_frame['sequence']
                seq_meta = self.sequence_metadata[seq_name]
                seq_start = seq_meta['start_idx']
                seq_end = seq_meta['end_idx']
                seq_len = seq_meta['num_frames']
                anchor_pos = anchor_idx - seq_start

        # Get target image shape
        target_image_shape = self.get_target_shape(aspect_ratio)

        # Process each frame
        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        semantic_weights_list = []
        original_sizes = []

        for idx in frame_indices:
            frame = self.frames[idx]

            # Load image
            image = read_image_cv2(frame['image_path'])
            original_size = np.array(image.shape[:2])

            # Load or generate depth
            # Accumulate LiDAR from nearby frames
            # Note: nearby_indices are local to current sequence, not global frame indices
            # We pass frame indices to _project_lidar_to_depth which handles sequence bounds
            depth_map, depth_mask = self._project_lidar_to_depth(
                frame, idx, original_size
            )

            # Load semantic mask (if available)
            if frame['semantic_path'] is not None and os.path.exists(frame['semantic_path']):
                semantic_mask = cv2.imread(frame['semantic_path'], cv2.IMREAD_GRAYSCALE)
                if semantic_mask.shape[:2] != tuple(original_size):
                    semantic_mask = cv2.resize(semantic_mask, (original_size[1], original_size[0]),
                                              interpolation=cv2.INTER_NEAREST)
            else:
                semantic_mask = np.zeros(original_size, dtype=np.uint8)

            # Compute semantic weights
            if self.semantic_weight_enabled:
                semantic_weights = self._compute_semantic_weights(semantic_mask, depth_mask)
            else:
                semantic_weights = depth_mask.astype(np.float32)

            # Get camera extrinsics and intrinsics
            # KITTI-360 pose is cam0 -> world, we need world -> cam0 (OpenCV convention)
            pose_cam_to_world = frame['pose']
            extri_opencv = np.linalg.inv(pose_cam_to_world).astype(np.float32)

            intri_opencv = self.calib['K'].astype(np.float32)

            # Process using BaseDataset utilities
            (
                image_processed,
                depth_map_processed,
                extri_processed,
                intri_processed,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=frame['image_path'],
            )

            # Also resize semantic weights to match processed image
            semantic_weights_resized = cv2.resize(
                semantic_weights,
                (intri_processed.shape[1], intri_processed.shape[0]) if len(intri_processed.shape) > 2 else (image_processed.shape[1], image_processed.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Apply point_mask to semantic weights
            semantic_weights_resized = semantic_weights_resized * point_mask

            images.append(image_processed)
            depths.append(depth_map_processed)
            extrinsics.append(extri_processed)
            intrinsics.append(intri_processed)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            semantic_weights_list.append(semantic_weights_resized)
            original_sizes.append(original_size)

        set_name = "kitti360"

        batch = {
            "seq_name": f"{set_name}_{seq_name}",
            "ids": np.array(frame_indices),  # Convert list to numpy array
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "semantic_weights": semantic_weights_list,  # New field
            "original_sizes": original_sizes,
        }

        return batch

    def __len__(self):
        return self.len_train
