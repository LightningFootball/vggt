#!/usr/bin/env python3
"""
KITTI-360 Dataset Preprocessing Script for VGGT LoRA Training

This script preprocesses KITTI-360 data to accelerate training by:
1. Pre-computing sampling buckets (building-rich/mixed/road scenes)
2. Pre-generating accumulated depth maps from multi-frame LiDAR
3. Generating validation statistics and visualizations

The preprocessing is organized in a layered structure:
- base/: Shared metadata (frames_index, buckets, calibration) - config-agnostic
- depths/{config}/: Configuration-specific depth maps

This allows maximum reusability across different training parameter combinations.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add training directory to path for data module imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

from training.data.datasets.kitti360 import SEMANTIC_CLASSES


class KITTI360Preprocessor:
    """
    KITTI-360 dataset preprocessor with layered architecture.

    Architecture:
    - Base Layer: Config-agnostic data (frames index, sampling buckets, calibration)
    - Depth Layer: Config-specific accumulated depth maps

    Parameters that require re-preprocessing:
    - camera_id: Different camera → different images/intrinsics
    - accumulation_frames: Different LiDAR accumulation window
    - depth_range: Different depth filtering range

    Parameters that DON'T require re-preprocessing:
    - max_img_per_gpu, accum_steps: Training-time batch construction
    - img_size, patch_size: Dynamic resizing during training
    - img_nums, aspects, scales: Dynamic augmentation during training
    - sampling_strategy, building_sampling_weights: Sampling logic only
    """

    def __init__(
        self,
        config_path: str,
        output_base_dir: str,
        workers: int = 8,
        overwrite: bool = False,
        split: str = 'all',
        skip_visualization: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            config_path: Path to training config YAML
            output_base_dir: Base output directory for preprocessed data
            workers: Number of parallel workers
            overwrite: Whether to overwrite existing files
            split: Which split to preprocess ('train', 'val', or 'all')
            skip_visualization: Skip validation visualizations (faster)
        """
        # Load training configuration
        print(f"Loading training config from: {config_path}")
        self.train_config = OmegaConf.load(config_path)
        dataset_cfg = self.train_config.data.train.dataset.dataset_configs[0]

        self.root_dir = Path(dataset_cfg.root_dir)
        self.output_base_dir = Path(output_base_dir) / "vggt_lora"

        # Extract core parameters (only these affect preprocessing)
        self.camera_id = dataset_cfg.camera_id
        self.camera_name = f"image_{self.camera_id:02d}"
        self.accumulation_frames = dataset_cfg.accumulation_frames
        self.depth_range = tuple(dataset_cfg.depth_range)
        self.min_valid_points = dataset_cfg.get('min_valid_points', 2000)

        # Base layer directory (shared across all configs)
        self.base_dir = self.output_base_dir / "base"

        # Depth layer directory (config-specific)
        depth_config_id = self._generate_config_id()
        self.depth_dir = self.output_base_dir / "depths" / depth_config_id
        self.config_id = depth_config_id

        # Processing options
        self.workers = workers
        self.overwrite = overwrite
        self.split = split
        self.skip_visualization = skip_visualization

        # All KITTI-360 sequences (training + test)
        self.train_sequences = [
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

        self.test_sequences = [
            '2013_05_28_drive_0008_sync',
            '2013_05_28_drive_0018_sync',
        ]

        self.all_sequences = self.train_sequences + self.test_sequences

        # Setup logging
        self.setup_logging()

        # Print configuration summary
        self.print_config_summary()

    def _generate_config_id(self) -> str:
        """Generate unique config identifier."""
        return f"cam{self.camera_id:02d}_af{self.accumulation_frames}_dr{self.depth_range[0]}-{self.depth_range[1]}"

    def _compute_config_hash(self) -> str:
        """Compute configuration hash for consistency checking."""
        config_str = json.dumps({
            'camera_id': self.camera_id,
            'accumulation_frames': self.accumulation_frames,
            'depth_range': self.depth_range,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.output_base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"preprocess_{self.config_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Log file: {log_file}")

    def print_config_summary(self):
        """Print configuration summary."""
        self.logger.info("=" * 80)
        self.logger.info("KITTI-360 Preprocessing Configuration")
        self.logger.info("=" * 80)
        self.logger.info(f"Root directory:        {self.root_dir}")
        self.logger.info(f"Output base directory: {self.output_base_dir}")
        self.logger.info(f"Config ID:             {self.config_id}")
        self.logger.info(f"Config hash:           {self._compute_config_hash()}")
        self.logger.info("")
        self.logger.info("Core parameters (affect preprocessing):")
        self.logger.info(f"  - camera_id:           {self.camera_id}")
        self.logger.info(f"  - accumulation_frames: {self.accumulation_frames}")
        self.logger.info(f"  - depth_range:         {self.depth_range}")
        self.logger.info("")
        self.logger.info("Processing options:")
        self.logger.info(f"  - workers:             {self.workers}")
        self.logger.info(f"  - overwrite:           {self.overwrite}")
        self.logger.info(f"  - split:               {self.split}")
        self.logger.info(f"  - skip_visualization:  {self.skip_visualization}")
        self.logger.info("=" * 80)

    # ==================== Stage 0: Base Layer ====================

    def run_stage_0_base(self):
        """
        Stage 0: Base layer preprocessing (config-agnostic).

        Generates:
        - frames_index.json: Global frame index with file existence flags
        - buckets.json: Sampling buckets (building-rich/mixed/road)
        - calibration.json: Cached calibration parameters

        This stage only needs to run once for the entire dataset.
        """
        if self.base_dir.exists() and not self.overwrite:
            self.logger.info(f"✅ Base metadata already exists at {self.base_dir}")
            self.logger.info(f"   Use --overwrite to regenerate")

            # Check if files exist
            required_files = ['frames_index.json', 'buckets.json', 'calibration.json']
            missing_files = [f for f in required_files if not (self.base_dir / f).exists()]

            if missing_files:
                self.logger.warning(f"⚠️  Missing files: {missing_files}")
                self.logger.info(f"   Regenerating base metadata...")
            else:
                self.logger.info(f"   Skipping Stage 0...")
                return

        self.logger.info("=" * 80)
        self.logger.info("Stage 0: Base Layer Preprocessing")
        self.logger.info("=" * 80)

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan all frames
        self.logger.info("Step 1/3: Scanning all frames...")
        frames_index = self._scan_all_frames()
        frames_index_path = self.base_dir / 'frames_index.json'
        with open(frames_index_path, 'w') as f:
            json.dump(frames_index, f, indent=2)
        self.logger.info(f"✅ Frames index saved to {frames_index_path}")

        # Step 2: Build sampling buckets
        self.logger.info("Step 2/3: Building sampling buckets...")
        buckets = self._build_sampling_buckets(frames_index)
        buckets_path = self.base_dir / 'buckets.json'
        with open(buckets_path, 'w') as f:
            json.dump(buckets, f, indent=2)
        self.logger.info(f"✅ Sampling buckets saved to {buckets_path}")

        # Step 3: Cache calibration
        self.logger.info("Step 3/3: Caching calibration parameters...")
        calibration = self._load_calibration()
        calibration_path = self.base_dir / 'calibration.json'
        with open(calibration_path, 'w') as f:
            json.dump(calibration, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        self.logger.info(f"✅ Calibration cached to {calibration_path}")

        self.logger.info("=" * 80)
        self.logger.info("✅ Stage 0 Complete: Base metadata generated")
        self.logger.info("=" * 80)

    def _scan_all_frames(self) -> Dict:
        """
        Scan all frames in the dataset and build index.

        Returns:
            Dictionary mapping sequence -> frame_idx -> file existence flags
        """
        frames_index = {}

        # Load official train/val split
        train_set, val_set = self._load_official_split()

        for seq_name in tqdm(self.all_sequences, desc="Scanning sequences"):
            seq_frames = {}

            # Determine if this is a test sequence
            is_test_seq = seq_name in self.test_sequences

            # Image directory (test sequences are in data_2d_test/)
            if is_test_seq:
                image_dir = self.root_dir / 'data_2d_test' / seq_name / self.camera_name / 'data_rect'
            else:
                image_dir = self.root_dir / 'data_2d_raw' / seq_name / self.camera_name / 'data_rect'

            if not image_dir.exists():
                self.logger.warning(f"Sequence {seq_name} not found at {image_dir}")
                continue

            # Load poses for this sequence
            pose_file = self.root_dir / 'data_poses' / seq_name / 'poses.txt'
            poses = self._parse_poses(pose_file) if pose_file.exists() else {}

            # Scan all images
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

            for img_file in image_files:
                frame_idx = int(os.path.splitext(img_file)[0])

                # Build paths
                img_path = image_dir / img_file
                lidar_path = self.root_dir / 'data_3d_raw' / seq_name / 'velodyne_points' / 'data' / f'{frame_idx:010d}.bin'

                # Semantic path differs for test sequences
                if is_test_seq:
                    semantic_path = self.root_dir / 'data_2d_semantics_test' / seq_name / self.camera_name / 'semantic' / f'{frame_idx:010d}.png'
                else:
                    semantic_path = self.root_dir / 'data_2d_semantics' / 'train' / seq_name / self.camera_name / 'semantic' / f'{frame_idx:010d}.png'

                # Determine split
                if is_test_seq:
                    split = 'test'
                else:
                    frame_key = (seq_name, self.camera_name, frame_idx)
                    if frame_key in train_set:
                        split = 'train'
                    elif frame_key in val_set:
                        split = 'val'
                    else:
                        split = 'unknown'

                seq_frames[str(frame_idx)] = {
                    'split': split,
                    'has_image': img_path.exists(),
                    'has_lidar': lidar_path.exists(),
                    'has_semantic': semantic_path.exists(),
                    'has_pose': frame_idx in poses,
                }

            frames_index[seq_name] = seq_frames

        # Statistics
        total_frames = sum(len(frames) for frames in frames_index.values())
        train_frames = sum(1 for seq in frames_index.values() for f in seq.values() if f['split'] == 'train')
        val_frames = sum(1 for seq in frames_index.values() for f in seq.values() if f['split'] == 'val')
        test_frames = sum(1 for seq in frames_index.values() for f in seq.values() if f['split'] == 'test')

        self.logger.info(f"   Total frames: {total_frames}")
        self.logger.info(f"   Train frames: {train_frames}")
        self.logger.info(f"   Val frames:   {val_frames}")
        self.logger.info(f"   Test frames:  {test_frames}")

        return frames_index

    def _load_official_split(self) -> Tuple[set, set]:
        """Load official train/val split."""
        train_file = self.root_dir / 'data_2d_semantics' / 'train' / '2013_05_28_drive_train_frames.txt'
        val_file = self.root_dir / 'data_2d_semantics' / 'train' / '2013_05_28_drive_val_frames.txt'

        train_set = set()
        val_set = set()

        if train_file.exists():
            with open(train_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        img_path = parts[0]
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 3:
                            seq = path_parts[1]
                            camera = path_parts[2]
                            frame = int(os.path.splitext(os.path.basename(img_path))[0])
                            train_set.add((seq, camera, frame))

        if val_file.exists():
            with open(val_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        img_path = parts[0]
                        path_parts = img_path.split('/')
                        if len(path_parts) >= 3:
                            seq = path_parts[1]
                            camera = path_parts[2]
                            frame = int(os.path.splitext(os.path.basename(img_path))[0])
                            val_set.add((seq, camera, frame))

        return train_set, val_set

    def _parse_poses(self, pose_file: Path) -> Dict[int, np.ndarray]:
        """Parse KITTI-360 poses.txt."""
        poses = {}
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 13:
                    continue

                frame_id = int(parts[0])
                values = list(map(float, parts[1:]))

                # Build 4x4 matrix
                pose = np.eye(4)
                pose[:3, :] = np.array(values).reshape(3, 4)

                poses[frame_id] = pose

        return poses

    def _build_sampling_buckets(self, frames_index: Dict) -> Dict:
        """
        Build sampling buckets based on building ratio.

        Buckets:
        - rich: building_ratio >= 0.30
        - mixed: 0.10 <= building_ratio < 0.30
        - road: building_ratio < 0.10
        """
        bucket_rich = []
        bucket_mixed = []
        bucket_road = []
        frame_metadata = {}

        building_class_id = SEMANTIC_CLASSES['building']

        # Collect all frames with semantic
        frames_to_process = []
        for seq_name, seq_frames in frames_index.items():
            for frame_idx, frame_info in seq_frames.items():
                has_semantic = frame_info.get('has_semantic', False)
                frames_to_process.append((seq_name, frame_idx, has_semantic))

        # Process frames
        for seq_name, frame_idx, has_semantic in tqdm(frames_to_process, desc="Computing building ratios"):
            semantic_path = self.root_dir / 'data_2d_semantics' / 'train' / seq_name / self.camera_name / 'semantic' / f'{int(frame_idx):010d}.png'

            building_ratio = None
            bucket = 'mixed'  # Default bucket when semantic data is missing

            if has_semantic and semantic_path.exists():
                # Load semantic and compute building ratio
                semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)

                if semantic is not None:
                    total_pixels = semantic.size
                    building_pixels = (semantic == building_class_id).sum()
                    building_ratio = float(building_pixels / total_pixels)

                    if building_ratio >= 0.30:
                        bucket = 'rich'
                    elif building_ratio >= 0.10:
                        bucket = 'mixed'
                    else:
                        bucket = 'road'

            # Classify into buckets
            frame_key = f"{seq_name}:{self.camera_name}:{int(frame_idx)}"

            if bucket == 'rich':
                bucket_rich.append([seq_name, self.camera_name, int(frame_idx)])
            elif bucket == 'mixed':
                bucket_mixed.append([seq_name, self.camera_name, int(frame_idx)])
            else:
                bucket_road.append([seq_name, self.camera_name, int(frame_idx)])

            frame_metadata[frame_key] = {
                'building_ratio': building_ratio,
                'bucket': bucket,
            }

        # Statistics
        total = len(bucket_rich) + len(bucket_mixed) + len(bucket_road)
        if total == 0:
            self.logger.warning("   No frames available for bucket statistics (dataset may be empty).")
        else:
            self.logger.info(f"   Bucket distribution:")
            self.logger.info(f"   - Rich:  {len(bucket_rich):5d} ({len(bucket_rich)/total*100:5.1f}%)")
            self.logger.info(f"   - Mixed: {len(bucket_mixed):5d} ({len(bucket_mixed)/total*100:5.1f}%)")
            self.logger.info(f"   - Road:  {len(bucket_road):5d} ({len(bucket_road)/total*100:5.1f}%)")

        return {
            'rich': bucket_rich,
            'mixed': bucket_mixed,
            'road': bucket_road,
            'frame_metadata': frame_metadata,
        }

    def _load_calibration(self) -> Dict:
        """Load camera calibration."""
        calib_dir = self.root_dir / 'calibration'
        calib = {}

        # Load perspective camera calibration
        perspective_file = calib_dir / 'perspective.txt'

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

                if key.startswith(f'P_rect_{self.camera_id:02d}'):
                    calib['P_rect'] = np.array(values).reshape(3, 4)
                    calib['K'] = calib['P_rect'][:, :3]
                elif key.startswith(f'R_rect_{self.camera_id:02d}'):
                    calib['R_rect'] = np.array(values).reshape(3, 3)

        # Load camera to velodyne calibration
        cam_to_velo_file = calib_dir / 'calib_cam_to_velo.txt'
        with open(cam_to_velo_file, 'r') as f:
            line = f.readline().strip()
            values = list(map(float, line.split()))

            if len(values) == 12:
                T_cam_to_velo = np.array(values).reshape(3, 4)
                calib['T_cam_to_velo_4x4'] = np.eye(4)
                calib['T_cam_to_velo_4x4'][:3, :] = T_cam_to_velo
                calib['T_velo_to_cam_4x4'] = np.linalg.inv(calib['T_cam_to_velo_4x4'])

        return calib

    # ==================== Stage 1: Depth Layer ====================

    def run_stage_1_depths(self):
        """
        Stage 1: Depth layer preprocessing (config-specific).

        Generates accumulated depth maps for current configuration.
        """
        self.logger.info("=" * 80)
        self.logger.info("Stage 1: Depth Layer Preprocessing")
        self.logger.info(f"Config: {self.config_id}")
        self.logger.info("=" * 80)

        # Create depth directory
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        # Generate metadata
        meta = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "config_id": self.config_id,
            "config_hash": self._compute_config_hash(),
            "parameters": {
                "root_dir": str(self.root_dir),
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "accumulation_frames": self.accumulation_frames,
                "depth_range": list(self.depth_range),
                "min_valid_points": self.min_valid_points,
            },
        }

        # Load and embed calibration
        calib_path = self.base_dir / 'calibration.json'
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                calib = json.load(f)
                meta['calibration'] = calib

        meta_path = self.depth_dir / 'meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        self.logger.info(f"✅ Metadata saved to {meta_path}")

        # Load frames index
        frames_index_path = self.base_dir / 'frames_index.json'
        if not frames_index_path.exists():
            self.logger.error(f"❌ Frames index not found at {frames_index_path}")
            self.logger.error(f"   Please run Stage 0 first!")
            return

        with open(frames_index_path, 'r') as f:
            frames_index = json.load(f)

        # Filter frames by split
        sequences_to_process = self._filter_sequences_by_split(frames_index)

        self.logger.info(f"Processing {len(sequences_to_process)} sequences...")

        # Process sequences in parallel
        if self.workers > 1:
            self.logger.info(f"Using {self.workers} parallel workers")
            with Pool(processes=self.workers) as pool:
                results = list(tqdm(
                    pool.imap(self._process_sequence_depths_wrapper, sequences_to_process),
                    total=len(sequences_to_process),
                    desc="Processing sequences"
                ))
        else:
            self.logger.info(f"Using single worker (sequential processing)")
            results = []
            for seq_info in tqdm(sequences_to_process, desc="Processing sequences"):
                results.append(self._process_sequence_depths(seq_info))

        # Aggregate statistics
        total_frames = sum(r['total_frames'] for r in results)
        successful_frames = sum(r['successful_frames'] for r in results)
        skipped_frames = sum(r['skipped_frames'] for r in results)

        self.logger.info("=" * 80)
        self.logger.info("✅ Stage 1 Complete: Depth maps generated")
        self.logger.info(f"   Total frames:      {total_frames}")
        self.logger.info(f"   Successful frames: {successful_frames}")
        self.logger.info(f"   Skipped frames:    {skipped_frames}")
        self.logger.info("=" * 80)

    def _filter_sequences_by_split(self, frames_index: Dict) -> List[Tuple]:
        """Filter sequences by split."""
        sequences_to_process = []

        for seq_name, seq_frames in frames_index.items():
            frame_list = []

            for frame_idx, frame_info in seq_frames.items():
                # Check split filter
                if self.split != 'all':
                    if frame_info['split'] != self.split:
                        continue

                # Check required files
                if not (frame_info['has_image'] and frame_info['has_lidar'] and frame_info['has_pose']):
                    continue

                frame_list.append(int(frame_idx))

            if frame_list:
                sequences_to_process.append((seq_name, sorted(frame_list)))

        return sequences_to_process

    def _process_sequence_depths_wrapper(self, seq_info: Tuple) -> Dict:
        """Wrapper for parallel processing."""
        return self._process_sequence_depths(seq_info)

    def _process_sequence_depths(self, seq_info: Tuple) -> Dict:
        """
        Process all frames in a sequence to generate accumulated depth maps.

        This is the core preprocessing function that:
        1. Loads LiDAR points from multiple frames
        2. Transforms points to current frame coordinate system
        3. Projects points to image plane
        4. Accumulates and averages overlapping points
        """
        seq_name, frame_indices = seq_info

        # Create sequence output directory
        seq_output_dir = self.depth_dir / 'sequences' / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)

        # Load poses for this sequence
        pose_file = self.root_dir / 'data_poses' / seq_name / 'poses.txt'
        poses = self._parse_poses(pose_file)

        # Load calibration
        calib_path = self.base_dir / 'calibration.json'
        with open(calib_path, 'r') as f:
            calib_json = json.load(f)
            K = np.array(calib_json['K'])
            T_velo_to_cam = np.array(calib_json['T_velo_to_cam_4x4'])

        stats = {
            'total_frames': len(frame_indices),
            'successful_frames': 0,
            'skipped_frames': 0,
        }

        for center_idx_in_list, center_frame_idx in enumerate(frame_indices):
            output_path = seq_output_dir / f'{center_frame_idx:010d}.npz'

            # Skip if already exists (unless overwrite)
            if output_path.exists() and not self.overwrite:
                stats['skipped_frames'] += 1
                continue

            # Get current frame pose
            if center_frame_idx not in poses:
                continue

            current_pose = poses[center_frame_idx]
            current_pose_inv = np.linalg.inv(current_pose)

            # Determine image resolution from RGB frame
            image_path = self.root_dir / 'data_2d_raw' / seq_name / self.camera_name / 'data_rect' / f'{center_frame_idx:010d}.png'
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.logger.warning(f"Skipping frame {seq_name}/{center_frame_idx:010d}: RGB image not found or unreadable")
                stats['skipped_frames'] += 1
                continue

            # Determine accumulation window
            start_idx_in_list = max(0, center_idx_in_list - self.accumulation_frames)
            end_idx_in_list = min(len(frame_indices), center_idx_in_list + self.accumulation_frames + 1)

            # Initialize depth accumulation arrays
            # Use original KITTI-360 resolution
            H, W = image.shape[:2]
            depth_map = np.zeros((H, W), dtype=np.float32)
            depth_count = np.zeros((H, W), dtype=np.int32)

            # Accumulate points from nearby frames
            for other_idx_in_list in range(start_idx_in_list, end_idx_in_list):
                other_frame_idx = frame_indices[other_idx_in_list]

                if other_frame_idx not in poses:
                    continue

                # Load LiDAR points
                lidar_path = self.root_dir / 'data_3d_raw' / seq_name / 'velodyne_points' / 'data' / f'{other_frame_idx:010d}.bin'
                if not lidar_path.exists():
                    continue

                points_velo = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)[:, :3]

                if len(points_velo) == 0:
                    continue

                # Transform: velo -> cam (other frame)
                points_hom = np.hstack([points_velo, np.ones((len(points_velo), 1))])
                points_cam_other = (T_velo_to_cam @ points_hom.T).T[:, :3]

                # Transform: cam (other) -> world -> cam (current)
                if other_idx_in_list != center_idx_in_list:
                    other_pose = poses[other_frame_idx]
                    points_hom = np.hstack([points_cam_other, np.ones((len(points_cam_other), 1))])
                    points_world = (other_pose @ points_hom.T).T[:, :3]
                    points_hom = np.hstack([points_world, np.ones((len(points_world), 1))])
                    points_cam = (current_pose_inv @ points_hom.T).T[:, :3]
                else:
                    points_cam = points_cam_other

                # Filter points behind camera
                valid_mask = points_cam[:, 2] > self.depth_range[0]
                points_cam = points_cam[valid_mask]

                if len(points_cam) == 0:
                    continue

                # Project to image
                points_2d_hom = (K @ points_cam.T).T
                depths = points_2d_hom[:, 2]

                valid_mask = depths > 1e-6
                points_2d = np.zeros((len(points_cam), 2))
                points_2d[valid_mask, 0] = points_2d_hom[valid_mask, 0] / depths[valid_mask]
                points_2d[valid_mask, 1] = points_2d_hom[valid_mask, 1] / depths[valid_mask]

                # Filter valid projections
                valid_mask = (
                    (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) &
                    (depths > self.depth_range[0]) & (depths < self.depth_range[1])
                )

                points_2d = points_2d[valid_mask].astype(np.int32)
                depths = depths[valid_mask]

                # Accumulate depths
                us = points_2d[:, 0].astype(np.intp)
                vs = points_2d[:, 1].astype(np.intp)
                np.add.at(depth_map, (vs, us), depths.astype(np.float32))
                np.add.at(depth_count, (vs, us), np.ones_like(vs, dtype=np.int32))

            # Average accumulated depths
            valid_mask = depth_count > 0
            depth_map[valid_mask] = depth_map[valid_mask] / depth_count[valid_mask]

            # Save
            valid_count = int(valid_mask.sum())
            coverage_ratio = float(valid_count / (H * W))

            np.savez_compressed(
                output_path,
                depth=depth_map.astype(np.float32),
                mask=valid_mask.astype(bool),
                valid_count=valid_count,
                coverage_ratio=coverage_ratio,
            )

            stats['successful_frames'] += 1

        return stats

    # ==================== Stage 2: Validation ====================

    def run_stage_2_validation(self):
        """
        Stage 2: Validation and visualization.

        Generates:
        - stats.json: Global statistics
        - visualization/: Sample visualizations
        - report.html: Quality check report
        """
        self.logger.info("=" * 80)
        self.logger.info("Stage 2: Validation and Visualization")
        self.logger.info("=" * 80)

        # Check if depth directory exists
        if not self.depth_dir.exists():
            self.logger.error(f"❌ Depth directory not found: {self.depth_dir}")
            self.logger.error(f"   Please run Stage 1 first!")
            return

        # Collect statistics
        self.logger.info("Collecting statistics...")
        stats = self._collect_statistics()

        stats_path = self.depth_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"✅ Statistics saved to {stats_path}")

        # Print summary
        self._print_statistics_summary(stats)

        # Generate visualizations (if not skipped)
        if not self.skip_visualization:
            self.logger.info("Generating visualizations...")
            self._generate_visualizations(stats)
        else:
            self.logger.info("Skipping visualizations (--skip-visualization)")

        self.logger.info("=" * 80)
        self.logger.info("✅ Stage 2 Complete: Validation finished")
        self.logger.info("=" * 80)

    def _collect_statistics(self) -> Dict:
        """Collect statistics from all generated depth maps."""
        coverage_ratios = []
        valid_counts = []
        low_quality_frames = []

        sequences_dir = self.depth_dir / 'sequences'

        for seq_dir in sequences_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name

            for depth_file in seq_dir.glob('*.npz'):
                frame_idx = int(depth_file.stem)

                data = np.load(depth_file)
                coverage_ratio = float(data['coverage_ratio'])
                valid_count = int(data['valid_count'])

                coverage_ratios.append(coverage_ratio)
                valid_counts.append(valid_count)

                # Flag low quality frames
                if coverage_ratio < 0.10 or valid_count < self.min_valid_points:
                    low_quality_frames.append({
                        'sequence': seq_name,
                        'frame_idx': frame_idx,
                        'coverage_ratio': coverage_ratio,
                        'valid_count': valid_count,
                    })

        if len(coverage_ratios) == 0:
            return {
                'total_frames': 0,
                'coverage': {},
                'valid_count': {},
                'low_quality_frames': [],
            }

        coverage_ratios = np.array(coverage_ratios)
        valid_counts = np.array(valid_counts)

        return {
            'total_frames': len(coverage_ratios),
            'coverage': {
                'mean': float(np.mean(coverage_ratios)),
                'median': float(np.median(coverage_ratios)),
                'std': float(np.std(coverage_ratios)),
                'min': float(np.min(coverage_ratios)),
                'max': float(np.max(coverage_ratios)),
                'percentiles': {
                    '5': float(np.percentile(coverage_ratios, 5)),
                    '25': float(np.percentile(coverage_ratios, 25)),
                    '75': float(np.percentile(coverage_ratios, 75)),
                    '95': float(np.percentile(coverage_ratios, 95)),
                },
            },
            'valid_count': {
                'mean': float(np.mean(valid_counts)),
                'median': float(np.median(valid_counts)),
                'std': float(np.std(valid_counts)),
                'min': int(np.min(valid_counts)),
                'max': int(np.max(valid_counts)),
                'percentiles': {
                    '5': int(np.percentile(valid_counts, 5)),
                    '25': int(np.percentile(valid_counts, 25)),
                    '75': int(np.percentile(valid_counts, 75)),
                    '95': int(np.percentile(valid_counts, 95)),
                },
            },
            'low_quality_frames': low_quality_frames[:100],  # Limit to 100 for JSON size
            'low_quality_count': len(low_quality_frames),
        }

    def _print_statistics_summary(self, stats: Dict):
        """Print statistics summary."""
        self.logger.info("")
        self.logger.info("Statistics Summary:")
        self.logger.info(f"  Total frames: {stats['total_frames']}")
        self.logger.info("")
        self.logger.info("  Coverage ratio:")
        self.logger.info(f"    Mean:   {stats['coverage']['mean']*100:6.2f}%")
        self.logger.info(f"    Median: {stats['coverage']['median']*100:6.2f}%")
        self.logger.info(f"    Std:    {stats['coverage']['std']*100:6.2f}%")
        self.logger.info(f"    Range:  [{stats['coverage']['min']*100:5.2f}%, {stats['coverage']['max']*100:5.2f}%]")
        self.logger.info("")
        self.logger.info("  Valid point count:")
        self.logger.info(f"    Mean:   {stats['valid_count']['mean']:8.1f}")
        self.logger.info(f"    Median: {stats['valid_count']['median']:8.1f}")
        self.logger.info(f"    Range:  [{stats['valid_count']['min']:6d}, {stats['valid_count']['max']:6d}]")
        self.logger.info("")
        self.logger.info(f"  Low quality frames: {stats['low_quality_count']} ({stats['low_quality_count']/stats['total_frames']*100:.1f}%)")

    def _generate_visualizations(self, stats: Dict):
        """Generate sample visualizations."""
        import matplotlib.pyplot as plt

        vis_dir = self.depth_dir / 'visualization'
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Sample frames for visualization
        sequences_dir = self.depth_dir / 'sequences'
        all_depth_files = list(sequences_dir.glob('*/*.npz'))

        if len(all_depth_files) == 0:
            self.logger.warning("No depth files found for visualization")
            return

        n_samples = min(20, len(all_depth_files))
        sample_indices = np.linspace(0, len(all_depth_files) - 1, n_samples, dtype=int)
        sample_files = [all_depth_files[i] for i in sample_indices]

        self.logger.info(f"Generating {n_samples} sample visualizations...")

        for i, depth_file in enumerate(tqdm(sample_files, desc="Generating visualizations")):
            seq_name = depth_file.parent.name
            frame_idx = int(depth_file.stem)

            # Load depth
            data = np.load(depth_file)
            depth_map = data['depth']
            mask = data['mask']

            # Load RGB image
            img_path = self.root_dir / 'data_2d_raw' / seq_name / self.camera_name / 'data_rect' / f'{frame_idx:010d}.png'
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(image)
            axes[0].set_title(f"RGB Image\n{seq_name} | Frame {frame_idx}")
            axes[0].axis('off')

            depth_vis = depth_map.copy()
            depth_vis[~mask] = 0
            im = axes[1].imshow(depth_vis, cmap='turbo', vmin=0, vmax=self.depth_range[1])
            axes[1].set_title(f"Accumulated Depth\nCoverage: {data['coverage_ratio']*100:.1f}% | Valid: {data['valid_count']}")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # Overlay
            alpha = 0.6
            overlay = image.copy().astype(np.float32) / 255.0
            depth_colored = plt.cm.turbo(depth_vis / self.depth_range[1])[:, :, :3]
            overlay[mask] = overlay[mask] * (1 - alpha) + depth_colored[mask] * alpha
            axes[2].imshow(overlay)
            axes[2].set_title("RGB + Depth Overlay")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(vis_dir / f'sample_{i:03d}_{seq_name}_{frame_idx}.png', dpi=100, bbox_inches='tight')
            plt.close()

        self.logger.info(f"✅ Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI-360 dataset for VGGT LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing (recommended for first run)
  python scripts/preprocess_kitti360_lora.py \\
      --config training/config/lora_kitti360_strategy_b.yaml \\
      --stages all

  # Only base layer (shared metadata)
  python scripts/preprocess_kitti360_lora.py \\
      --config training/config/lora_kitti360_strategy_b.yaml \\
      --stages base

  # Only depth layer (for new configuration)
  python scripts/preprocess_kitti360_lora.py \\
      --config training/config/lora_kitti360_strategy_b_af6.yaml \\
      --stages depths

  # Validation only
  python scripts/preprocess_kitti360_lora.py \\
      --config training/config/lora_kitti360_strategy_b.yaml \\
      --stages validation
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='training/config/lora_kitti360_strategy_b.yaml',
        help='Path to training config YAML (default: training/config/lora_kitti360_strategy_b.yaml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for preprocessed data (default: <root_dir>/precomputed)'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['base', 'depths', 'validation', 'all'],
        default=['all'],
        help='Stages to run (default: all)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing preprocessed files'
    )

    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split to preprocess (default: all)'
    )

    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip validation visualizations (faster)'
    )

    args = parser.parse_args()

    # Auto-detect output directory if not specified
    if args.output is None:
        train_config = OmegaConf.load(args.config)
        root_dir = train_config.data.train.dataset.dataset_configs[0].root_dir
        args.output = os.path.join(root_dir, 'precomputed')

    # Create preprocessor
    preprocessor = KITTI360Preprocessor(
        config_path=args.config,
        output_base_dir=args.output,
        workers=args.workers,
        overwrite=args.overwrite,
        split=args.split,
        skip_visualization=args.skip_visualization,
    )

    # Determine stages to run
    if 'all' in args.stages:
        stages = ['base', 'depths', 'validation']
    else:
        stages = args.stages

    # Run stages
    start_time = time.time()

    for stage in stages:
        if stage == 'base':
            preprocessor.run_stage_0_base()
        elif stage == 'depths':
            preprocessor.run_stage_1_depths()
        elif stage == 'validation':
            preprocessor.run_stage_2_validation()

    elapsed = time.time() - start_time

    preprocessor.logger.info("")
    preprocessor.logger.info("=" * 80)
    preprocessor.logger.info(f"✅ Preprocessing Complete!")
    preprocessor.logger.info(f"   Total time: {elapsed/60:.1f} minutes")
    preprocessor.logger.info(f"   Output: {preprocessor.output_base_dir}")
    preprocessor.logger.info("=" * 80)


if __name__ == '__main__':
    main()
