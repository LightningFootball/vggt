# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for ETH3D dataset support

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import glob

logger = logging.getLogger(__name__)


class ETH3DDataset(Dataset):
    """
    ETH3D Multi-view Stereo Dataset for training VGGT with LoRA.

    Dataset structure:
        root_dir/
        ├── multi_view_training_dslr_undistorted/
        │   └── {scene}/
        │       ├── images/dslr_images_undistorted/*.JPG
        │       └── dslr_calibration_undistorted/
        │           ├── cameras.txt
        │           ├── images.txt
        │           └── points3D.txt
        ├── multi_view_training_dslr_occlusion/
        │   └── {scene}/
        │       └── masks_for_images/dslr_images/*.png
        └── multi_view_training_dslr_scan_eval/  (not used for training)

    Args:
        root_dir: Path to ETH3D Stereo High-res_multi-view directory
        split: 'train' or 'val'
        img_size: Target image size (default: 518)
        sequence_length: Number of frames per sequence (default: 8)
        scenes: List of scene names to use (if None, use all available)
        min_depth: Minimum depth value for normalization
        max_depth: Maximum depth value for normalization
    """

    # All 13 scenes in ETH3D High-res multi-view
    ALL_SCENES = [
        'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
        'office', 'pipes', 'playground', 'relief', 'relief_2',
        'terrace', 'terrains', 'courtyard'
    ]

    # Scenes with strong building facades (prioritize for building LoRA)
    BUILDING_SCENES = ['facade', 'electro', 'office', 'terrace', 'delivery_area']

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        common_config: Optional[Dict] = None,
        img_size: Optional[int] = None,
        sequence_length: int = 8,
        scenes: Optional[List[str]] = None,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        use_building_scenes_only: bool = True,
        train_val_split: float = 0.85,
        **kwargs,  # Accept any additional arguments from dataloader
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split

        # Extract img_size from common_config if not provided directly
        if common_config is not None:
            self.img_size = common_config.get('img_size', img_size or 518)
        else:
            self.img_size = img_size or 518

        self.sequence_length = sequence_length
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Select scenes
        if scenes is not None:
            self.scenes = scenes
        elif use_building_scenes_only:
            self.scenes = self.BUILDING_SCENES
        else:
            self.scenes = self.ALL_SCENES

        # Load all sequences
        self.sequences = []
        for scene_name in self.scenes:
            scene_seqs = self._load_scene(scene_name)
            self.sequences.extend(scene_seqs)

        # Train/val split
        np.random.seed(42)
        np.random.shuffle(self.sequences)
        split_idx = int(len(self.sequences) * train_val_split)

        if split == 'train':
            self.sequences = self.sequences[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]

        logger.info(f"Loaded {len(self.sequences)} sequences from {len(self.scenes)} scenes for {split}")

    def _load_scene(self, scene_name: str) -> List[Dict]:
        """Load all camera data for a scene"""
        scene_dir = os.path.join(
            self.root_dir, 'multi_view_training_dslr_undistorted', scene_name
        )
        calib_dir = os.path.join(scene_dir, 'dslr_calibration_undistorted')
        # Note: images.txt contains relative paths like "dslr_images_undistorted/DSC_0422.JPG"
        # so we only need to join with the 'images' directory
        image_dir = os.path.join(scene_dir, 'images')
        mask_dir = os.path.join(
            self.root_dir, 'multi_view_training_dslr_occlusion', scene_name,
            'masks_for_images'
        )

        if not os.path.exists(calib_dir):
            logger.warning(f"Scene {scene_name} not found at {scene_dir}")
            return []

        # Parse COLMAP files
        cameras = self._parse_cameras(os.path.join(calib_dir, 'cameras.txt'))
        images_data = self._parse_images(os.path.join(calib_dir, 'images.txt'))
        points3d = self._parse_points3d(os.path.join(calib_dir, 'points3D.txt'))

        # Create frame list
        frames = []
        for img_id, img_data in images_data.items():
            img_path = os.path.join(image_dir, img_data['name'])
            # Mask paths: convert "dslr_images_undistorted/DSC_XXX.JPG" -> "dslr_images/DSC_XXX.png"
            mask_name = img_data['name'].replace('dslr_images_undistorted', 'dslr_images').replace('.JPG', '.png')
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(img_path):
                continue

            camera = cameras[img_data['camera_id']]
            frames.append({
                'scene': scene_name,
                'image_path': img_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'qvec': img_data['qvec'],  # Quaternion (qw, qx, qy, qz)
                'tvec': img_data['tvec'],  # Translation
                'camera_id': img_data['camera_id'],
                'intrinsics': camera['params'],  # [fx, fy, cx, cy]
                'width': camera['width'],
                'height': camera['height'],
                'point2d': img_data['xys'],  # 2D points for depth generation
                'point3d_ids': img_data['point3d_ids'],  # Corresponding 3D point IDs
            })

        # Create sequences by sliding window
        sequences = []
        for i in range(0, len(frames), self.sequence_length // 2):  # 50% overlap
            end_idx = min(i + self.sequence_length, len(frames))
            if end_idx - i < 3:  # Skip too short sequences
                break
            sequences.append({
                'scene': scene_name,
                'frames': frames[i:end_idx],
                'points3d': points3d,  # Share 3D points for depth generation
            })

        logger.info(f"Scene {scene_name}: {len(frames)} frames, {len(sequences)} sequences")
        return sequences

    def _parse_cameras(self, cameras_txt: str) -> Dict:
        """Parse COLMAP cameras.txt"""
        cameras = {}
        with open(cameras_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = list(map(float, parts[4:]))  # [fx, fy, cx, cy] for PINHOLE

                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params,
                }
        return cameras

    def _parse_images(self, images_txt: str) -> Dict:
        """Parse COLMAP images.txt"""
        images = {}
        with open(images_txt, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue

            parts = line.split()
            image_id = int(parts[0])
            qvec = np.array(list(map(float, parts[1:5])))  # qw, qx, qy, qz
            tvec = np.array(list(map(float, parts[5:8])))
            camera_id = int(parts[8])
            name = parts[9]

            # Next line contains 2D-3D correspondences
            i += 1
            points_line = lines[i].strip() if i < len(lines) else ""
            point_data = points_line.split()

            xys = []
            point3d_ids = []
            for j in range(0, len(point_data), 3):
                if j + 2 < len(point_data):
                    xys.append([float(point_data[j]), float(point_data[j+1])])
                    point3d_ids.append(int(point_data[j+2]))

            images[image_id] = {
                'name': name,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'xys': np.array(xys),
                'point3d_ids': np.array(point3d_ids),
            }
            i += 1

        return images

    def _parse_points3d(self, points3d_txt: str) -> Dict:
        """Parse COLMAP points3D.txt"""
        points3d = {}
        with open(points3d_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                point3d_id = int(parts[0])
                xyz = np.array(list(map(float, parts[1:4])))
                points3d[point3d_id] = xyz
        return points3d

    def _generate_depth_from_points(
        self,
        point2d: np.ndarray,
        point3d_ids: np.ndarray,
        points3d: Dict,
        qvec: np.ndarray,
        tvec: np.ndarray,
        width: int,
        height: int,
        radius: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a sparse depth map from 3D points and lightly densify by local splatting.

        Args:
            point2d: Nx2 COLMAP 2D keypoints (pixel coords in original image size)
            point3d_ids: Nx1 COLMAP 3D point ids (−1 if invalid)
            points3d: dict mapping id -> XYZ (world)
            qvec, tvec: camera pose (world-to-camera)
            width, height: original image size
            radius: splat radius in pixels (1 => 3x3)
        """
        # Initialize depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)

        # Quaternion to rotation matrix
        R = self._qvec2rotmat(qvec)

        # Transform points to camera space
        for i, p3d_id in enumerate(point3d_ids):
            if p3d_id == -1 or p3d_id not in points3d:
                continue

            # World to camera transformation
            p3d_world = points3d[p3d_id]
            p3d_cam = R @ p3d_world + tvec
            depth = p3d_cam[2]

            if depth <= 0:
                continue

            # Get 2D pixel coordinates
            x, y = point2d[i]
            x_int, y_int = int(round(x)), int(round(y))

            if 0 <= x_int < width and 0 <= y_int < height:
                # Splat to a small neighborhood to increase supervision density
                x0 = max(0, x_int - radius)
                x1 = min(width, x_int + radius + 1)
                y0 = max(0, y_int - radius)
                y1 = min(height, y_int + radius + 1)
                depth_map[y0:y1, x0:x1] = depth
                mask[y0:y1, x0:x1] = True

        return depth_map, mask

    @staticmethod
    def _qvec2rotmat(qvec):
        """Convert quaternion to rotation matrix"""
        qvec = qvec / np.linalg.norm(qvec)
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    @staticmethod
    def _rotmat2qvec(R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns a batch compatible with VGGT training.

        Args:
            idx: Can be either:
                - int: sequence index (for simple indexing)
                - tuple: (seq_index, img_per_seq, aspect_ratio) from DynamicDataloader

        Output format:
            {
                'images': [S, 3, H, W] - RGB images in range [0, 1]
                'depths': [S, H, W] - Depth maps
                'extrinsics': [S, 4, 4] - Camera extrinsics (world-to-camera)
                'intrinsics': [S, 3, 3] - Camera intrinsics
                'cam_points': [S, H, W, 3] - 3D points in camera space
                'world_points': [S, H, W, 3] - 3D points in world space
                'point_masks': [S, H, W] - Valid point mask
                'seq_name': str - Sequence name
            }
        """
        # Handle tuple indexing from DynamicDataloader
        if isinstance(idx, tuple):
            seq_idx = idx[0]  # Extract sequence index
            # img_per_seq and aspect_ratio are ignored for now
        else:
            seq_idx = idx

        seq = self.sequences[seq_idx]
        frames = seq['frames']
        points3d = seq['points3d']

        S = len(frames)
        H, W = self.img_size, self.img_size

        # Initialize outputs
        images = torch.zeros(S, 3, H, W, dtype=torch.float32)
        depths = torch.zeros(S, H, W, dtype=torch.float32)
        extrinsics = torch.zeros(S, 4, 4, dtype=torch.float32)
        intrinsics = torch.zeros(S, 3, 3, dtype=torch.float32)
        cam_points = torch.zeros(S, H, W, 3, dtype=torch.float32)
        world_points = torch.zeros(S, H, W, 3, dtype=torch.float32)
        point_masks = torch.zeros(S, H, W, dtype=torch.bool)

        for s, frame in enumerate(frames):
            # Load and resize image
            img = Image.open(frame['image_path']).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize((W, H), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            images[s] = torch.from_numpy(img_array).permute(2, 0, 1)

            # Load mask (occlusion mask)
            if frame['mask_path'] is not None and os.path.exists(frame['mask_path']):
                mask_img = Image.open(frame['mask_path']).convert('L')
                mask_img = mask_img.resize((W, H), Image.Resampling.NEAREST)
                mask_values = np.array(mask_img, dtype=np.uint8)
                # ETH3D mask semantics: 0=visible, 1=occluded, 2=undefined.
                # Only pixels marked visible should contribute to supervision to avoid
                # gradients from occlusions or invalid regions.
                mask_array = mask_values == 0
            else:
                mask_array = np.ones((H, W), dtype=bool)

            # Generate sparse depth from COLMAP points
            depth_orig, depth_mask_orig = self._generate_depth_from_points(
                frame['point2d'],
                frame['point3d_ids'],
                points3d,
                frame['qvec'],
                frame['tvec'],
                frame['width'],
                frame['height'],
            )

            # Resize depth and mask
            depth_img = Image.fromarray(depth_orig)
            depth_mask_img = Image.fromarray(depth_mask_orig.astype(np.uint8) * 255)
            depth_img = depth_img.resize((W, H), Image.Resampling.NEAREST)
            depth_mask_img = depth_mask_img.resize((W, H), Image.Resampling.NEAREST)
            depth_array = np.array(depth_img)
            depth_mask_array = np.array(depth_mask_img) > 128

            # Combine depth mask with occlusion mask
            final_mask = depth_mask_array & mask_array
            point_masks[s] = torch.from_numpy(final_mask)

            # Clamp and normalize depth
            depth_array = np.clip(depth_array, self.min_depth, self.max_depth)
            depths[s] = torch.from_numpy(depth_array)

            # Build extrinsics matrix (world-to-camera)
            R = self._qvec2rotmat(frame['qvec'])
            t = frame['tvec']
            extr = np.eye(4)
            extr[:3, :3] = R
            extr[:3, 3] = t
            extrinsics[s] = torch.from_numpy(extr.astype(np.float32))

            # Build intrinsics matrix (adjust for resize)
            fx, fy, cx, cy = frame['intrinsics']
            scale_x = W / orig_w
            scale_y = H / orig_h
            intr = np.array([
                [fx * scale_x, 0, cx * scale_x],
                [0, fy * scale_y, cy * scale_y],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics[s] = torch.from_numpy(intr)

            # Generate 3D point maps from depth
            # Create pixel grid
            u, v = np.meshgrid(np.arange(W), np.arange(H))

            # Unproject to camera space
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            z = depth_array
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            cam_pts = np.stack([x, y, z], axis=-1)
            cam_points[s] = torch.from_numpy(cam_pts.astype(np.float32))

            # Transform to world space
            R_inv = R.T
            t_inv = -R_inv @ t
            world_pts = cam_pts @ R_inv.T + t_inv[None, None, :]
            world_points[s] = torch.from_numpy(world_pts.astype(np.float32))

        return {
            'images': images,
            'depths': depths,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'cam_points': cam_points,
            'world_points': world_points,
            'point_masks': point_masks,
            'seq_name': f"{seq['scene']}_{idx}",
        }
