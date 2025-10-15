#!/usr/bin/env python3
"""
Analyze ETH3D dataset sparsity to verify depth supervision density.
"""

import numpy as np
import os
from collections import defaultdict

def parse_cameras(cameras_txt):
    """Parse COLMAP cameras.txt"""
    cameras = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            width = int(parts[2])
            height = int(parts[3])
            cameras[camera_id] = {'width': width, 'height': height}
    return cameras

def parse_images(images_txt):
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
        camera_id = int(parts[8])
        name = parts[9]

        # Next line contains 2D-3D correspondences
        i += 1
        points_line = lines[i].strip() if i < len(lines) else ""
        point_data = points_line.split()

        # Count valid 3D points (point3d_id != -1)
        num_points = 0
        for j in range(2, len(point_data), 3):  # Skip x, y, take point3d_id
            if j < len(point_data):
                point3d_id = int(point_data[j])
                if point3d_id != -1:
                    num_points += 1

        images[image_id] = {
            'name': name,
            'camera_id': camera_id,
            'num_2d3d_points': num_points
        }
        i += 1

    return images

def analyze_scene(scene_dir, scene_name):
    """Analyze one scene"""
    calib_dir = os.path.join(scene_dir, 'dslr_calibration_undistorted')

    cameras_txt = os.path.join(calib_dir, 'cameras.txt')
    images_txt = os.path.join(calib_dir, 'images.txt')
    points3d_txt = os.path.join(calib_dir, 'points3D.txt')

    if not all(os.path.exists(f) for f in [cameras_txt, images_txt, points3d_txt]):
        print(f"⚠️  Scene {scene_name}: Missing COLMAP files")
        return None

    # Parse data
    cameras = parse_cameras(cameras_txt)
    images = parse_images(images_txt)

    # Count total 3D points
    with open(points3d_txt, 'r') as f:
        num_3d_points = sum(1 for line in f if line.strip() and not line.startswith('#'))

    # Analyze per-image sparsity
    point_counts = []
    coverage_ratios = []

    for img_id, img_data in images.items():
        camera = cameras[img_data['camera_id']]
        total_pixels = camera['width'] * camera['height']
        num_points = img_data['num_2d3d_points']
        coverage = num_points / total_pixels * 100

        point_counts.append(num_points)
        coverage_ratios.append(coverage)

    results = {
        'scene_name': scene_name,
        'num_images': len(images),
        'num_3d_points': num_3d_points,
        'avg_points_per_image': np.mean(point_counts),
        'median_points_per_image': np.median(point_counts),
        'min_points_per_image': np.min(point_counts),
        'max_points_per_image': np.max(point_counts),
        'avg_coverage': np.mean(coverage_ratios),
        'median_coverage': np.median(coverage_ratios),
        'image_resolution': f"{cameras[0]['width']}x{cameras[0]['height']}",
        'total_pixels': cameras[0]['width'] * cameras[0]['height']
    }

    return results

def main():
    eth3d_root = '/home/zerun/data/dataset/ETH3D/Stereo/High-res_multi-view'
    undistorted_dir = os.path.join(eth3d_root, 'multi_view_training_dslr_undistorted')

    # BUILDING_SCENES as defined in your dataset
    building_scenes = ['facade', 'electro', 'office', 'terrace', 'delivery_area']

    print("=" * 80)
    print("ETH3D Dataset Sparsity Analysis")
    print("=" * 80)
    print()

    all_results = []

    for scene_name in building_scenes:
        scene_dir = os.path.join(undistorted_dir, scene_name)
        if not os.path.exists(scene_dir):
            print(f"⚠️  Scene {scene_name} not found")
            continue

        results = analyze_scene(scene_dir, scene_name)
        if results:
            all_results.append(results)

            print(f"📊 Scene: {scene_name}")
            print(f"   Images: {results['num_images']}")
            print(f"   3D Points: {results['num_3d_points']:,}")
            print(f"   Image Resolution: {results['image_resolution']} ({results['total_pixels']:,} pixels)")
            print(f"   Points per Image: {results['avg_points_per_image']:.0f} (avg), {results['median_points_per_image']:.0f} (median)")
            print(f"   Coverage: {results['avg_coverage']:.4f}% (avg), {results['median_coverage']:.4f}% (median)")
            print(f"   Min/Max points: {results['min_points_per_image']:.0f} / {results['max_points_per_image']:.0f}")
            print()

    if not all_results:
        print("❌ No valid scenes found")
        return

    # Overall statistics
    print("=" * 80)
    print("📈 Overall Statistics (Building Scenes)")
    print("=" * 80)

    avg_coverage_all = np.mean([r['avg_coverage'] for r in all_results])
    avg_points_all = np.mean([r['avg_points_per_image'] for r in all_results])

    print(f"Average coverage across all scenes: {avg_coverage_all:.4f}%")
    print(f"Average points per image: {avg_points_all:.0f}")
    print()

    # Compare with target for VGGT training
    print("=" * 80)
    print("🔍 Comparison with VGGT Training Requirements")
    print("=" * 80)

    # After resize to 518x518
    target_size = 518 * 518
    expected_points_per_image = avg_points_all * (target_size / all_results[0]['total_pixels'])

    print(f"After resize to 518x518 ({target_size:,} pixels):")
    print(f"   Expected points per image: {expected_points_per_image:.0f}")
    print(f"   Expected coverage: {expected_points_per_image / target_size * 100:.4f}%")
    print()

    print("⚠️  VGGT Training Thresholds:")
    print(f"   Camera loss valid frame: > 10 points (Current: {expected_points_per_image:.0f}) {'✅ PASS' if expected_points_per_image > 10 else '❌ FAIL'}")
    print(f"   Depth loss skip batch: > 10 points (Current: {expected_points_per_image:.0f}) {'✅ PASS' if expected_points_per_image > 10 else '❌ FAIL'}")
    print(f"   Co3D typical coverage: ~90% dense depth")
    print(f"   ETH3D actual coverage: {expected_points_per_image / target_size * 100:.4f}% sparse depth")
    print()

    # Verdict
    print("=" * 80)
    print("🎯 Verdict")
    print("=" * 80)

    if avg_coverage_all < 1.0:
        print("❌ CRITICAL: ETH3D provides SPARSE supervision (<1% coverage)")
        print("   This is fundamentally different from Co3D/VKitti dense depth.")
        print()
        print("   Recommendations:")
        print("   1. Use datasets with dense depth (Replica, ScanNet, Matterport3D)")
        print("   2. Or generate pseudo-dense depth using monocular depth models")
        print("   3. Or modify loss to only supervise sparse keypoints (major code change)")
    else:
        print("✅ Coverage sufficient for dense supervision")

    print("=" * 80)

if __name__ == '__main__':
    main()
