#!/usr/bin/env python3
"""
Visualize KITTI-360 evaluation metrics from CSV file.

This script generates comprehensive charts to analyze LoRA training progress
on KITTI-360 building-focused metrics.

Usage:
    python scripts/visualize_kitti360_metrics.py --csv evaluate/kitti360_b/metrics_overall.csv --output evaluate/kitti360_b/plots
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Set better default style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def load_csv(csv_path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    """Load CSV file and return checkpoint names and metrics."""
    checkpoints = []
    metrics = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint = row['checkpoint']
            checkpoints.append(checkpoint)

            # Convert all values to float (except checkpoint name)
            metric_dict = {}
            for key, value in row.items():
                if key != 'checkpoint':
                    try:
                        metric_dict[key] = float(value)
                    except (ValueError, TypeError):
                        metric_dict[key] = np.nan
            metrics.append(metric_dict)

    return checkpoints, metrics


def get_checkpoint_numbers(checkpoints: List[str]) -> np.ndarray:
    """Extract checkpoint numbers for x-axis."""
    numbers = []
    for ckpt in checkpoints:
        if ckpt == 'baseline':
            numbers.append(-1)  # Baseline at position -1
        else:
            # Extract number from checkpoint_X
            num = int(ckpt.split('_')[1])
            numbers.append(num)
    return np.array(numbers)


def plot_main_depth_metrics(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 1: Main depth metrics trend."""
    x = get_checkpoint_numbers(checkpoints)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Subplot 1: Relative error (lower is better)
    building_abs_rel = [m['depth.building.abs_rel.mean'] for m in metrics]
    full_abs_rel = [m['depth.full.abs_rel.mean'] for m in metrics]

    ax1.plot(x, building_abs_rel, 'o-', linewidth=2, markersize=4, label='Building Region', color='#e74c3c')
    ax1.plot(x, full_abs_rel, 's-', linewidth=2, markersize=4, label='Full Image', color='#3498db')

    # Mark baseline
    baseline_idx = np.where(x == -1)[0][0]
    ax1.axhline(y=building_abs_rel[baseline_idx], color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=full_abs_rel[baseline_idx], color='#3498db', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('Absolute Relative Error (lower is better)')
    ax1.set_title('Depth Estimation Error: Buildings vs Full Image')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Delta1 accuracy (higher is better)
    building_delta1 = [m['depth.building.delta1.mean'] for m in metrics]
    full_delta1 = [m['depth.full.delta1.mean'] for m in metrics]

    ax2.plot(x, building_delta1, 'o-', linewidth=2, markersize=4, label='Building Region', color='#e74c3c')
    ax2.plot(x, full_delta1, 's-', linewidth=2, markersize=4, label='Full Image', color='#3498db')

    # Mark baseline
    ax2.axhline(y=building_delta1[baseline_idx], color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=full_delta1[baseline_idx], color='#3498db', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('δ < 1.25 Accuracy (higher is better)')
    ax2.set_title('Depth Accuracy (δ < 1.25): Buildings vs Full Image')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '01_main_depth_metrics.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '01_main_depth_metrics.png'}")


def plot_building_vs_nonbuilding(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 2: Building vs Non-building comparison."""
    x = get_checkpoint_numbers(checkpoints)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Metric pairs to compare
    metric_configs = [
        ('abs_rel', 'Absolute Relative Error', 'lower', axes[0, 0]),
        ('delta1', 'δ < 1.25 Accuracy', 'higher', axes[0, 1]),
        ('rmse', 'RMSE (meters)', 'lower', axes[1, 0]),
        ('mae', 'MAE (meters)', 'lower', axes[1, 1]),
    ]

    baseline_idx = np.where(x == -1)[0][0]

    for metric, title, direction, ax in metric_configs:
        building_key = f'depth.building.{metric}.mean'
        non_building_key = f'depth.non_building.{metric}.mean'

        building_vals = [m[building_key] for m in metrics]
        non_building_vals = [m[non_building_key] for m in metrics]

        ax.plot(x, building_vals, 'o-', linewidth=2, markersize=4,
                label='Building Region', color='#e74c3c')
        ax.plot(x, non_building_vals, 's-', linewidth=2, markersize=4,
                label='Non-Building Region', color='#27ae60')

        # Mark baseline
        ax.axhline(y=building_vals[baseline_idx], color='#e74c3c',
                   linestyle='--', alpha=0.5, linewidth=1, label='Baseline (Building)')
        ax.axhline(y=non_building_vals[baseline_idx], color='#27ae60',
                   linestyle='--', alpha=0.5, linewidth=1, label='Baseline (Non-Building)')

        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel(f'{title} ({direction} is better)')
        ax.set_title(f'{title}: Building vs Non-Building')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Building-Specific Depth Metrics Comparison', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / '02_building_vs_nonbuilding.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '02_building_vs_nonbuilding.png'}")


def plot_geometry_quality(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 3: Normal and planarity metrics (geometric quality)."""
    x = get_checkpoint_numbers(checkpoints)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    baseline_idx = np.where(x == -1)[0][0]

    # Normal angular mean error (lower is better)
    normal_mean = [m['normals.building.angular_mean.mean'] for m in metrics]
    axes[0, 0].plot(x, normal_mean, 'o-', linewidth=2, markersize=4, color='#9b59b6')
    axes[0, 0].axhline(y=normal_mean[baseline_idx], color='#9b59b6',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 0].set_xlabel('Checkpoint Number')
    axes[0, 0].set_ylabel('Angular Error (degrees)')
    axes[0, 0].set_title('Surface Normal Mean Angular Error (lower is better)')
    axes[0, 0].grid(True, alpha=0.3)

    # Normal inlier ratios (higher is better)
    inlier_11 = [m['normals.building.inlier_11_25.mean'] for m in metrics]
    inlier_22 = [m['normals.building.inlier_22_5.mean'] for m in metrics]
    inlier_30 = [m['normals.building.inlier_30.mean'] for m in metrics]

    axes[0, 1].plot(x, inlier_11, 'o-', linewidth=2, markersize=4, label='< 11.25°', color='#e67e22')
    axes[0, 1].plot(x, inlier_22, 's-', linewidth=2, markersize=4, label='< 22.5°', color='#f39c12')
    axes[0, 1].plot(x, inlier_30, '^-', linewidth=2, markersize=4, label='< 30°', color='#f1c40f')
    axes[0, 1].set_xlabel('Checkpoint Number')
    axes[0, 1].set_ylabel('Inlier Ratio')
    axes[0, 1].set_title('Surface Normal Accuracy (higher is better)')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Planarity residual mean (lower is better)
    plane_residual = [m['planarity.building.residual_mean.mean'] for m in metrics]
    axes[1, 0].plot(x, plane_residual, 'o-', linewidth=2, markersize=4, color='#16a085')
    axes[1, 0].axhline(y=plane_residual[baseline_idx], color='#16a085',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Checkpoint Number')
    axes[1, 0].set_ylabel('Mean Residual (meters)')
    axes[1, 0].set_title('Plane Fitting Residual (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)

    # Planarity inlier ratio (higher is better)
    plane_inlier = [m['planarity.building.inlier_ratio.mean'] for m in metrics]
    axes[1, 1].plot(x, plane_inlier, 'o-', linewidth=2, markersize=4, color='#1abc9c')
    axes[1, 1].axhline(y=plane_inlier[baseline_idx], color='#1abc9c',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Checkpoint Number')
    axes[1, 1].set_ylabel('Inlier Ratio (residual < 0.1m)')
    axes[1, 1].set_title('Planarity Inlier Ratio (higher is better)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Geometric Quality: Surface Normals and Planarity', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / '03_geometry_quality.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '03_geometry_quality.png'}")


def plot_pose_errors(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 4: Camera pose estimation errors."""
    x = get_checkpoint_numbers(checkpoints)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    baseline_idx = np.where(x == -1)[0][0]

    # ATE translation (lower is better)
    ate_trans = [m['pose.ate_trans.mean'] for m in metrics]
    axes[0, 0].plot(x, ate_trans, 'o-', linewidth=2, markersize=4, color='#c0392b')
    axes[0, 0].axhline(y=ate_trans[baseline_idx], color='#c0392b',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 0].set_xlabel('Checkpoint Number')
    axes[0, 0].set_ylabel('Translation Error (units)')
    axes[0, 0].set_title('Absolute Trajectory Error - Translation (lower is better)')
    axes[0, 0].grid(True, alpha=0.3)

    # ATE rotation (lower is better)
    ate_rot = [m['pose.ate_rot_deg.mean'] for m in metrics]
    axes[0, 1].plot(x, ate_rot, 's-', linewidth=2, markersize=4, color='#8e44ad')
    axes[0, 1].axhline(y=ate_rot[baseline_idx], color='#8e44ad',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Checkpoint Number')
    axes[0, 1].set_ylabel('Rotation Error (degrees)')
    axes[0, 1].set_title('Absolute Trajectory Error - Rotation (lower is better)')
    axes[0, 1].grid(True, alpha=0.3)

    # RPE translation (lower is better)
    rpe_trans = [m['pose.rpe_trans.mean'] for m in metrics]
    axes[1, 0].plot(x, rpe_trans, 'o-', linewidth=2, markersize=4, color='#d35400')
    axes[1, 0].axhline(y=rpe_trans[baseline_idx], color='#d35400',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Checkpoint Number')
    axes[1, 0].set_ylabel('Translation Error (units)')
    axes[1, 0].set_title('Relative Pose Error - Translation (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)

    # RPE rotation (lower is better)
    rpe_rot = [m['pose.rpe_rot_deg.mean'] for m in metrics]
    axes[1, 1].plot(x, rpe_rot, 's-', linewidth=2, markersize=4, color='#2980b9')
    axes[1, 1].axhline(y=rpe_rot[baseline_idx], color='#2980b9',
                       linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Checkpoint Number')
    axes[1, 1].set_ylabel('Rotation Error (degrees)')
    axes[1, 1].set_title('Relative Pose Error - Rotation (lower is better)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Camera Pose Estimation Quality', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / '04_pose_errors.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '04_pose_errors.png'}")


def plot_radar_comparison(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 5: Radar chart comparing baseline vs best checkpoint."""
    # Find baseline and checkpoint_20 (the breakthrough point)
    baseline_idx = checkpoints.index('baseline')

    # Find checkpoint_20 or the best performing checkpoint
    try:
        best_idx = checkpoints.index('checkpoint_20')
    except ValueError:
        # If checkpoint_20 doesn't exist, find the checkpoint with best building delta1
        building_delta1 = [m['depth.building.delta1.mean'] for m in metrics]
        best_idx = np.argmax(building_delta1[1:]) + 1  # Skip baseline

    # Metrics for radar chart (normalized to [0, 1], higher is better)
    metric_configs = [
        ('depth.building.delta1.mean', 'Building\nDepth Acc.', False),  # higher is better
        ('normals.building.inlier_30.mean', 'Normal\nAcc. (<30°)', False),  # higher is better
        ('planarity.building.inlier_ratio.mean', 'Planarity', False),  # higher is better
        ('photometric.building.inlier_ratio.mean', 'Photometric\nConsistency', False),  # higher is better
        ('depth.building.abs_rel.mean', 'Building\nDepth Quality', True),  # lower is better, invert
        ('pose.ate_trans.mean', 'Pose\nAccuracy', True),  # lower is better, invert
    ]

    # Extract and normalize values
    num_vars = len(metric_configs)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    baseline_values = []
    best_values = []
    labels = []

    for key, label, invert in metric_configs:
        baseline_val = metrics[baseline_idx][key]
        best_val = metrics[best_idx][key]

        # Normalize to [0, 1]
        if invert:
            # For metrics where lower is better, use reciprocal
            max_val = max(baseline_val, best_val)
            baseline_norm = 1.0 - (baseline_val / max_val)
            best_norm = 1.0 - (best_val / max_val)
        else:
            # For metrics where higher is better
            max_val = max(baseline_val, best_val)
            baseline_norm = baseline_val / max_val if max_val > 0 else 0
            best_norm = best_val / max_val if max_val > 0 else 0

        baseline_values.append(baseline_norm)
        best_values.append(best_norm)
        labels.append(label)

    # Close the plot
    baseline_values += baseline_values[:1]
    best_values += best_values[:1]
    angles += angles[:1]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#95a5a6')
    ax.fill(angles, baseline_values, alpha=0.15, color='#95a5a6')

    ax.plot(angles, best_values, 'o-', linewidth=2, label=f'{checkpoints[best_idx]}', color='#e74c3c')
    ax.fill(angles, best_values, alpha=0.25, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Overall Performance Comparison\n(Normalized, higher is better)',
              size=13, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / '05_radar_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '05_radar_comparison.png'}")


def plot_training_stability(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 6: Training stability analysis."""
    x = get_checkpoint_numbers(checkpoints)

    # Skip baseline for this plot
    train_indices = x > -1
    x_train = x[train_indices]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Subplot 1: Normalized depth metrics
    ax1 = fig.add_subplot(gs[0, :])

    depth_metrics_keys = [
        'depth.building.abs_rel.mean',
        'depth.building.delta1.mean',
        'depth.building.rmse.mean',
        'normals.building.angular_mean.mean',
    ]
    depth_metrics_labels = [
        'Building Abs Rel (inverted)',
        'Building δ<1.25',
        'Building RMSE (inverted)',
        'Normal Angular Error (inverted)',
    ]

    colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']

    for key, label, color in zip(depth_metrics_keys, depth_metrics_labels, colors):
        values = np.array([m[key] for m in metrics])[train_indices]

        # Normalize and invert if needed
        if 'delta' in key or 'inlier' in key:
            normalized = values / np.max(values)
        else:
            # Invert metrics where lower is better
            normalized = 1.0 - (values / np.max(values))

        ax1.plot(x_train, normalized, 'o-', linewidth=2, markersize=3,
                label=label, color=color, alpha=0.8)

    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('Normalized Score (higher is better)')
    ax1.set_title('Training Progress: Normalized Metrics')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # Subplot 2: Checkpoint 20 transition analysis
    ax2 = fig.add_subplot(gs[1, 0])

    # Focus on checkpoints around 20
    focus_range = (x >= 15) & (x <= 25)
    x_focus = x[focus_range]

    building_abs_rel = np.array([m['depth.building.abs_rel.mean'] for m in metrics])[focus_range]
    building_delta1 = np.array([m['depth.building.delta1.mean'] for m in metrics])[focus_range]

    ax2_twin = ax2.twinx()

    line1 = ax2.plot(x_focus, building_abs_rel, 'o-', linewidth=2, markersize=5,
                     label='Abs Rel Error', color='#e74c3c')
    line2 = ax2_twin.plot(x_focus, building_delta1, 's-', linewidth=2, markersize=5,
                          label='δ<1.25 Accuracy', color='#3498db')

    ax2.axvline(x=20, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(20, ax2.get_ylim()[1] * 0.95, 'Checkpoint 20\nBreakthrough',
             ha='center', fontsize=9, color='#f39c12', weight='bold')

    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Abs Rel Error (lower better)', color='#e74c3c')
    ax2_twin.set_ylabel('δ<1.25 Accuracy (higher better)', color='#3498db')
    ax2.set_title('Transition Analysis: Checkpoints 15-25')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#3498db')
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')

    # Subplot 3: Coefficient of variation (stability measure)
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate rolling coefficient of variation for key metrics
    window = 5
    building_delta1_all = np.array([m['depth.building.delta1.mean'] for m in metrics])[train_indices]

    cv_values = []
    cv_x = []
    for i in range(len(building_delta1_all) - window + 1):
        window_data = building_delta1_all[i:i+window]
        cv = np.std(window_data) / np.mean(window_data) if np.mean(window_data) > 0 else 0
        cv_values.append(cv)
        cv_x.append(x_train[i + window // 2])

    ax3.plot(cv_x, cv_values, 'o-', linewidth=2, markersize=4, color='#16a085')
    ax3.axhline(y=0.05, color='#27ae60', linestyle='--', linewidth=1,
                label='Stable threshold (5%)', alpha=0.7)
    ax3.set_xlabel('Checkpoint Number')
    ax3.set_ylabel('Coefficient of Variation (5-checkpoint window)')
    ax3.set_title('Training Stability: δ<1.25 Metric Variability')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Training Stability and Convergence Analysis', fontsize=14, y=0.995)
    plt.savefig(output_dir / '06_training_stability.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '06_training_stability.png'}")


def plot_comprehensive_score(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Plot 7: Comprehensive score trend."""
    x = get_checkpoint_numbers(checkpoints)

    # Calculate comprehensive score
    # Weights: depth accuracy (30%), depth quality (20%), normals (20%),
    #          planarity (15%), photometric (15%)
    scores = []
    for m in metrics:
        score = (
            m['depth.building.delta1.mean'] * 0.30 +
            (1.0 - min(m['depth.building.abs_rel.mean'], 1.0)) * 0.20 +
            m['normals.building.inlier_30.mean'] * 0.20 +
            m['planarity.building.inlier_ratio.mean'] * 0.15 +
            m['photometric.building.inlier_ratio.mean'] * 0.15
        )
        scores.append(score)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Subplot 1: Overall score trend
    ax1.plot(x, scores, 'o-', linewidth=2.5, markersize=5, color='#2c3e50')
    ax1.fill_between(x, scores, alpha=0.2, color='#2c3e50')

    # Mark baseline
    baseline_idx = np.where(x == -1)[0][0]
    ax1.axhline(y=scores[baseline_idx], color='#e74c3c', linestyle='--',
                linewidth=2, label=f'Baseline Score: {scores[baseline_idx]:.3f}', alpha=0.7)

    # Mark best checkpoint
    best_idx = np.argmax(scores)
    ax1.plot(x[best_idx], scores[best_idx], 'r*', markersize=20,
             label=f'Best: {checkpoints[best_idx]} ({scores[best_idx]:.3f})', zorder=5)

    # Annotate checkpoint 20
    if 'checkpoint_20' in checkpoints:
        ckpt20_idx = checkpoints.index('checkpoint_20')
        ax1.annotate('Breakthrough\nCheckpoint 20',
                    xy=(x[ckpt20_idx], scores[ckpt20_idx]),
                    xytext=(x[ckpt20_idx] + 5, scores[ckpt20_idx] - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2),
                    fontsize=10, color='#f39c12', weight='bold')

    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('Comprehensive Score')
    ax1.set_title('Overall Training Quality Score\n(Weighted: Depth 50%, Normals 20%, Planarity 15%, Photometric 15%)')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Score improvement over baseline
    score_improvement = np.array(scores) - scores[baseline_idx]
    improvement_pct = (score_improvement / scores[baseline_idx]) * 100

    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvement_pct]

    ax2.bar(x, improvement_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Improvement over Baseline (%)')
    ax2.set_title('Relative Performance Improvement')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate best improvement
    best_imp_idx = np.argmax(improvement_pct)
    ax2.text(x[best_imp_idx], improvement_pct[best_imp_idx] + 2,
             f'{checkpoints[best_imp_idx]}\n+{improvement_pct[best_imp_idx]:.1f}%',
             ha='center', fontsize=9, weight='bold', color='#27ae60')

    plt.tight_layout()
    plt.savefig(output_dir / '07_comprehensive_score.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / '07_comprehensive_score.png'}")


def generate_summary_report(checkpoints: List[str], metrics: List[Dict], output_dir: Path):
    """Generate a text summary report."""
    baseline_idx = checkpoints.index('baseline')

    # Find best checkpoint by comprehensive score
    scores = []
    for m in metrics:
        score = (
            m['depth.building.delta1.mean'] * 0.30 +
            (1.0 - min(m['depth.building.abs_rel.mean'], 1.0)) * 0.20 +
            m['normals.building.inlier_30.mean'] * 0.20 +
            m['planarity.building.inlier_ratio.mean'] * 0.15 +
            m['photometric.building.inlier_ratio.mean'] * 0.15
        )
        scores.append(score)

    best_idx = np.argmax(scores)

    report = []
    report.append("=" * 80)
    report.append("KITTI-360 LoRA Training Evaluation Summary")
    report.append("=" * 80)
    report.append("")

    report.append(f"Total Checkpoints Evaluated: {len(checkpoints)}")
    report.append(f"Best Checkpoint: {checkpoints[best_idx]}")
    report.append(f"Best Comprehensive Score: {scores[best_idx]:.4f}")
    report.append(f"Baseline Score: {scores[baseline_idx]:.4f}")
    report.append(f"Improvement: {((scores[best_idx] - scores[baseline_idx]) / scores[baseline_idx] * 100):.2f}%")
    report.append("")

    report.append("-" * 80)
    report.append("Key Metrics Comparison (Baseline vs Best)")
    report.append("-" * 80)

    key_metrics = [
        ('depth.building.abs_rel.mean', 'Building Abs Rel Error', 'lower'),
        ('depth.building.delta1.mean', 'Building δ<1.25 Accuracy', 'higher'),
        ('depth.building.rmse.mean', 'Building RMSE', 'lower'),
        ('normals.building.angular_mean.mean', 'Normal Angular Error (deg)', 'lower'),
        ('planarity.building.inlier_ratio.mean', 'Planarity Inlier Ratio', 'higher'),
        ('pose.ate_trans.mean', 'ATE Translation', 'lower'),
        ('pose.ate_rot_deg.mean', 'ATE Rotation (deg)', 'lower'),
    ]

    for key, name, direction in key_metrics:
        baseline_val = metrics[baseline_idx][key]
        best_val = metrics[best_idx][key]

        if direction == 'lower':
            improvement = ((baseline_val - best_val) / baseline_val * 100)
            symbol = '↓'
        else:
            improvement = ((best_val - baseline_val) / baseline_val * 100)
            symbol = '↑'

        report.append(f"{name:40s} | Baseline: {baseline_val:8.4f} | Best: {best_val:8.4f} | {symbol} {improvement:+6.2f}%")

    report.append("")
    report.append("-" * 80)
    report.append("Training Progress Notes")
    report.append("-" * 80)

    # Analyze checkpoint 20 transition
    if 'checkpoint_20' in checkpoints:
        ckpt20_idx = checkpoints.index('checkpoint_20')
        ckpt19_idx = checkpoints.index('checkpoint_19') if 'checkpoint_19' in checkpoints else None

        if ckpt19_idx is not None:
            delta1_before = metrics[ckpt19_idx]['depth.building.delta1.mean']
            delta1_after = metrics[ckpt20_idx]['depth.building.delta1.mean']

            report.append(f"• Checkpoint 19→20 transition shows significant change:")
            report.append(f"  Building δ<1.25: {delta1_before:.4f} → {delta1_after:.4f} ({((delta1_after - delta1_before) / delta1_before * 100):+.2f}%)")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_path = output_dir / 'evaluation_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved: {report_path}")

    # Print to console
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Visualize KITTI-360 evaluation metrics')
    parser.add_argument('--csv', type=str, default='evaluate/kitti360_b/metrics_overall.csv',
                       help='Path to metrics CSV file')
    parser.add_argument('--output', type=str, default='evaluate/kitti360_b/plots',
                       help='Output directory for plots')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from: {csv_path}")
    checkpoints, metrics = load_csv(csv_path)
    print(f"Loaded {len(checkpoints)} checkpoints")

    print("\nGenerating visualizations...")
    print("-" * 80)

    # Generate all plots
    plot_main_depth_metrics(checkpoints, metrics, output_dir)
    plot_building_vs_nonbuilding(checkpoints, metrics, output_dir)
    plot_geometry_quality(checkpoints, metrics, output_dir)
    plot_pose_errors(checkpoints, metrics, output_dir)
    plot_radar_comparison(checkpoints, metrics, output_dir)
    plot_training_stability(checkpoints, metrics, output_dir)
    plot_comprehensive_score(checkpoints, metrics, output_dir)

    print("-" * 80)
    print("\nGenerating summary report...")
    generate_summary_report(checkpoints, metrics, output_dir)

    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
