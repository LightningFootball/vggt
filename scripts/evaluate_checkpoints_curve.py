#!/usr/bin/env python3
"""
Evaluate all checkpoints in a training run and plot metric curves.

This script evaluates all checkpoints from a LoRA training experiment,
computes depth metrics for each checkpoint, and generates visualization
plots to observe the fine-tuning progress over epochs.

Usage:
    python scripts/evaluate_checkpoints_curve.py --ckpt-dir logs/lora_eth3d_strategy_b_r16/ckpts --config lora_eth3d_strategy_b
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from training.data.datasets.eth3d import ETH3DDataset
from training.lora_utils import apply_lora_to_model

try:
    from peft import PeftModel  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CheckpointEvaluator:
    """Evaluates multiple checkpoints and tracks metrics over epochs"""

    def __init__(
        self,
        model_base: torch.nn.Module,
        dataset: ETH3DDataset,
        device: str = 'cuda',
        align_mode: str = 'scale_shift',
    ):
        self.model_base = model_base
        self.dataset = dataset
        self.device = device
        self.align_mode = align_mode

    @staticmethod
    def _align_scale_shift(pred, gt, mask):
        """Scale and shift alignment between prediction and ground truth"""
        p = pred[mask]
        g = gt[mask]
        if p.numel() == 0:
            return pred
        pm = p.mean()
        gm = g.mean()
        var_p = ((p - pm) * (p - pm)).mean()
        if var_p.item() == 0:
            return pred
        cov_pg = ((p - pm) * (g - gm)).mean()
        s = cov_pg / var_p
        b = gm - s * pm
        return pred * s + b

    @staticmethod
    def _align_scale(pred, gt, mask):
        """Scale-only alignment between prediction and ground truth"""
        eps = 1e-6
        p = pred[mask]
        g = gt[mask]
        if p.numel() == 0:
            return pred
        scale = torch.median(g) / (torch.median(p) + eps)
        return pred * scale

    def compute_depth_metrics(self, pred_depth, gt_depth, mask):
        """
        Compute depth estimation metrics.

        Args:
            pred_depth: Predicted depth [H, W]
            gt_depth: Ground truth depth [H, W]
            mask: Valid pixel mask [H, W]

        Returns:
            Dictionary of metrics
        """
        if mask.sum() == 0:
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'abs_rel': float('nan'),
                'sq_rel': float('nan'),
                'delta_1': float('nan'),
                'delta_2': float('nan'),
                'delta_3': float('nan'),
            }

        # Apply alignment
        if self.align_mode == 'scale_shift':
            pred_depth = self._align_scale_shift(pred_depth, gt_depth, mask)
        elif self.align_mode in ('scale', 'median'):
            pred_depth = self._align_scale(pred_depth, gt_depth, mask)

        pred = pred_depth[mask]
        gt = gt_depth[mask]

        # Compute metrics
        mae = torch.abs(pred - gt).mean().item()
        rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
        abs_rel = (torch.abs(pred - gt) / (gt + 1e-6)).mean().item()
        sq_rel = (((pred - gt) ** 2) / (gt + 1e-6)).mean().item()

        thresh = torch.maximum(pred / (gt + 1e-6), gt / (pred + 1e-6))
        delta_1 = (thresh < 1.25).float().mean().item()
        delta_2 = (thresh < 1.25 ** 2).float().mean().item()
        delta_3 = (thresh < 1.25 ** 3).float().mean().item()

        return {
            'mae': mae,
            'rmse': rmse,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'delta_1': delta_1,
            'delta_2': delta_2,
            'delta_3': delta_3,
        }

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict:
        """Evaluate a single checkpoint on the dataset"""
        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        # Load state dict
        model = self.model_base
        if isinstance(model, PeftModel):
            # Handle LoRA model
            keys = list(state.keys())
            sample_key = keys[0] if keys else ''
            has_prefix = sample_key.startswith('base_model.model.')

            if not has_prefix:
                state = {f'base_model.model.{k}': v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)

        model = model.to(self.device).eval()

        # Evaluate on dataset
        all_metrics = []
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                batch = self.dataset[idx]

                # Use first 8 frames
                S_use = min(8, batch['images'].shape[0])
                imgs = batch['images'][:S_use]
                gts = batch['depths'][:S_use]
                masks = batch['point_masks'][:S_use]

                # Move to device
                images = imgs.unsqueeze(0).to(self.device)
                gt_depths = gts.to(self.device)
                gt_masks = masks.to(self.device)

                # Forward pass
                outputs = model(images=images)
                pred_depths = outputs['depth'].squeeze(0).squeeze(-1)  # [S, H, W]

                # Compute metrics for each frame
                for s in range(pred_depths.shape[0]):
                    metrics = self.compute_depth_metrics(
                        pred_depths[s],
                        gt_depths[s],
                        gt_masks[s],
                    )
                    all_metrics.append(metrics)

        # Aggregate metrics
        valid_metrics = [m for m in all_metrics if not np.isnan(m['mae'])]

        if not valid_metrics:
            return {}

        aggregated = {}
        for key in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'delta_1', 'delta_2', 'delta_3']:
            values = [m[key] for m in valid_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)

        aggregated['num_valid_frames'] = len(valid_metrics)
        aggregated['num_total_frames'] = len(all_metrics)

        # Free memory
        model.cpu()
        del images, gt_depths, gt_masks, outputs, pred_depths
        torch.cuda.empty_cache()

        return aggregated


def plot_metrics_curve(
    epochs: List[int],
    metrics_by_epoch: Dict[int, Dict],
    output_dir: Path,
    exp_name: str,
):
    """Plot metric curves over training epochs"""

    # Prepare data for plotting
    metric_names = ['mae', 'rmse', 'abs_rel', 'sq_rel', 'delta_1', 'delta_2', 'delta_3']

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        mean_key = f'{metric_name}_mean'
        std_key = f'{metric_name}_std'

        means = [metrics_by_epoch[ep].get(mean_key, np.nan) for ep in epochs]
        stds = [metrics_by_epoch[ep].get(std_key, np.nan) for ep in epochs]

        # Filter out NaN values
        valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_epochs = [epochs[i] for i in valid_indices]
        valid_means = [means[i] for i in valid_indices]
        valid_stds = [stds[i] for i in valid_indices]

        if not valid_epochs:
            continue

        # Plot mean line
        ax.plot(valid_epochs, valid_means, 'o-', linewidth=2, markersize=4, label='mean')

        # Plot std as shaded area
        if all(not np.isnan(s) for s in valid_stds):
            lower = np.array(valid_means) - np.array(valid_stds)
            upper = np.array(valid_means) + np.array(valid_stds)
            ax.fill_between(valid_epochs, lower, upper, alpha=0.2)

        # Formatting
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name.upper().replace('_', ' '), fontsize=11)
        ax.set_title(f'{metric_name.upper()} over Training', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Highlight best value
        if metric_name in ['mae', 'rmse', 'abs_rel', 'sq_rel']:
            # Lower is better
            best_idx = np.argmin(valid_means)
            best_color = 'green'
        else:
            # Higher is better (delta metrics)
            best_idx = np.argmax(valid_means)
            best_color = 'blue'

        ax.axvline(valid_epochs[best_idx], color=best_color, linestyle='--', alpha=0.5, linewidth=1.5)
        ax.plot(valid_epochs[best_idx], valid_means[best_idx], 'r*', markersize=12)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.suptitle(f'Training Progress: {exp_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f'metrics_curve_{exp_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved metrics curve to {plot_path}")
    plt.close()

    # Create focused plots for key metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    key_metrics = [
        ('mae', 'MAE (Mean Absolute Error)', 'lower'),
        ('rmse', 'RMSE (Root Mean Square Error)', 'lower'),
        ('delta_1', 'δ < 1.25 (Accuracy)', 'higher'),
    ]

    for idx, (metric_name, title, better) in enumerate(key_metrics):
        ax = axes[idx]

        mean_key = f'{metric_name}_mean'
        std_key = f'{metric_name}_std'

        means = [metrics_by_epoch[ep].get(mean_key, np.nan) for ep in epochs]
        stds = [metrics_by_epoch[ep].get(std_key, np.nan) for ep in epochs]

        valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_epochs = [epochs[i] for i in valid_indices]
        valid_means = [means[i] for i in valid_indices]
        valid_stds = [stds[i] for i in valid_indices]

        if not valid_epochs:
            continue

        ax.plot(valid_epochs, valid_means, 'o-', linewidth=2.5, markersize=6, color='steelblue')

        if all(not np.isnan(s) for s in valid_stds):
            lower = np.array(valid_means) - np.array(valid_stds)
            upper = np.array(valid_means) + np.array(valid_stds)
            ax.fill_between(valid_epochs, lower, upper, alpha=0.25, color='steelblue')

        # Highlight best
        best_idx = np.argmin(valid_means) if better == 'lower' else np.argmax(valid_means)
        ax.axvline(valid_epochs[best_idx], color='green', linestyle='--', alpha=0.6, linewidth=2)
        ax.plot(valid_epochs[best_idx], valid_means[best_idx], 'r*', markersize=15)

        # Annotate best value
        best_val = valid_means[best_idx]
        best_ep = valid_epochs[best_idx]
        ax.annotate(f'Best: {best_val:.4f}\n(Epoch {best_ep})',
                   xy=(best_ep, best_val),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Key Metrics Progress: {exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / f'key_metrics_{exp_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved key metrics plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints and plot curves')
    parser.add_argument('--ckpt-dir', type=str, required=True,
                       help='Directory containing checkpoints (e.g., logs/lora_eth3d_strategy_b_r16/ckpts)')
    parser.add_argument('--config', type=str, default='lora_eth3d_strategy_a',
                       help='Config name for LoRA settings')
    parser.add_argument('--data-root', type=str,
                       default='/home/zerun/data/dataset/ETH3D/Stereo/High-res_multi-view',
                       help='Path to ETH3D dataset')
    parser.add_argument('--img-size', type=int, default=518, help='Image size')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as ckpt-dir parent)')
    parser.add_argument('--align', type=str, default='scale_shift',
                       choices=['none', 'scale', 'scale_shift', 'median'],
                       help='Alignment mode for depth metrics')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Setup paths
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory not found: {ckpt_dir}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_dir.parent / 'eval_curves'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract experiment name from path
    exp_name = ckpt_dir.parent.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Find all checkpoints
    checkpoint_files = sorted(ckpt_dir.glob('checkpoint_*.pt'))
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {ckpt_dir}")
        return

    logger.info(f"Found {len(checkpoint_files)} checkpoints")

    # Extract epoch numbers
    checkpoints_data = []
    for ckpt_path in checkpoint_files:
        match = re.search(r'checkpoint_(\d+)\.pt', ckpt_path.name)
        if match:
            epoch = int(match.group(1))
            checkpoints_data.append((epoch, ckpt_path))

    checkpoints_data.sort(key=lambda x: x[0])

    if not checkpoints_data:
        logger.error("No valid checkpoint files found")
        return

    logger.info(f"Evaluating epochs: {[ep for ep, _ in checkpoints_data]}")

    # Load LoRA config
    def _read_lora_config(config_name: str):
        cfg_path = Path('training/config') / f'{config_name}.yaml'
        if not cfg_path.exists():
            logger.warning(f"Config file not found: {cfg_path}")
            return None
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            return cfg.get('lora', None)
        except Exception as e:
            logger.warning(f"Failed to parse config {cfg_path}: {e}")
            return None

    lora_cfg = _read_lora_config(args.config)

    # Create base model
    logger.info("Creating base model...")
    model = VGGT(
        img_size=args.img_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
    )

    # Apply LoRA if needed
    if lora_cfg and lora_cfg.get('enabled', False):
        if not PEFT_AVAILABLE:
            logger.error("PEFT is required but not available. Please install 'peft'.")
            return
        logger.info("Applying LoRA adapters to model")
        model = apply_lora_to_model(model, lora_cfg, verbose=False)

    # Create dataset
    logger.info("Loading dataset...")
    dataset = ETH3DDataset(
        root_dir=args.data_root,
        split='val',
        img_size=args.img_size,
        sequence_length=8,
        use_building_scenes_only=True,
        train_val_split=0.85,
    )

    logger.info(f"Dataset: {len(dataset)} sequences")

    # Create evaluator
    evaluator = CheckpointEvaluator(
        model_base=model,
        dataset=dataset,
        device=args.device,
        align_mode=args.align,
    )

    # Evaluate all checkpoints
    metrics_by_epoch = {}
    epochs = []

    for epoch, ckpt_path in tqdm(checkpoints_data, desc="Evaluating checkpoints"):
        try:
            metrics = evaluator.evaluate_checkpoint(str(ckpt_path))
            if metrics:
                metrics_by_epoch[epoch] = metrics
                epochs.append(epoch)
                logger.info(f"Epoch {epoch}: MAE={metrics.get('mae_mean', 'N/A'):.4f}, "
                          f"RMSE={metrics.get('rmse_mean', 'N/A'):.4f}, "
                          f"δ<1.25={metrics.get('delta_1_mean', 'N/A'):.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate checkpoint at epoch {epoch}: {e}")
            continue

    if not epochs:
        logger.error("No checkpoints were successfully evaluated")
        return

    # Save metrics to file
    metrics_file = output_dir / f'metrics_all_epochs_{exp_name}_{timestamp}.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Alignment: {args.align}\n")
        f.write(f"Total checkpoints evaluated: {len(epochs)}\n")
        f.write("=" * 80 + "\n\n")

        for epoch in epochs:
            metrics = metrics_by_epoch[epoch]
            f.write(f"Epoch {epoch}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    logger.info(f"Saved detailed metrics to {metrics_file}")

    # Plot curves
    logger.info("Generating plots...")
    plot_metrics_curve(epochs, metrics_by_epoch, output_dir, exp_name)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)

    # Find best checkpoints
    mae_values = [(ep, metrics_by_epoch[ep]['mae_mean']) for ep in epochs]
    best_mae_ep, best_mae = min(mae_values, key=lambda x: x[1])

    delta1_values = [(ep, metrics_by_epoch[ep]['delta_1_mean']) for ep in epochs]
    best_delta1_ep, best_delta1 = max(delta1_values, key=lambda x: x[1])

    logger.info(f"Best MAE: {best_mae:.4f} at epoch {best_mae_ep}")
    logger.info(f"Best δ<1.25: {best_delta1:.4f} at epoch {best_delta1_ep}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
