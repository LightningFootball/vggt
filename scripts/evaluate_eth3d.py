#!/usr/bin/env python3
"""
Evaluation script for ETH3D dataset with LoRA-trained VGGT model.

This script evaluates depth estimation quality on ETH3D test set,
with special focus on building facades and window regions.

Usage:
    python scripts/evaluate_eth3d.py --checkpoint path/to/checkpoint.pt --config lora_eth3d_strategy_a
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from training.data.datasets.eth3d import ETH3DDataset
from training.lora_utils import load_lora_checkpoint, apply_lora_to_model

try:
    from peft import PeftModel  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ETH3DEvaluator:
    """Evaluator for ETH3D depth estimation"""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: ETH3DDataset,
        device: str = 'cuda',
        save_visualizations: bool = False,
        output_dir: str = './eval_results',
        align_mode: str = 'none',  # 'none' | 'scale' | 'scale_shift' | 'median'
        test_seq_lens: list[int] | None = None,
        vis_subdir_name: str = None,  # Subdirectory name for visualizations
    ):
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.device = device
        self.save_visualizations = save_visualizations
        self.output_dir = Path(output_dir)
        self.align_mode = align_mode
        self.test_seq_lens = test_seq_lens or [8]
        self.vis_subdir_name = vis_subdir_name

        if save_visualizations:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            vis_base = self.output_dir / 'visualizations'
            vis_base.mkdir(exist_ok=True)
            if self.vis_subdir_name:
                self.vis_dir = vis_base / self.vis_subdir_name
                self.vis_dir.mkdir(exist_ok=True)
            else:
                self.vis_dir = vis_base

    @staticmethod
    def _align_scale(pred, gt, mask):
        eps = 1e-6
        p = pred[mask]
        g = gt[mask]
        if p.numel() == 0:
            return pred
        scale = torch.median(g) / (torch.median(p) + eps)
        return pred * scale

    @staticmethod
    def _align_scale_shift(pred, gt, mask):
        # Solve min_{s,b} || s * p + b - g ||^2 over valid mask
        p = pred[mask]
        g = gt[mask]
        if p.numel() == 0:
            return pred
        # s = cov(p,g)/var(p); b = mean(g) - s*mean(p)
        pm = p.mean()
        gm = g.mean()
        var_p = ((p - pm) * (p - pm)).mean()
        if var_p.item() == 0:
            return pred
        cov_pg = ((p - pm) * (g - gm)).mean()
        s = cov_pg / var_p
        b = gm - s * pm
        return pred * s + b

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

        # Optional alignment before computing metrics
        if self.align_mode in ('scale', 'median'):
            pred_depth = self._align_scale(pred_depth, gt_depth, mask)
        elif self.align_mode == 'scale_shift':
            pred_depth = self._align_scale_shift(pred_depth, gt_depth, mask)

        pred = pred_depth[mask]
        gt = gt_depth[mask]

        # Mean Absolute Error
        mae = torch.abs(pred - gt).mean().item()

        # Root Mean Square Error
        rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()

        # Absolute Relative Error
        abs_rel = (torch.abs(pred - gt) / (gt + 1e-6)).mean().item()

        # Square Relative Error
        sq_rel = (((pred - gt) ** 2) / (gt + 1e-6)).mean().item()

        # Threshold Accuracy
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

    def evaluate(self):
        """Run evaluation on the dataset for each requested sequence length"""
        results_by_len: dict[int, dict] = {}

        logger.info(f"Evaluating on {len(self.dataset)} sequences...")

        with torch.no_grad():
            for seq_len in self.test_seq_lens:
                all_metrics = []
                pbar = tqdm(range(len(self.dataset)), desc=f"Evaluating (len={seq_len})")
                for idx in pbar:
                    batch = self.dataset[idx]

                    # Slice to requested sequence length
                    S_avail = batch['images'].shape[0]
                    S_use = min(seq_len, S_avail)
                    imgs = batch['images'][:S_use]
                    gts = batch['depths'][:S_use]
                    masks = batch['point_masks'][:S_use]

                    # Move to device
                    images = imgs.unsqueeze(0).to(self.device)
                    gt_depths = gts.to(self.device)
                    gt_masks = masks.to(self.device)

                    # Forward pass
                    outputs = self.model(images=images)
                    pred_depths = outputs['depth'].squeeze(0)  # [S, H, W, 1]
                    pred_depths = pred_depths.squeeze(-1)  # [S, H, W]

                    # Compute metrics for each frame
                    S = pred_depths.shape[0]
                    for s in range(S):
                        metrics = self.compute_depth_metrics(
                            pred_depths[s],
                            gt_depths[s],
                            gt_masks[s],
                        )
                        metrics['scene'] = batch['seq_name']
                        metrics['frame'] = s
                        metrics['seq_len'] = S
                        all_metrics.append(metrics)

                    # Save visualization for first sequence per seq_len
                    if self.save_visualizations and idx == 0:
                        vis_dir = self.vis_dir / f'len_{seq_len}'
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        self._save_visualization(
                            imgs[0],
                            pred_depths[0],
                            gt_depths[0],
                            gt_masks[0],
                            f"len{seq_len}_{batch['seq_name']}_frame0",
                            vis_dir
                        )

                # Aggregate for this seq_len
                results_by_len[seq_len] = self._aggregate_metrics(all_metrics)

        return results_by_len

    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across all frames"""
        # Filter out NaN values
        valid_metrics = [m for m in all_metrics if not np.isnan(m['mae'])]

        if not valid_metrics:
            logger.warning("No valid metrics computed!")
            return {}

        # Compute mean for each metric
        aggregated = {}
        for key in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'delta_1', 'delta_2', 'delta_3']:
            values = [m[key] for m in valid_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_median'] = np.median(values)

        aggregated['num_valid_frames'] = len(valid_metrics)
        aggregated['num_total_frames'] = len(all_metrics)

        return aggregated

    def _save_visualization(self, image, pred_depth, gt_depth, mask, name, vis_dir):
        """Save depth visualization"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # Predicted depth
        pred_np = pred_depth.cpu().numpy()
        im1 = axes[0, 1].imshow(pred_np, cmap='plasma')
        axes[0, 1].set_title('Predicted Depth')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # Ground truth depth
        gt_np = gt_depth.cpu().numpy()
        im2 = axes[1, 0].imshow(gt_np, cmap='plasma')
        axes[1, 0].set_title('Ground Truth Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])

        # Error map
        error = np.abs(pred_np - gt_np)
        error[~mask.cpu().numpy()] = 0
        im3 = axes[1, 1].imshow(error, cmap='hot')
        axes[1, 1].set_title('Absolute Error')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(vis_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGT on ETH3D dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--lora-checkpoint', type=str, default=None, help='Path to LoRA checkpoint (if separate)')
    parser.add_argument('--config', type=str, default='lora_eth3d_strategy_a', help='Config name')
    parser.add_argument('--data-root', type=str,
                        default='/home/zerun/data/dataset/ETH3D/Stereo/High-res_multi-view',
                        help='Path to ETH3D dataset')
    parser.add_argument('--img-size', type=int, default=518, help='Image size')
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--align', type=str, default='none', choices=['none', 'scale', 'scale_shift', 'median'],
                        help='Alignment between prediction and GT before metric computation')
    parser.add_argument('--test-seq-lens', type=str, default='8',
                        help='Comma-separated list of sequence lengths to test, e.g., "1,2,4,8"')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Extract config letter (a, b, c, etc.) from config name
    config_match = re.search(r'strategy_([a-z])', args.config.lower())
    config_letter = config_match.group(1) if config_match else args.config

    # Extract epoch number from checkpoint filename
    checkpoint_match = re.search(r'checkpoint_(\d+)', args.checkpoint)
    epoch_num = checkpoint_match.group(1) if checkpoint_match else '0'

    # Format alignment string (replace 'none' with 'no_align')
    align_str = 'no_align' if args.align == 'none' else args.align

    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d%H%M')

    # Update output directory if not explicitly set
    if args.output_dir == './eval_results':
        args.output_dir = f'./eval_results_{config_letter}_{epoch_num}_{align_str}_{timestamp}'

    # Load model
    logger.info("Loading model...")
    model = VGGT(
        img_size=args.img_size,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
    )

    # --- Robust LoRA + checkpoint loading ---
    def _read_training_lora_config(config_name: str):
        """Read LoRA config dict from training/config/<config_name>.yaml if present."""
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

    def _checkpoint_has_lora_keys(state_dict_keys) -> bool:
        for k in state_dict_keys:
            # Typical LoRA parameter names in PEFT: lora_A, lora_B, lora_embedding_A/B
            if ('lora_A' in k) or ('lora_B' in k) or ('.lora_' in k):
                return True
        return False

    # 1) Inspect checkpoint to decide loading path
    assert args.checkpoint is not None, "--checkpoint is required"
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    keys = list(state.keys())
    sample_key = keys[0] if keys else ''
    has_prefix = sample_key.startswith('base_model.model.')
    has_lora = _checkpoint_has_lora_keys(keys)

    # 2) If a separate LoRA checkpoint is provided, load that after base weights
    #    Otherwise, apply LoRA based on training config (if needed) to match checkpoint layout
    #    so that state_dict keys align.
    lora_cfg = _read_training_lora_config(args.config)
    want_lora = (lora_cfg is not None and lora_cfg.get('enabled', False)) or has_lora or (args.lora_checkpoint is not None)

    if want_lora:
        if not PEFT_AVAILABLE:
            logger.error("PEFT is required for loading LoRA weights but is not available.\n"
                         "Please install 'peft' in your environment.")
        else:
            # Ensure we have a usable lora config. If missing, fall back to defaults
            if lora_cfg is None:
                # Fallback to Strategy A defaults
                lora_cfg = {
                    'enabled': True,
                    'rank': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'target_modules': [
                        'depth_head.projects.*',
                        'depth_head.scratch.*',
                        'depth_head.scratch.refinenet*.resConfUnit*.conv*',
                    ],
                }
            # Apply LoRA adapters to model to create a PeftModel wrapper
            logger.info("Applying LoRA adapters to model to match checkpoint layout")
            model = apply_lora_to_model(model, lora_cfg, verbose=True)

    # 3) Now load the main checkpoint with prefix adaptation
    if want_lora and PEFT_AVAILABLE and not isinstance(model, PeftModel):
        # Safety: If we wanted LoRA but did not get a PEFT model (e.g. PEFT missing), we still try best-effort
        logger.warning("LoRA requested but model is not a PeftModel. Proceeding best-effort without LoRA.")

    if isinstance(model, PeftModel):
        # LoRA-wrapped model expects keys prefixed with 'base_model.model.'
        if not has_prefix:
            logger.info("Adding 'base_model.model.' prefix to checkpoint keys for LoRA model")
            state = {f'base_model.model.{k}': v for k, v in state.items()}
    else:
        # Non-LoRA model expects raw backbone keys; if checkpoint has the prefix, strip it
        if has_prefix:
            logger.info("Stripping 'base_model.model.' prefix from checkpoint keys for non-LoRA model")
            new_state = {}
            for k, v in state.items():
                if k.startswith('base_model.model.'):
                    new_state[k[len('base_model.model.'):]] = v
                else:
                    new_state[k] = v
            state = new_state

    # Perform the actual load
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        logger.debug(f"Missing (first 20): {missing[:20]}")
    if len(unexpected) > 0:
        logger.debug(f"Unexpected (first 20): {unexpected[:20]}")

    # 4) If a separate LoRA checkpoint was specified, load it now (overrides)
    if args.lora_checkpoint and PEFT_AVAILABLE and isinstance(model, PeftModel):
        logger.info(f"Loading LoRA checkpoint from {args.lora_checkpoint}")
        model = load_lora_checkpoint(model.base_model.model, args.lora_checkpoint)  # ensure correct base passed

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
    # Parse test sequence lengths
    try:
        test_seq_lens = [int(x) for x in args.test_seq_lens.split(',') if x.strip()]
    except Exception:
        test_seq_lens = [8]

    # Generate visualization subdirectory name based on metrics filename (without .txt)
    vis_subdir_name = f'{config_letter}_{epoch_num}_{align_str}_{timestamp}'

    evaluator = ETH3DEvaluator(
        model=model,
        dataset=dataset,
        device=args.device,
        save_visualizations=args.save_vis,
        output_dir=args.output_dir,
        align_mode=args.align,
        test_seq_lens=test_seq_lens,
        vis_subdir_name=vis_subdir_name,
    )

    # Run evaluation
    results_by_len = evaluator.evaluate()

    # Print and save consolidated results
    # Generate descriptive filename: metrics_{config}_{epoch}_{align}_{timestamp}.txt
    metrics_filename = f'metrics_{config_letter}_{epoch_num}_{align_str}_{timestamp}.txt'
    output_file = Path(args.output_dir) / metrics_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Alignment: {args.align}")

    with open(output_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"alignment: {args.align}\n")

        for seq_len in sorted(results_by_len.keys()):
            metrics = results_by_len[seq_len]
            logger.info(f"\nSequence length: {seq_len}")
            logger.info(f"  Valid frames: {metrics['num_valid_frames']} / {metrics['num_total_frames']}")
            logger.info("  Depth Metrics:")
            logger.info(f"    MAE:       {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f} (median: {metrics['mae_median']:.4f})")
            logger.info(f"    RMSE:      {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f} (median: {metrics['rmse_median']:.4f})")
            logger.info(f"    Abs Rel:   {metrics['abs_rel_mean']:.4f} ± {metrics['abs_rel_std']:.4f}")
            logger.info(f"    Sq Rel:    {metrics['sq_rel_mean']:.4f} ± {metrics['sq_rel_std']:.4f}")
            logger.info("  Threshold Accuracy:")
            logger.info(f"    δ < 1.25:  {metrics['delta_1_mean']:.4f}")
            logger.info(f"    δ < 1.25²: {metrics['delta_2_mean']:.4f}")
            logger.info(f"    δ < 1.25³: {metrics['delta_3_mean']:.4f}")

            f.write(f"\nseq_len: {seq_len}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    logger.info("=" * 80)
    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
