#!/usr/bin/env python3
"""
Utility script to reproduce VGGT forward passes on specific KITTI-360 frame indices.

Allows toggling AMP and LoRA to diagnose numerical instabilities on problematic batches.
"""

import argparse
import logging
import os
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure local imports resolve regardless of invocation path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINING_ROOT = os.path.join(REPO_ROOT, "training")
import sys

for path in (REPO_ROOT, TRAINING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.data.datasets.kitti360 import KITTI360Dataset  # noqa: E402
from training.data.dataset_util import read_image_cv2  # noqa: E402
from training.lora_utils import apply_lora_to_model  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def _prepare_common_conf(cfg) -> OmegaConf:
    """Convert common_config DictConfig to an object with attribute access."""
    container = OmegaConf.to_container(cfg.data.train.common_config, resolve=True)
    return OmegaConf.create(container)


def _prepare_dataset(cfg, common_conf) -> KITTI360Dataset:
    dataset_cfg = cfg.data.train.dataset.dataset_configs[0]
    dataset_kwargs = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_kwargs.pop("_target_", None)
    return KITTI360Dataset(common_conf=common_conf, **dataset_kwargs)


def _process_frames(dataset: KITTI360Dataset, frame_indices: List[int]) -> np.ndarray:
    """Process specified dataset frame indices into normalized image tensors."""
    aspect_ratio = 1.0
    target_shape = dataset.get_target_shape(aspect_ratio)
    processed_images = []

    for idx in frame_indices:
        frame = dataset.frames[idx]
        image = read_image_cv2(frame["image_path"])
        if image is None:
            raise RuntimeError(f"Failed to load image for frame index {idx}")
        original_size = np.array(image.shape[:2])

        depth_map, _ = dataset._project_lidar_to_depth(frame, idx, original_size)
        pose_cam_to_world = frame["pose"]
        extri_opencv = np.linalg.inv(pose_cam_to_world).astype(np.float32)
        intri_opencv = dataset.calib["K"].astype(np.float32)

        (
            image_processed,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = dataset.process_one_image(
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            original_size,
            target_shape,
            filepath=frame["image_path"],
        )

        processed_images.append(image_processed.astype(np.float32))

    stacked = np.stack(processed_images, axis=0)  # [S, H, W, 3]
    stacked = torch.from_numpy(stacked).permute(0, 3, 1, 2) / 255.0  # [S, 3, H, W]
    return stacked.unsqueeze(0).contiguous()  # [1, S, 3, H, W]


def _maybe_apply_lora(model: torch.nn.Module, cfg, disable_lora: bool) -> torch.nn.Module:
    lora_cfg = OmegaConf.to_container(cfg.lora, resolve=True) if cfg.get("lora") else None
    if disable_lora or not lora_cfg or not lora_cfg.get("enabled", False):
        logging.info("LoRA disabled for this run.")
        return model
    logging.info("Applying LoRA with config: %s", {k: v for k, v in lora_cfg.items() if k != "target_modules"})
    model = apply_lora_to_model(model, lora_config=lora_cfg, verbose=True)
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = False) -> None:
    logging.info("Loading checkpoint %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not state:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'model' key.")
    sample_key = next(iter(state.keys()))
    if not sample_key.startswith("base_model.model."):
        logging.info("Adding base_model.model prefix to checkpoint keys")
        state = {f"base_model.model.{k}": v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    logging.info("Loaded checkpoint. Missing keys=%d, unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.debug("Missing keys sample: %s", missing[:10])
    if unexpected:
        logging.debug("Unexpected keys sample: %s", unexpected[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug VGGT on specific KITTI-360 frames.")
    parser.add_argument(
        "--config",
        default="training/config/lora_kitti360_strategy_b.yaml",
        help="Path to Hydra config used for LoRA fine-tuning.",
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/lora_kitti360_strategy_b_r16/ckpts/checkpoint_21.pt",
        help="Checkpoint path for model weights.",
    )
    parser.add_argument(
        "--frame-indices",
        nargs="+",
        type=int,
        default=[3751, 3762, 3796, 3781, 3740, 3798, 3801, 3790],
        help="Frame indices corresponding to suspected bad batch.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the forward pass on.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable autocast for the forward pass.",
    )
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Run with base model only (no LoRA adapters).",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    common_conf = _prepare_common_conf(cfg)
    dataset = _prepare_dataset(cfg, common_conf)
    logging.info("Dataset prepared with %d frames", len(dataset.frames))
    logging.info("Processing frame indices: %s", args.frame_indices)

    images = _process_frames(dataset, args.frame_indices)
    device = torch.device(args.device)
    images = images.to(device)
    logging.info("Images tensor shape: %s", tuple(images.shape))

    model = VGGT(
        img_size=cfg.get("img_size", 518),
        patch_size=cfg.get("patch_size", 14),
        enable_camera=cfg.model.get("enable_camera", True),
        enable_depth=cfg.model.get("enable_depth", True),
        enable_point=cfg.model.get("enable_point", False),
        enable_track=cfg.model.get("enable_track", False),
    )
    model = _maybe_apply_lora(model, cfg, disable_lora=args.disable_lora)

    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, strict=False)

    model.to(device)
    model.eval()

    amp_dtype = torch.bfloat16 if cfg.optim.amp.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    autocast_enabled = not args.disable_amp and cfg.optim.amp.get("enabled", True)

    logging.info(
        "Running forward pass | AMP=%s | dtype=%s | LoRA=%s",
        autocast_enabled,
        amp_dtype,
        "disabled" if args.disable_lora else "enabled",
    )

    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=autocast_enabled and device.type == "cuda", dtype=amp_dtype):
                outputs = model(images=images)
        logging.info("Forward pass completed successfully.")
        if outputs:
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    finite = torch.isfinite(value)
                    logging.info(
                        "Output %s: shape=%s | finite=%s | max_abs=%s",
                        key,
                        tuple(value.shape),
                        bool(finite.all()),
                        float(value.abs().max().item()) if finite.any() else "nan",
                    )
    except RuntimeError as err:
        logging.error("Forward pass failed: %s", err, exc_info=True)


if __name__ == "__main__":
    main()
