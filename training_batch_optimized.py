import argparse
import os
import sys
import logging

from hydra import initialize, compose
from omegaconf import OmegaConf

# Ensure 'training/' is on sys.path so absolute imports like 'train_utils.*' work
_ROOT = os.path.dirname(__file__)
_TRAIN_DIR = os.path.join(_ROOT, "training")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from training.trainer_batch_optimized import TrainerBatchOptimized


def main():
    parser = argparse.ArgumentParser(description="Batch-frames sweep training with periodic val and extended logging")
    parser.add_argument("--config", type=str, default="default", help="Hydra config name under training/config (without .yaml)")
    parser.add_argument("--batch_frames", type=int, required=True, help="Total frames per GPU (max_img_per_gpu)")
    parser.add_argument("--steps", type=int, required=True, help="Number of training steps to run (single epoch)")
    parser.add_argument("--val_every", type=int, default=50, help="Run validation every N training steps (0 to disable)")
    parser.add_argument("--val_batches", type=int, default=50, help="Max validation batches to run per val trigger")
    parser.add_argument("--warmup_skip", type=int, default=5, help="Skip first K steps in summary aggregation")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name to isolate logs and checkpoints")
    args = parser.parse_args()

    # Load base config
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name=args.config)

    # Basic overrides (unify accum, single epoch, steps limit)
    cfg.accum_steps = 1
    cfg.max_epochs = 1
    cfg.limit_train_batches = max(0, args.steps - 1)
    cfg.limit_val_batches = int(args.val_batches)

    # Inject step-based val and warmup handling into trainer
    cfg.val_step_freq = int(args.val_every)
    cfg.warmup_skip_steps = int(args.warmup_skip)
    cfg.jsonl_dir = "report/baselines"

    # Batch-frames override (train/val)
    bf = int(args.batch_frames)
    # train
    cfg.data.train.max_img_per_gpu = bf
    if "common_config" in cfg.data.train and cfg.data.train.common_config is not None:
        cfg.data.train.common_config.max_img_per_gpu = bf
    # val
    if "val" in cfg.data and cfg.data.val is not None:
        cfg.data.val.max_img_per_gpu = bf
        if "common_config" in cfg.data.val and cfg.data.val.common_config is not None:
            cfg.data.val.common_config.max_img_per_gpu = bf

    # Logging/exp isolation
    base_exp = args.exp_name or cfg.exp_name
    cfg.exp_name = f"{base_exp}_bf{bf}_s{args.steps}"
    cfg.logging.log_dir = os.path.join("logs", cfg.exp_name)
    # Disable resuming unless explicitly set
    if "checkpoint" in cfg and cfg.checkpoint is not None:
        cfg.checkpoint.resume_checkpoint_path = None
        # Keep checkpoint dir under the same exp folder
        cfg.checkpoint.save_dir = os.path.join("logs", cfg.exp_name, "ckpts")

    # Optional: make epoch-driven val effectively disabled during training; final val still runs after .run()
    cfg.val_epoch_freq = 999999

    # Show final config (rank-0 will print it inside trainer as well)
    print("===== Effective Config =====")
    print(OmegaConf.to_yaml(cfg))

    # Run training
    trainer = TrainerBatchOptimized(**cfg)
    trainer.run()
    trainer.write_summaries()


if __name__ == "__main__":
    main()
