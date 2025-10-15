# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from statistics import median


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.hooks
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers

try:
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
except ImportError:
    PeftModel = None  # type: ignore[assignment]
    LoraLayer = ()  # type: ignore[assignment]


class LoRANumericalError(RuntimeError):
    """Raised when the LoRA guard detects non-finite activations."""
    pass


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        lora: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            lora: Hydra config for LoRA (optional).
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.lora_conf = lora
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        # Guard rails for data anomalies and logging spam control
        self.enable_batch_sanity_checks = bool(int(os.getenv("VGGT_ENABLE_BATCH_SANITY_CHECKS", "1")))
        self.bad_batch_log_limit = int(os.getenv("VGGT_BAD_BATCH_LOG_LIMIT", "20"))
        self._bad_batch_reports = 0
        self.bad_batch_retry_limit = int(os.getenv("VGGT_BAD_BATCH_RETRY_LIMIT", "1"))
        self.bad_batch_blacklist: Set[str] = set()
        self._bad_batch_failures: Dict[str, int] = defaultdict(int)

        self.profile_segments = bool(int(os.getenv("VGGT_PROFILE_TIMING", "0")))
        if self.profile_segments and not torch.cuda.is_available():
            logging.warning(
                "VGGT_PROFILE_TIMING is enabled but CUDA is unavailable; disabling segment profiling."
            )
            self.profile_segments = False
        self._profiling_records = defaultdict(list) if self.profile_segments else None
        self._iteration_profile_events = []
        self._current_iteration_profile = None
        self._nan_guard_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._nan_guard_triggered: bool = False
        self._nan_guard_batch_meta: str = "metadata=unavailable"
        self.lora_guard_max_abs = float(os.getenv("VGGT_LORA_GUARD_MAX_ABS", "25000"))
        self.enable_cuda_sync_guard = bool(int(os.getenv("VGGT_ENABLE_CUDA_SYNC_GUARD", "1")))

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        # First, try to resume from a previously saved training checkpoint
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is not None:
            self._load_resuming_checkpoint(ckpt_path)
        elif self.checkpoint_conf.resume_checkpoint_path is not None:
            # If no training checkpoint exists, load from pretrained model
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _reset_iteration_profile_buffers(self) -> None:
        if not self.profile_segments:
            return
        if self._profiling_records is None:
            self._profiling_records = defaultdict(list)
        self._iteration_profile_events = []
        self._current_iteration_profile = defaultdict(float)

    def _profile_event_start(self, segment: str):
        if not self.profile_segments or self._current_iteration_profile is None:
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return segment, start_event

    def _profile_event_end(self, token) -> None:
        if not self.profile_segments or token is None:
            return
        segment, start_event = token
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._iteration_profile_events.append((segment, start_event, end_event))

    def _profile_finalize_iteration(self) -> None:
        if not self.profile_segments or self._current_iteration_profile is None:
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            logging.warning("Failed to synchronize CUDA while finalizing profiling iteration.")
            return

        for segment, start_event, end_event in self._iteration_profile_events:
            try:
                elapsed = start_event.elapsed_time(end_event)
            except RuntimeError:
                logging.warning(f"Failed to compute elapsed time for segment {segment}.")
                continue
            self._current_iteration_profile[segment] += elapsed

        for segment, total in self._current_iteration_profile.items():
            self._profiling_records[segment].append(total)
        self._current_iteration_profile = None
        self._iteration_profile_events = []

    def _profile_tensor_item(self, tensor: torch.Tensor, label: str) -> float:
        if not self.profile_segments:
            return tensor.item()
        token = self._profile_event_start(f"item:{label}")
        value = tensor.item()
        self._profile_event_end(token)
        return value

    def _profile_tensor_cpu(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        if not self.profile_segments:
            return tensor.cpu()
        token = self._profile_event_start(f"cpu:{label}")
        result = tensor.cpu()
        self._profile_event_end(token)
        return result

    def _profile_tensor_numpy(self, tensor: torch.Tensor, label: str):
        if not self.profile_segments:
            return tensor.numpy()
        token = self._profile_event_start(f"numpy:{label}")
        result = tensor.numpy()
        self._profile_event_end(token)
        return result

    def _format_batch_metadata(self, batch: Mapping) -> str:
        """Builds a compact description of key batch identifiers for logging."""
        parts: List[str] = []
        seq_name = batch.get("seq_name")
        if isinstance(seq_name, str):
            parts.append(f"seq={seq_name}")
        elif isinstance(seq_name, Sequence):
            unique_seq = list(dict.fromkeys(seq_name))
            if unique_seq:
                preview = unique_seq[:2]
                if len(unique_seq) > 2:
                    preview.append("...")
                parts.append(f"seq={preview}")
        ids = batch.get("ids")
        if isinstance(ids, torch.Tensor):
            try:
                ids_list = ids.detach().cpu().view(-1).tolist()
            except RuntimeError:
                ids_list = None
        elif isinstance(ids, (list, tuple)):
            ids_list = list(ids)
        else:
            ids_list = None
        if ids_list:
            preview = ids_list[:8]
            if len(ids_list) > 8:
                preview.append("...")
            parts.append(f"ids={preview}")
        frame_num = batch.get("frame_num")
        if isinstance(frame_num, int):
            parts.append(f"frames={frame_num}")
        return ", ".join(parts) if parts else "metadata=unavailable"

    def _log_bad_batch(self, message: str, level: str = "warning") -> None:
        """Logs bad batch information with optional throttling."""
        logger_fn = getattr(logging, level, logging.warning)
        if self.bad_batch_log_limit < 0 or self._bad_batch_reports < self.bad_batch_log_limit:
            logger_fn(message)
        elif self._bad_batch_reports == self.bad_batch_log_limit:
            logger_fn("Bad batch log limit reached; suppressing further messages.")
        self._bad_batch_reports += 1

    def _record_bad_batch_skip(
        self,
        *,
        phase: str,
        reason: str,
        batch: Mapping,
        log_level: str = "warning",
        exception: Optional[Exception] = None,
    ) -> None:
        """Records that a batch is skipped along with contextual metadata."""
        iter_idx = self.steps.get(phase, 0) if isinstance(self.steps, Mapping) else 0
        meta = self._format_batch_metadata(batch)
        message = f"Skipping {phase} batch (iter={iter_idx}): {reason}. Context: {meta}"
        if exception is not None:
            message = f"{message} | Exception: {exception}"
        self._log_bad_batch(message, level=log_level)

    def _note_bad_batch(self, batch: Mapping, reason: str, meta: Optional[str] = None) -> None:
        """Track failures per batch and blacklist after exceeding retry limit."""
        if self.bad_batch_retry_limit == 0:
            return

        batch_key = meta if meta is not None else self._format_batch_metadata(batch)
        if not batch_key:
            return

        self._bad_batch_failures[batch_key] += 1
        failure_count = self._bad_batch_failures[batch_key]

        logging.debug(
            "Bad batch note | key=%s | count=%d | reason=%s",
            batch_key,
            failure_count,
            reason,
        )

        if self.bad_batch_retry_limit > 0 and failure_count >= self.bad_batch_retry_limit:
            if batch_key not in self.bad_batch_blacklist:
                self.bad_batch_blacklist.add(batch_key)
                logging.error(
                    "Blacklisting batch after %d failures (%s). Future occurrences will be skipped.",
                    failure_count,
                    batch_key,
                )

    def _validate_batch_tensors(
        self,
        batch: Mapping,
    ) -> Tuple[bool, List[Tuple[str, int, List[float]]]]:
        """Checks for non-finite values inside a batch."""
        invalid_entries: List[Tuple[str, int, List[float]]] = []
        if not self.enable_batch_sanity_checks:
            return True, invalid_entries

        with torch.no_grad():
            for key, value in batch.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    tensor = value.detach()
                    nonfinite_mask = ~torch.isfinite(tensor)
                    if nonfinite_mask.any():
                        count = int(nonfinite_mask.sum().item())
                        try:
                            samples = (
                                tensor[nonfinite_mask]
                                .reshape(-1)[:3]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                        except RuntimeError:
                            samples = []
                        invalid_entries.append((key, count, samples))

        return len(invalid_entries) == 0, invalid_entries

    def _handle_step_failure(
        self,
        *,
        phase: str,
        batch: Mapping,
        err: Exception,
        location: str,
    ) -> None:
        """Handles failures during forward or backward passes."""
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

        self._record_bad_batch_skip(
            phase=phase,
            reason=f"{location} failure",
            batch=batch,
            log_level="error",
            exception=err,
        )
        self._note_bad_batch(batch, f"{location} failure", meta=self._nan_guard_batch_meta)

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

    def _log_iteration_profile_stats(self) -> None:
        if not self.profile_segments or not self._profiling_records:
            return
        summary = []
        for segment, values in self._profiling_records.items():
            if not values:
                continue
            segment_median = median(values) / 1000.0
            summary.append(f"{segment}: median={segment_median:.6f}s (n={len(values)})")
        if summary:
            logging.info("[Timing] CUDA segment medians per iteration -> " + "; ".join(sorted(summary)))
        self._profiling_records = defaultdict(list)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # Handle LoRA model prefix mismatch
        # LoRA wraps models with 'base_model.model.' prefix
        # If loading pretrained weights into LoRA model, add prefix
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            # Check if checkpoint keys have the prefix
            sample_key = next(iter(model_state_dict.keys()))
            if not sample_key.startswith('base_model.model.'):
                logging.info("Adding 'base_model.model.' prefix to checkpoint keys for LoRA model")
                model_state_dict = {
                    f'base_model.model.{k}': v for k, v in model_state_dict.items()
                }

        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {len(missing)} keys. Unexpected keys: {len(unexpected)} keys.")
            if len(missing) > 0 and len(missing) <= 10:
                logging.info(f"Missing keys sample: {missing[:10]}")
            if len(unexpected) > 0 and len(unexpected) <= 10:
                logging.info(f"Unexpected keys sample: {unexpected[:10]}")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint and self.mode == "train":
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            opt_state = checkpoint["optimizer"]
            try:
                if isinstance(opt_state, list):
                    if len(opt_state) != len(self.optims):
                        logging.warning(
                            f"Checkpoint has {len(opt_state)} optimizers, but trainer has {len(self.optims)}. Skipping optimizer load."
                        )
                    else:
                        for opt, state in zip(self.optims, opt_state):
                            opt.optimizer.load_state_dict(state)
                else:
                    if len(self.optims) == 1:
                        self.optims[0].optimizer.load_state_dict(opt_state)
                    else:
                        logging.warning(
                            "Checkpoint stores a single optimizer state but trainer has multiple. Skipping optimizer load."
                        )
            except Exception as e:
                logging.error(f"Failed to load optimizer state: {e}")

        # Load training progress
        epoch_in_ckpt = checkpoint.get("epoch", checkpoint.get("prev_epoch", None))
        if epoch_in_ckpt is not None:
            # Checkpoint contains the completed epoch number
            # Resume training from the next epoch
            self.epoch = epoch_in_ckpt + 1
            logging.info(f"Resuming from epoch {self.epoch} (completed epoch {epoch_in_ckpt})")
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Apply LoRA if enabled
        lora_config = getattr(self, 'lora_conf', None)
        if lora_config is not None and lora_config.get('enabled', False):
            logging.info("[Start] Applying LoRA to model")
            try:
                from lora_utils import apply_lora_to_model
                self.model = apply_lora_to_model(
                    self.model,
                    lora_config=lora_config,
                    verbose=(self.rank == 0)
                )
                logging.info("[Done] LoRA applied successfully")
            except ImportError as e:
                logging.error(f"Failed to import LoRA utilities: {e}")
                logging.error("Please install PEFT library: pip install peft")
                raise
            except Exception as e:
                logging.error(f"Failed to apply LoRA: {e}")
                raise

            self._register_lora_nan_guard()

        self._annotate_module_names()

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )
        self._annotate_module_names()

    def _cleanup_dataset_loader(self, dataset: Any, dataset_name: str) -> None:
        """Shuts down and clears a cached DynamicTorchDataset loader to free memory."""
        if dataset is None or not hasattr(dataset, "_loader"):
            return

        loader = getattr(dataset, "_loader", None)
        if loader is None:
            return

        try:
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()

            if hasattr(loader, "_shutdown_workers"):
                loader._shutdown_workers()
        except Exception as exc:
            logging.warning(f"Failed to shutdown {dataset_name} loader workers: {exc}")

        setattr(dataset, "_loader", None)
        del loader
        gc.collect()

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint_{epoch}" only.
        """
        # Allow disabling checkpoints via config: checkpoint.enabled: False
        if getattr(self.checkpoint_conf, "enabled", True) is False:
            logging.info("Checkpoint saving is disabled by config; skipping save.")
            return

        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            # Always use numbered checkpoint format
            checkpoint_names = [f"checkpoint_{int(epoch)}"]

        checkpoint_content = {
            "epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            self._cleanup_dataset_loader(self.train_dataset, dataset_name="train")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

                # Training loader has already been cleared prior to validation, but call
                # helper again to ensure no stale workers remain after val.
                self._cleanup_dataset_loader(self.train_dataset, dataset_name="train")

            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        # CRITICAL: Aggressive memory cleanup before validation to prevent OOM
        # Training leaves ~30GB allocated on GPU, need to free as much as possible
        logging.info(f"[Pre-validation cleanup] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Force synchronize to ensure all CUDA operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logging.info(f"[Post-cleanup] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)

        # Critical: Force cleanup of persistent worker memory
        # This prevents RAM accumulation in DataLoader workers between val and train epochs
        self._cleanup_dataset_loader(self.val_dataset, dataset_name="val")

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        # CRITICAL: Use same chunking as training to prevent OOM
        # Training uses accum_steps to chunk batches, validation should too
        val_chunk_size = self.accum_steps if hasattr(self, 'accum_steps') and self.accum_steps > 1 else 1
        if val_chunk_size > 1:
            logging.info(f"[Validation] Using batch chunking with chunk_size={val_chunk_size} to reduce memory")

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=False):
                batch = self._process_batch(batch)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # CRITICAL: Chunk batch to reduce memory usage (same as training)
            if val_chunk_size > 1:
                chunked_batches = chunk_batch_for_accum_steps(batch, val_chunk_size)
            else:
                chunked_batches = [batch]

            # Process each chunk separately to reduce peak memory
            for chunk_idx, chunk_batch in enumerate(chunked_batches):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_type):
                        val_loss_dict = self._step(
                            chunk_batch, self.model, phase, loss_meters
                        )

                # Clean up intermediate tensors between chunks
                if chunk_idx < len(chunked_batches) - 1:
                    torch.cuda.empty_cache()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        skipped_batches = 0

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break

            self._reset_iteration_profile_buffers()
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            batch_meta = self._format_batch_metadata(batch)
            if batch_meta in self.bad_batch_blacklist:
                failure_count = self._bad_batch_failures.get(batch_meta, 0)
                self._record_bad_batch_skip(
                    phase=phase,
                    reason=f"blacklisted batch (failures={failure_count})",
                    batch=batch,
                    log_level="warning",
                )
                skipped_batches += 1
                self._profile_finalize_iteration()
                continue

            h2d_token = self._profile_event_start("H2D")
            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            self._profile_event_end(h2d_token)

            if self.enable_batch_sanity_checks:
                is_valid, invalid_entries = self._validate_batch_tensors(batch)
                if not is_valid:
                    detail_msg = "; ".join(
                        f"{name}: nonfinite={count}, samples={samples}"
                        for name, count, samples in invalid_entries
                    )
                    self._record_bad_batch_skip(
                        phase=phase,
                        reason=f"non-finite tensors detected before forward ({detail_msg})",
                        batch=batch,
                        log_level="error",
                    )
                    self._note_bad_batch(batch, "non-finite tensors before forward", meta=batch_meta)
                    skipped_batches += 1
                    self._profile_finalize_iteration()
                    continue

            with torch.amp.autocast('cuda', enabled=False):
                batch = self._process_batch(batch)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            batch_success = self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # If batch had NaN/Inf, skip optimizer step and continue to next batch
            if not batch_success:
                logging.warning(f"Skipping optimizer step for iteration {data_iter} due to NaN/Inf loss")
                self._note_bad_batch(batch, "forward/backward failure", meta=batch_meta)
                skipped_batches += 1
                self._profile_finalize_iteration()
                continue

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            optim_step_token = self._profile_event_start("optimizer.step")
            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()
            self._profile_event_end(optim_step_token)

            if not self._cuda_sync_safely(
                phase=phase,
                batch=batch,
                location="post_optimizer_step_sync",
            ):
                self._note_bad_batch(batch, "cuda sync failure", meta=batch_meta)
                skipped_batches += 1
                self._profile_finalize_iteration()
                continue

            self._profile_finalize_iteration()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        if skipped_batches > 0:
            logging.info(
                f"Epoch {self.epoch} skipped {skipped_batches} {phase} batches due to anomalies."
            )

        self._log_iteration_profile_stats()
        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ) -> bool:
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward

        Returns:
            True if successful, False if NaN/Inf detected (skip this batch)
        """

        zero_grad_token = self._profile_event_start("optimizer.zero_grad")
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)
        self._profile_event_end(zero_grad_token)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        self._nan_guard_triggered = False
        for i, chunked_batch in enumerate(chunked_batches):
            self._nan_guard_batch_meta = self._format_batch_metadata(chunked_batch)
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_type):
                    try:
                        loss_dict = self._step(
                            chunked_batch, self.model, phase, loss_meters
                        )
                    except (RuntimeError, FloatingPointError) as err:
                        self._handle_step_failure(
                            phase=phase,
                            batch=chunked_batch,
                            err=err,
                            location="forward/loss",
                        )
                        return False

            if not self._cuda_sync_safely(
                phase=phase,
                batch=chunked_batch,
                location="post_forward_sync",
            ):
                return False

            loss_tensor = loss_dict["objective"]
            loss_key = f"Loss/{phase}_loss_objective"
            batch_size = chunked_batch["images"].shape[0]

            loss_value = self._profile_tensor_item(loss_tensor, "loss_check")

            if not math.isfinite(loss_value):
                error_msg = f"Loss is {loss_value}, skipping this batch and zeroing gradients"
                logging.warning(error_msg)
                self._note_bad_batch(chunked_batch, "non-finite loss value", meta=self._nan_guard_batch_meta)
                # Zero out gradients to avoid corrupted gradient state
                for optim in self.optims:
                    optim.zero_grad(set_to_none=True)
                return False
            loss = loss_tensor / accum_steps
            try:
                backward_token = self._profile_event_start("backward")
                try:
                    self.scaler.scale(loss).backward()
                finally:
                    self._profile_event_end(backward_token)
            except RuntimeError as err:
                self._handle_step_failure(
                    phase=phase,
                    batch=chunked_batch,
                    err=err,
                    location="backward",
                )
                return False
            loss_meter_value = self._profile_tensor_item(loss, "loss_meter")

            if not self._cuda_sync_safely(
                phase=phase,
                batch=chunked_batch,
                location="post_backward_sync",
            ):
                return False

            loss_meters[loss_key].update(loss_meter_value, batch_size)

        return True


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):
        # Debug: Print batch shapes
        if self.rank == 0 and self.steps.get('train', 0) == 0:
            logging.info("Batch shapes at start of _process_batch:")
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    logging.info(f"  {k}: {v.shape}")

        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        forward_token = self._profile_event_start("forward")
        try:
            y_hat = model(images=batch["images"])
        finally:
            self._profile_event_end(forward_token)
        
        # Loss computation
        loss_token = self._profile_event_start("loss")
        try:
            loss_dict = self.loss(y_hat, batch)
        finally:
            self._profile_event_end(loss_token)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _reset_amp_scaler(self) -> None:
        """Re-initializes the AMP scaler to avoid propagating NaNs."""
        if not isinstance(self.scaler, torch.cuda.amp.GradScaler):
            return
        if not self.optim_conf.amp.enabled:
            return
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        logging.warning("AMP GradScaler has been reset due to LoRA numerical instability.")

    def _cuda_sync_safely(
        self,
        *,
        phase: str,
        batch: Mapping,
        location: str,
    ) -> bool:
        """
        Forces CUDA synchronization to surface asynchronous kernel errors.

        Returns:
            True if synchronization succeeded or guard disabled. False otherwise.
        """
        if not self.enable_cuda_sync_guard or not torch.cuda.is_available():
            return True
        try:
            torch.cuda.synchronize()
        except RuntimeError as err:
            self._handle_step_failure(
                phase=phase,
                batch=batch,
                err=err,
                location=location,
            )
            return False
        return True

    def _register_lora_nan_guard(self) -> None:
        """Installs forward hooks on LoRA-targeted modules to detect non-finite activations."""
        if PeftModel is None or not isinstance(self.model, PeftModel):
            logging.debug("Skipping LoRA NaN guard registration: model is not a PeftModel.")
            return

        guard_modules: List[str] = []
        for name, module in self.model.named_modules():
            if self._module_has_lora(module):
                handle = module.register_forward_hook(self._make_nan_guard_hook(name))
                self._nan_guard_handles.append(handle)
                guard_modules.append(name)

        if guard_modules:
            logging.info(
                "Registered LoRA runtime guard on %d modules (rank %d). Examples: %s",
                len(guard_modules),
                self.rank,
                ", ".join(guard_modules[:5]),
            )
        else:
            logging.warning("No LoRA-aware modules found for runtime guard registration.")

    @staticmethod
    def _module_has_lora(module: nn.Module) -> bool:
        if isinstance(LoraLayer, tuple) and len(LoraLayer) == 0:
            return hasattr(module, "lora_A") or hasattr(module, "lora_embedding_A")
        return isinstance(module, LoraLayer)  # type: ignore[arg-type]

    def _make_nan_guard_hook(self, module_name: str) -> Callable:
        """Creates a forward hook that checks for NaN/Inf outputs."""

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if self._nan_guard_triggered:
                return

            issues: List[str] = []
            magnitudes_exceeded: List[str] = []
            with torch.no_grad():
                for idx, tensor in enumerate(self._iter_tensors(output)):
                    if tensor.numel() == 0 or not tensor.is_floating_point():
                        continue
                    detached = tensor.detach()
                    mask = ~torch.isfinite(detached)
                    if mask.any():
                        sample_values = (
                            detached[mask].reshape(-1)[:3].cpu().tolist()
                        )
                        issues.append(
                            f"idx={idx}, shape={tuple(detached.shape)}, samples={sample_values}"
                        )
                        continue

                    max_abs = detached.abs().max()
                    if torch.isfinite(max_abs) and max_abs > self.lora_guard_max_abs:
                        magnitudes_exceeded.append(
                            f"idx={idx}, shape={tuple(detached.shape)}, max_abs={float(max_abs)}"
                        )

            if issues or magnitudes_exceeded:
                self._nan_guard_triggered = True
                self._handle_lora_guard_alert(
                    module_name,
                    nonfinite=issues,
                    overflow=magnitudes_exceeded,
                )
                reason = issues[0] if issues else magnitudes_exceeded[0]
                raise LoRANumericalError(
                    f"LoRA guard triggered in module '{module_name}': {reason}"
                )

        return hook

    @staticmethod
    def _iter_tensors(obj: Any):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from Trainer._iter_tensors(item)
        elif isinstance(obj, Mapping):
            for item in obj.values():
                yield from Trainer._iter_tensors(item)

    def _handle_lora_guard_alert(
        self,
        module_name: str,
        *,
        nonfinite: List[str],
        overflow: List[str],
    ) -> None:
        """Logs the offending batch/module and resets state to keep training alive."""
        reasons = []
        if nonfinite:
            reasons.append(f"non-finite: {', '.join(nonfinite[:3])}")
        if overflow:
            reasons.append(f"magnitude>{self.lora_guard_max_abs}: {', '.join(overflow[:3])}")
        reason_text = " | ".join(reasons) if reasons else "unknown"
        logging.error(
            "Detected unstable LoRA activations in module '%s' | Batch: %s | "
            "Details: %s",
            module_name,
            self._nan_guard_batch_meta,
            reason_text,
        )
        self._reset_amp_scaler()

    def _annotate_module_names(self) -> None:
        """Attach hierarchical names to modules for improved debug logging."""
        module_iter = self.model.named_modules()
        for name, module in module_iter:
            debug_name = name if name else module.__class__.__name__
            try:
                setattr(module, "_debug_name", debug_name)
            except AttributeError:
                continue


    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                if torch.is_tensor(data[key]):
                    value = self._profile_tensor_item(data[key], f"scalar:{key}")
                else:
                    value = data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            def to_grid(x: torch.Tensor) -> torch.Tensor:
                # Normalize variety of shapes to [N,C,H,W]
                if x.dim() == 5:  # [B,S,C,H,W] or [B,S,H,W,1]
                    x = x[0]
                if x.dim() == 4:
                    # [S,C,H,W] or [S,H,W,1]
                    if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                        # likely NHWC -> NCHW
                        x = x.permute(0, 3, 1, 2).contiguous()
                elif x.dim() == 3:
                    # Could be [S,H,W] (no channel) or [C,H,W] (single image)
                    if x.shape[0] not in (1, 3):
                        # treat as [S,H,W]
                        x = x.unsqueeze(1)
                    else:
                        x = x.unsqueeze(0)
                elif x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                # Ensure channel in {1,3}
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                elif x.shape[1] != 3:
                    # Reduce or expand to 3 channels by selecting/averaging first 3
                    if x.shape[1] > 3:
                        x = x[:, :3]
                    else:
                        reps = (3 + x.shape[1] - 1) // x.shape[1]
                        x = x.repeat(1, reps, 1, 1)[:, :3]

                grid = torchvision.utils.make_grid(
                    x,
                    nrow=self.logging_conf.visuals_per_batch_to_log,
                )
                return grid

            grids = []
            for key in keys_to_log:
                if key in batch:
                    try:
                        grids.append(to_grid(batch[key]))
                    except Exception:
                        continue

            if not grids:
                return

            # Ensure same spatial size by min-cropping to smallest H,W
            min_h = min(g.shape[1] for g in grids)
            min_w = min(g.shape[2] for g in grids)
            grids = [g[:, :min_h, :min_w] for g in grids]

            visuals_to_log = torchvision.utils.make_grid(grids, nrow=1).clamp(-1, 1)

            visuals_to_log = self._profile_tensor_cpu(visuals_to_log, "visuals")
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = self._profile_tensor_numpy(visuals_to_log, "visuals")

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
