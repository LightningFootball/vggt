import os
import json
import time
import logging
import math
import contextlib
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist

# Reuse utilities from the existing training stack
from train_utils.general import (
    AverageMeter,
    ProgressMeter,
    copy_data_to_device,
    safe_makedirs,
    is_dist_avail_and_initialized,
)

from .trainer import Trainer as BaseTrainer


class TrainerBatchOptimized(BaseTrainer):
    """
    A thin extension of the existing Trainer that:
    - Forces accum_steps=1 (unified per requirement)
    - Logs additional step-level metrics (VRAM, step/data time, frames/sec, utilization)
    - Supports periodic validation by step (val_step_freq)
    - Writes JSONL per-step metrics and run-level summary with optional warmup skip

    Note: Original training code remains untouched; this is a separate entrypoint.
    """

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
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        # New knobs
        val_step_freq: int = 0,
        warmup_skip_steps: int = 0,
        jsonl_dir: Optional[str] = "report/baselines",
        **kwargs,
    ):
        # Force accum_steps to 1 as requested
        super_kwargs = dict(
            data=data,
            model=model,
            logging=logging,
            checkpoint=checkpoint,
            max_epochs=max_epochs,
            mode=mode,
            device=device,
            seed_value=seed_value,
            val_epoch_freq=val_epoch_freq,
            distributed=distributed,
            cuda=cuda,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            optim=optim,
            loss=loss,
            env_variables=env_variables,
            accum_steps=1,
        )
        super().__init__(**super_kwargs)

        # New configs
        self.val_step_freq = int(val_step_freq) if val_step_freq is not None else 0
        self.warmup_skip_steps = int(warmup_skip_steps) if warmup_skip_steps is not None else 0

        # JSONL logging setup (rank 0 only)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.jsonl_dir = jsonl_dir or "report/baselines"
        self.jsonl_path = None
        self._jsonl_fh = None
        if self.rank == 0:
            safe_makedirs(self.jsonl_dir)
            self.jsonl_path = os.path.join(self.jsonl_dir, f"metrics_{self.run_id}.jsonl")
            self._jsonl_fh = open(self.jsonl_path, "w", buffering=1)
            logging.info(f"JSONL metrics will be written to: {self.jsonl_path}")

        # Accumulators for summaries
        self._train_records: List[Dict[str, Any]] = []
        self._val_records: List[Dict[str, Any]] = []

        # Cache basic experiment info
        try:
            self.batch_frames_capacity = int(self.data_conf.train.max_img_per_gpu)
        except Exception:
            # Fallback if not present
            self.batch_frames_capacity = 0

    # ------------- Helpers -------------
    def _all_reduce_sum(self, tensor_val: float) -> float:
        if not is_dist_avail_and_initialized():
            return float(tensor_val)
        t = torch.tensor([tensor_val], device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def _all_reduce_max(self, tensor_val: float) -> float:
        if not is_dist_avail_and_initialized():
            return float(tensor_val)
        t = torch.tensor([tensor_val], device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def _tb_log(self, tag: str, value: Any, step: int) -> None:
        try:
            self.tb_writer.log(tag, value, step)
        except Exception:
            pass

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        if self._jsonl_fh is None:
            return
        try:
            self._jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.warning(f"Failed to write JSONL record: {e}")

    def _collect_scalar_values(self, phase: str, loss_meters: Dict[str, AverageMeter]) -> Dict[str, float]:
        # Reads current `val` from AverageMeters conforming to logging.scalar_keys_to_log
        keys = self._get_scalar_log_keys(phase)
        scalars: Dict[str, float] = {}
        for k in keys:
            meter_name = f"Loss/{phase}_{k}"
            if meter_name in loss_meters:
                scalars[k] = float(loss_meters[meter_name].val)
        return scalars

    def _log_step_metrics(
        self,
        *,
        phase: str,
        step_idx: int,
        batch: Mapping,
        batch_time_meter: AverageMeter,
        data_time_meter: AverageMeter,
        loss_meters: Dict[str, AverageMeter],
    ) -> Dict[str, Any]:
        """Compute and log extended metrics for the current step; return the record."""
        # Basic batch structure
        images = batch.get("images", None)
        if images is not None and isinstance(images, torch.Tensor) and images.ndim >= 2:
            B = int(images.shape[0])
            S = int(images.shape[1]) if images.ndim >= 5 else 1
        else:
            B, S = 0, 0
        frames = B * S

        # Timings
        step_time = float(batch_time_meter.val)
        data_time = float(data_time_meter.val)
        fps_local = float(frames / step_time) if step_time > 0 and frames > 0 else 0.0

        # Memory
        if torch.cuda.is_available():
            mem_alloc = float(torch.cuda.memory_allocated() / 1e9)
            mem_reserved = float(torch.cuda.memory_reserved() / 1e9)
            mem_peak_alloc_epoch = float(torch.cuda.max_memory_allocated() / 1e9)
        else:
            mem_alloc = mem_reserved = mem_peak_alloc_epoch = 0.0

        # World-level reductions
        world_frames = self._all_reduce_sum(frames)
        world_step_time = self._all_reduce_max(step_time)
        fps_world = float(world_frames / world_step_time) if world_step_time > 0 and world_frames > 0 else 0.0
        mem_alloc_world_max = self._all_reduce_max(mem_alloc)

        # Utilization
        frames_capacity = max(1, int(self.batch_frames_capacity))
        frames_util = float(frames / frames_capacity) if frames_capacity > 0 else 0.0

        # Collect configured scalar losses
        loss_scalars = self._collect_scalar_values(phase, loss_meters)
        loss_objective = loss_scalars.get("loss_objective", None)

        # TensorBoard logging
        self._tb_log(f"Perf/{phase}/StepTime_s", step_time, step_idx)
        self._tb_log(f"Perf/{phase}/DataTime_s", data_time, step_idx)
        self._tb_log(f"Perf/{phase}/FramesPerStep_local", frames, step_idx)
        self._tb_log(f"Perf/{phase}/FramesPerSec_local", fps_local, step_idx)
        self._tb_log(f"Perf/{phase}/FramesPerSec_world", fps_world, step_idx)
        self._tb_log(f"System/{phase}/VRAM_Allocated_GB", mem_alloc, step_idx)
        self._tb_log(f"System/{phase}/VRAM_Reserved_GB", mem_reserved, step_idx)
        self._tb_log(f"System/{phase}/VRAM_PeakAllocated_GB_epoch", mem_peak_alloc_epoch, step_idx)
        self._tb_log(f"System/{phase}/VRAM_Allocated_GB_world_max", mem_alloc_world_max, step_idx)
        self._tb_log(f"Batch/{phase}/NumSeqs_B", B, step_idx)
        self._tb_log(f"Batch/{phase}/FramesPerSeq_S", S, step_idx)
        self._tb_log(f"Batch/{phase}/FramesCapacity", frames_capacity, step_idx)
        self._tb_log(f"Batch/{phase}/FramesUtilization", frames_util, step_idx)

        # Create record for JSONL and summary
        record = {
            "run_id": self.run_id,
            "phase": phase,
            "epoch": float(self.epoch),
            "step": int(step_idx),
            "where": float(self.where),
            "B": B,
            "S": S,
            "frames": frames,
            "frames_capacity": frames_capacity,
            "frames_utilization": frames_util,
            "step_time_s": step_time,
            "data_time_s": data_time,
            "fps_local": fps_local,
            "fps_world": fps_world,
            "vram_alloc_gb": mem_alloc,
            "vram_reserved_gb": mem_reserved,
            "vram_peak_alloc_epoch_gb": mem_peak_alloc_epoch,
            "vram_alloc_world_max_gb": mem_alloc_world_max,
        }
        # Add loss scalars (unified and components)
        for k, v in loss_scalars.items():
            record[k] = float(v)

        # Persist and cache
        if phase == "train":
            self._train_records.append(record)
        else:
            self._val_records.append(record)
        if self.rank == 0:
            self._write_jsonl(record)

        return record

    # ------------- Overrides -------------
    @torch.no_grad()
    def val_epoch(self, val_loader):  # type: ignore[override]
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

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    _ = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # Extended metrics logging (per step)
            self._log_step_metrics(
                phase=phase,
                step_idx=self.steps[phase],
                batch=batch,
                batch_time_meter=batch_time,
                data_time_meter=data_time,
                loss_meters=loss_meters,
            )

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def train_epoch(self, train_loader):  # type: ignore[override]
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

        if self.gradient_clipper is not None:
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

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Enforce accum_steps==1
            chunked_batches = [batch]

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

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
                                f"{i}_" if len(self.optims) > 1 else ("" + f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
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

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # Extended metrics logging (per step)
            self._log_step_metrics(
                phase=phase,
                step_idx=self.steps[phase],
                batch=batch,
                batch_time_meter=batch_time,
                data_time_meter=data_time,
                loss_meters=loss_meters,
            )

            # Periodic validation by step (if configured)
            if (
                self.val_step_freq is not None
                and self.val_step_freq > 0
                and self.steps[phase] % self.val_step_freq == 0
            ):
                if self.val_dataset is not None:
                    val_loader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
                    try:
                        self.val_epoch(val_loader)
                    finally:
                        del val_loader
                        torch.cuda.empty_cache()

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    # ------------- Summary -------------
    def _aggregate(self, records: List[Dict[str, Any]], warmup_skip: int) -> Dict[str, Any]:
        def _stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            mean = float(sum(vals_sorted) / n)
            median = float(vals_sorted[n // 2])
            p90 = float(vals_sorted[min(n - 1, int(math.ceil(0.9 * n)) - 1)])
            return {"mean": mean, "median": median, "p90": p90, "max": float(max(vals_sorted))}

        # Apply warmup skip
        filtered = [r for r in records if r.get("step", 0) > warmup_skip]

        def col(name: str) -> List[float]:
            return [float(r.get(name, 0.0)) for r in filtered if name in r]

        out = {
            "count": len(filtered),
            "loss_objective": _stats(col("loss_objective")),
            "step_time_s": _stats(col("step_time_s")),
            "fps_local": _stats(col("fps_local")),
            "fps_world": _stats(col("fps_world")),
            "frames_utilization": _stats(col("frames_utilization")),
            "vram_alloc_gb": _stats(col("vram_alloc_gb")),
            "vram_peak_alloc_epoch_gb": _stats(col("vram_peak_alloc_epoch_gb")),
        }
        # Component losses (if present)
        for key in [
            "loss_T",
            "loss_R",
            "loss_FL",
            "loss_conf_depth",
            "loss_reg_depth",
            "loss_grad_depth",
        ]:
            if any(key in r for r in filtered):
                out[key] = _stats(col(key))
        return out

    def write_summaries(self) -> Optional[str]:
        """Write run-level summary JSON to report/baselines; return path on rank0."""
        if self.rank == 0:
            safe_makedirs(self.jsonl_dir)
            bf = self.batch_frames_capacity
            total_steps = (self.limit_train_batches + 1) if self.limit_train_batches is not None else len(self._train_records)
            summary_path = os.path.join(self.jsonl_dir, f"summary_bf{bf}_s{total_steps}.json")
            summary_payload = {
                "run_id": self.run_id,
                "batch_frames": bf,
                "steps": total_steps,
                "warmup_skip_steps": self.warmup_skip_steps,
                "train": self._aggregate(self._train_records, self.warmup_skip_steps),
            }
            if self._val_records:
                summary_payload["val"] = self._aggregate(self._val_records, 0)
            with open(summary_path, "w") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            logging.info(f"Summary written to: {summary_path}")
            return summary_path
        return None

    def __del__(self):
        try:
            if self._jsonl_fh is not None:
                self._jsonl_fh.close()
        except Exception:
            pass

