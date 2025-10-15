# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False
# Controls whether scaled_dot_product_attention is used.
# Default is disabled for LoRA fine-tuning stability; set the env variable
# VGGT_ENABLE_FLASH_ATTENTION=1 to re-enable fused attention kernels.
FLASH_ATTENTION_ENABLED = bool(int(os.getenv("VGGT_ENABLE_FLASH_ATTENTION", "0")))
ATTENTION_DEBUG_ENABLED = bool(int(os.getenv("VGGT_DEBUG_ATTENTION", "1")))
logger = logging.getLogger(__name__)


def _summarize_nonfinite_tensors(
    tensors: Iterable[Tuple[str, Tensor]]
) -> Tuple[bool, str]:
    """
    Inspect tensors for non-finite values (nan/inf) and return a summary string
    when issues are detected.
    """
    issues = []
    with torch.no_grad():
        for name, tensor in tensors:
            if tensor is None:
                continue
            detached = tensor.detach()
            if detached.numel() == 0:
                continue
            finite_mask = torch.isfinite(detached)
            if finite_mask.all():
                continue

            nan_count = torch.isnan(detached).sum().item()
            inf_count = torch.isinf(detached).sum().item()

            if finite_mask.any():
                max_abs = (
                    detached[finite_mask].abs().max().item()
                    if finite_mask.any()
                    else float("nan")
                )
            else:
                max_abs = float("nan")

            issues.append(
                f"{name}: nan={nan_count}, inf={inf_count}, max_abs={max_abs}"
            )
    if not issues:
        return False, ""
    return True, "; ".join(issues)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: Optional[bool] = None,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        if fused_attn is None:
            fused_attn = FLASH_ATTENTION_ENABLED
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        has_issue, summary = _summarize_nonfinite_tensors(
            (("q", q), ("k", k), ("v", v))
        )
        if has_issue:
            module_name = getattr(self, "_debug_name", self.__class__.__name__)
            lora_info = ""
            if hasattr(self.qkv, "lora_A"):
                try:
                    with torch.no_grad():
                        lora_entries = []
                        for key, layer in self.qkv.lora_A.items():
                            abs_max = layer.weight.detach().abs().max().item()
                            norm_val = layer.weight.detach().norm().item()
                            lora_entries.append(
                                f"A[{key}]:abs_max={abs_max:.3e},norm={norm_val:.3e}"
                            )
                        for key, layer in self.qkv.lora_B.items():
                            abs_max = layer.weight.detach().abs().max().item()
                            norm_val = layer.weight.detach().norm().item()
                            lora_entries.append(
                                f"B[{key}]:abs_max={abs_max:.3e},norm={norm_val:.3e}"
                            )
                        lora_info = "; ".join(lora_entries)
                except Exception as err:  # pragma: no cover - debug only
                    lora_info = f"error_collecting_lora_stats={err}"

            extra_debug = ""
            if ATTENTION_DEBUG_ENABLED:
                with torch.no_grad():
                    def _tensor_stats(t: Tensor) -> str:
                        finite_mask = torch.isfinite(t)
                        if not finite_mask.any():
                            return "all_nonfinite"
                        vals = t[finite_mask]
                        return (
                            f"min={vals.min().item():.3e},"
                            f"max={vals.max().item():.3e},"
                            f"mean={vals.mean().item():.3e},"
                            f"std={vals.std().item():.3e}"
                        )

                    extra_debug = (
                        f" | x_stats=({_tensor_stats(x)})"
                        f" | q_stats=({_tensor_stats(q)})"
                        f" | k_stats=({_tensor_stats(k)})"
                        f" | v_stats=({_tensor_stats(v)})"
                    )

            logger.error(
                "Non-finite attention inputs detected | module=%s | batch=%d, tokens=%d, heads=%d | %s%s%s",
                module_name,
                B,
                N,
                self.num_heads,
                summary,
                f" | lora=({lora_info})" if lora_info else "",
                extra_debug,
            )
            raise RuntimeError(
                f"Non-finite attention inputs detected before SDPA: {summary}"
            )

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
