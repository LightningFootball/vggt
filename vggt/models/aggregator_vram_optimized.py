import logging
from typing import Optional, Tuple, List, Dict, Any, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class LayerSelectView:
    """
    A lightweight, list-like view over a sparse set of selected layer outputs.

    - Pretends to have length = `depth` (full number of AA blocks),
      so absolute indexing like [-1] and [23] works.
    - Materializes and stores only the selected layers' tensors.
    - __getitem__(abs_idx) returns the stored tensor for that absolute index;
      raises KeyError for indices not in the selected set.
    """

    def __init__(self, depth: int, selected_outputs: List[torch.Tensor], orig_indices: List[int]):
        self._depth = int(depth)
        self._selected_outputs = selected_outputs
        self._orig_indices = list(orig_indices)
        self._abs_to_rel: Dict[int, int] = {abs_i: rel_i for rel_i, abs_i in enumerate(self._orig_indices)}

    def __len__(self) -> int:
        return self._depth

    def __getitem__(self, idx: int) -> torch.Tensor:
        if isinstance(idx, slice):
            raise TypeError("LayerSelectView does not support slicing; index by absolute layer id.")
        if idx < 0:
            idx = self._depth + idx
        if idx < 0 or idx >= self._depth:
            raise IndexError(f"Index {idx} out of range for depth {self._depth}")
        if idx not in self._abs_to_rel:
            raise KeyError(
                f"Requested layer index {idx} not selected. Available: {self._orig_indices}"
            )
        return self._selected_outputs[self._abs_to_rel[idx]]

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def selected_count(self) -> int:
        return len(self._selected_outputs)

    @property
    def orig_indices(self) -> List[int]:
        return list(self._orig_indices)

    def __repr__(self) -> str:
        return f"LayerSelectView(depth={self._depth}, selected={self._orig_indices})"


class AggregatorVramOptimized(nn.Module):
    """
    Drop-in alternative to Aggregator that reduces VRAM/CPU RAM by storing
    only a subset of intermediate outputs. It returns a list-like view which
    supports absolute indexing (e.g., [-1], [4], [11], ...), so existing heads
    that reference absolute layer indices keep working without changes.

    Args are kept compatible with the original Aggregator. Additional arg:
    - selected_layer_idx (Optional[Sequence[int]]): absolute indices to retain.
      If None, keeps all layers (exactly baseline behavior).
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        selected_layer_idx: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Special tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False

        # Selection config
        if selected_layer_idx is None:
            self._selected_abs_idx: Optional[List[int]] = None
        else:
            uniq_sorted = sorted(set(int(i) for i in selected_layer_idx))
            for i in uniq_sorted:
                if i < 0 or i >= self.depth:
                    raise ValueError(f"selected_layer_idx contains out-of-range index {i} for depth {self.depth}")
            self._selected_abs_idx = uniq_sorted

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[LayerSelectView, int, List[int]]:
        """
        Args:
            images (torch.Tensor): [B, S, 3, H, W], range [0, 1]

        Returns:
            (LayerSelectView, int, List[int]):
                - A list-like view supporting absolute indexing (length=self.depth),
                  materialized only on the selected layer indices (if provided).
                - patch_start_idx
                - orig_indices: the absolute indices actually stored (sorted)
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            pos = pos + 1 if pos is not None else None
            if pos is not None:
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        selected_outputs: List[torch.Tensor] = []
        selected_abs_indices: List[int] = []

        layer_id = 0
        want_all = self._selected_abs_idx is None
        want_set = set() if want_all else set(self._selected_abs_idx)

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                if want_all or (layer_id in want_set):
                    selected_outputs.append(concat_inter)
                    selected_abs_indices.append(layer_id)
                layer_id += 1

        view = LayerSelectView(depth=self.depth, selected_outputs=selected_outputs, orig_indices=selected_abs_indices)
        return view, self.patch_start_idx, selected_abs_indices

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=False)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

