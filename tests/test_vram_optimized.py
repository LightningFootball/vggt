import os
import math
import torch

from vggt.models.aggregator import Aggregator
from vggt.models.aggregator_vram_optimized import AggregatorVramOptimized, LayerSelectView
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead


@torch.no_grad()
def test_selected_view_indexing_and_shape():
    torch.manual_seed(0)
    B, S, H, W = 1, 3, 64, 64
    embed_dim = 64
    depth = 8
    num_heads = 8
    patch_size = 16  # ensure H and W are multiples of patch
    images = torch.rand(B, S, 3, H, W)

    # Baseline aggregator (stores all layers)
    agg_full = Aggregator(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
    ).eval()

    # Optimized aggregator (stores only selected)
    selected_idx = [1, 3, 5, 7]
    agg_opt = AggregatorVramOptimized(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
        selected_layer_idx=selected_idx,
    ).eval()

    full_list, patch_start_full = agg_full(images)
    view, patch_start_opt, kept = agg_opt(images)

    assert isinstance(view, LayerSelectView)
    assert len(full_list) == depth
    assert len(view) == depth
    assert patch_start_full == patch_start_opt
    assert kept == sorted(selected_idx)
    assert view.selected_count == len(selected_idx)

    # Access last absolute layer (-1) must work and match
    x_full_last = full_list[-1]
    x_view_last = view[-1]
    assert torch.allclose(x_full_last, x_view_last, rtol=1e-5, atol=1e-6)

    # Access a non-selected index should raise
    try:
        _ = view[0]
        assert False, "Expected KeyError for non-selected index"
    except KeyError:
        pass


@torch.no_grad()
def test_heads_numerical_equivalence():
    torch.manual_seed(1234)
    B, S, H, W = 1, 3, 64, 64
    embed_dim = 64
    depth = 8
    num_heads = 8
    patch_size = 16  # ensure multiples
    images = torch.rand(B, S, 3, H, W)

    # Aggregators
    agg_full = Aggregator(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
    ).eval()

    sel = [1, 3, 5, 7]
    agg_opt = AggregatorVramOptimized(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
        selected_layer_idx=sel,
    ).eval()

    # Run aggregators
    full_list, patch_start_full = agg_full(images)
    view, patch_start_opt, kept = agg_opt(images)
    assert patch_start_full == patch_start_opt

    # Camera head compares last layer only
    cam_head = CameraHead(dim_in=2 * embed_dim).eval()
    cam_full = cam_head(full_list)
    cam_opt = cam_head(view)
    assert len(cam_full) == len(cam_opt) == 4
    assert torch.allclose(cam_full[-1], cam_opt[-1], rtol=1e-5, atol=1e-6)

    # DPT head using absolute indices (robust scheme)
    dpt_head = DPTHead(
        dim_in=2 * embed_dim,
        output_dim=2,
        features=64,
        out_channels=[64, 64, 64, 64],
        intermediate_layer_idx=sel,  # absolute indices
        patch_size=patch_size,
        pos_embed=False,
    ).eval()

    d_full_pred, d_full_conf = dpt_head(full_list, images=images, patch_start_idx=patch_start_full)
    d_opt_pred, d_opt_conf = dpt_head(view, images=images, patch_start_idx=patch_start_opt)

    assert torch.allclose(d_full_pred, d_opt_pred, rtol=1e-4, atol=1e-5)
    assert torch.allclose(d_full_conf, d_opt_conf, rtol=1e-4, atol=1e-5)
