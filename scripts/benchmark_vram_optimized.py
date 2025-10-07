"""
Benchmark VRAM-optimized aggregator vs. baseline.

This script compares the baseline Aggregator with the VRAM-optimized
AggregatorVramOptimized on random inputs. It prints:
- Numerical equality checks for heads (camera and DPT depth).
- Wall-clock timings for baseline vs optimized forward passes.
- Peak CUDA memory (if running on GPU) for both variants.
- Estimated memory of stored intermediate layer outputs (in MB).

Arguments:
- `--device` (str): Device to run on. Defaults to `cuda` if available,
  otherwise `cpu`.
- `--batch` (int): Batch size B. Default: 1.
- `--seq` (int): Sequence length S (number of frames). Default: 4.
- `--img` (int): Square input size. H=W will be adjusted to be a multiple
  of `--patch`. Default: 112.
- `--patch` (int): Patch size used by the (conv) PatchEmbed. Default: 14.
- `--embed_dim` (int): Embedding dimension for the aggregators/heads.
  Default: 128.
- `--depth` (int): Number of alternating-attention blocks (total depth).
  Default: 24.
- `--heads` (int): Number of attention heads. Default: 16.
- `--selected` (str): Comma-separated absolute layer indices to keep in the
  optimized aggregator (e.g., "4,11,17,23"). When fewer layers are kept,
  intermediate activation storage is reduced.

Usage examples:
- `python scripts/benchmark_vram_optimized.py --device cuda`
- `python scripts/benchmark_vram_optimized.py --seq 8 --selected 3,7,11,15`
"""

import time
import argparse
import torch

from vggt.models.aggregator import Aggregator
from vggt.models.aggregator_vram_optimized import AggregatorVramOptimized, LayerSelectView
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--img", type=int, default=112, help="square image H=W")
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--selected", type=str, default="4,11,17,23")
    return ap.parse_args()


def tensor_list_nbytes(tensors):
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.element_size() * t.nelement()
    return total


def main():
    args = parse_args()
    torch.manual_seed(0)

    device = torch.device(args.device)
    B, S, img_size, patch_size = args.batch, args.seq, args.img, args.patch
    H = W = img_size - (img_size % patch_size)

    embed_dim, depth, num_heads = args.embed_dim, args.depth, args.heads
    sel = [int(x) for x in args.selected.split(",") if x.strip()]

    images = torch.rand(B, S, 3, H, W, device=device)

    agg_full = Aggregator(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
    ).to(device).eval()

    agg_opt = AggregatorVramOptimized(
        img_size=H,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
        selected_layer_idx=sel,
    ).to(device).eval()

    # align weights for equality check fairness
    agg_opt.load_state_dict(agg_full.state_dict(), strict=False)

    cam_head = CameraHead(dim_in=2 * embed_dim).to(device).eval()
    dpt_head = DPTHead(
        dim_in=2 * embed_dim,
        output_dim=2,
        features=min(128, embed_dim),
        out_channels=[min(128, embed_dim)] * 4,
        intermediate_layer_idx=sel,
        pos_embed=False,
        patch_size=patch_size,
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        full_list, patch_start = agg_full(images)
        # estimate bytes of stored outputs
        full_bytes = tensor_list_nbytes(full_list)
        cam_full = cam_head(full_list)
        d_full_pred, d_full_conf = dpt_head(full_list, images=images, patch_start_idx=patch_start)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    mem_full = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.no_grad():
        view, patch_start2, kept = agg_opt(images)
        assert isinstance(view, LayerSelectView)
        opt_bytes = tensor_list_nbytes(view._selected_outputs)  # type: ignore[attr-defined]
        cam_opt = cam_head(view)
        d_opt_pred, d_opt_conf = dpt_head(view, images=images, patch_start_idx=patch_start2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    mem_opt = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    print("Equality checks:")
    print("  pose_enc last equal:", bool(torch.allclose(cam_full[-1], cam_opt[-1], rtol=1e-4, atol=1e-5)))
    print("  depth pred equal:", bool(torch.allclose(d_full_pred, d_opt_pred, rtol=1e-4, atol=1e-5)))
    print("  depth conf equal:", bool(torch.allclose(d_full_conf, d_opt_conf, rtol=1e-4, atol=1e-5)))
    print()

    print(f"Timings (s): full={t1 - t0:.3f}, optimized={t3 - t2:.3f}")
    if mem_full is not None:
        print(f"Peak CUDA memory (bytes): full={mem_full}, optimized={mem_opt}")
    else:
        print("CUDA not available; memory stats skipped.")

    print(f"Stored layer outputs size (MB): full={full_bytes/1e6:.2f}, optimized={opt_bytes/1e6:.2f}")


if __name__ == "__main__":
    main()
