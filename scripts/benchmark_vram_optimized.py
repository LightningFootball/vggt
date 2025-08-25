import time
import torch

from vggt.models.aggregator import Aggregator
from vggt.models.aggregator_vram_optimized import AggregatorVramOptimized
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, S, H, W = 1, 4, 112, 112
    embed_dim = 128
    depth = 24
    num_heads = 16

    images = torch.rand(B, S, 3, H, W, device=device)

    sel = [4, 11, 17, 23]

    agg_full = Aggregator(
        img_size=H,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
    ).to(device).eval()

    agg_opt = AggregatorVramOptimized(
        img_size=H,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed="conv",
        selected_layer_idx=sel,
    ).to(device).eval()

    cam_head = CameraHead(dim_in=2 * embed_dim).to(device).eval()
    dpt_head = DPTHead(
        dim_in=2 * embed_dim,
        output_dim=2,
        features=128,
        out_channels=[128, 128, 128, 128],
        intermediate_layer_idx=sel,
        pos_embed=False,
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        full_list, patch_start = agg_full(images)
        cam_full = cam_head(full_list)
        d_full_pred, d_full_conf = dpt_head(full_list, images=images, patch_start_idx=patch_start)
    t1 = time.perf_counter()
    mem_full = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    with torch.no_grad():
        view, patch_start2, kept = agg_opt(images)
        cam_opt = cam_head(view)
        d_opt_pred, d_opt_conf = dpt_head(view, images=images, patch_start_idx=patch_start2)
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


if __name__ == "__main__":
    main()
