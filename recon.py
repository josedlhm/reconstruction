# recon.py  – minimal, uses depth_scale.rgbd_scale_and_export with defaults
from pathlib import Path
from typing import Tuple

from .colmap_backend import run_colmap
from scale import rgbd_scale_and_export          # full RGB-D scaling routine
from .config import ReconConfig


def _load_intrinsics(src) -> Tuple[float, float, float, float]:
    """Accept tuple, CSV string, or txt file → (fx, fy, cx, cy)."""
    if isinstance(src, (tuple, list)) and len(src) == 4:
        return tuple(map(float, src))
    if isinstance(src, str) and "," in src:
        return tuple(map(float, src.split(",")))
    with open(src) as f:
        return tuple(map(float, f.readline().replace(",", " ").split()))


def reconstruct(images_dir=None, depth_dir=None, intrinsics=None, *,
                out_root="02_colmap", cfg: ReconConfig | None = None) -> Path:
    # 1) read from cfg if provided
    if cfg is not None:
        images_dir, depth_dir, intrinsics = \
            cfg.images_dir, cfg.depth_dir, cfg.intrinsics
        out_root = cfg.out_root

    images_dir = Path(images_dir).expanduser()
    depth_dir  = Path(depth_dir).expanduser()
    out_root   = Path(out_root).expanduser()

    fx, fy, cx, cy = _load_intrinsics(intrinsics)

    # 2) COLMAP
    run_colmap(images_dir, out_root, str(intrinsics))

    # 3) depth-aware scaling (uses its own 2 m depth limit & 2 k samples)
    poses_path = rgbd_scale_and_export(
        model_dir   = out_root,            # images.txt / points3D.txt live here
        depth_dir   = depth_dir,
        intrinsics  = (fx, fy, cx, cy),    # just pass the numbers
        out_path    = out_root / "poses_mm_yup.txt",
        # ← all other parameters left to their default values
    )
    return poses_path
