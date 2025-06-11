# recon.py  –  single-line fix
from pathlib import Path
from typing import Tuple
from colmap_backend import run_colmap
from scale import rgbd_scale_and_export


def _load_intrinsics(src) -> Tuple[float, float, float, float]:
    if isinstance(src, (tuple, list)) and len(src) == 4:
        return tuple(map(float, src))
    if isinstance(src, str) and "," in src:
        return tuple(map(float, src.split(",")))
    with open(src) as f:
        return tuple(map(float, f.readline().replace(",", " ").split()))


def reconstruct(images_dir, depth_dir, intrinsics, *, out_root="02_colmap"):
    images_dir = Path(images_dir).expanduser()
    depth_dir  = Path(depth_dir).expanduser()
    out_root   = Path(out_root).expanduser()

    fx, fy, cx, cy = _load_intrinsics(intrinsics)
    intrin_csv = f"{fx},{fy},{cx},{cy}"          # ← NEW: guaranteed COLMAP-safe

    # 1) COLMAP
    run_colmap(images_dir, out_root, intrin_csv)

    # 2) RGB-D scaling
    return rgbd_scale_and_export(
        model_dir  = out_root,
        depth_dir  = depth_dir,
        intrinsics = (fx, fy, cx, cy),
        out_path   = out_root / "poses_mm_yup.txt",
    )
