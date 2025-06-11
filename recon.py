from pathlib import Path
import numpy as np
from .colmap_backend import run_colmap
from .depth_scale import metric_scale_mm, flip_y_down_to_up
from .config import ReconConfig

def reconstruct(images_dir=None, depth_dir=None, intrinsics_txt=None,
                *, out_root="02_colmap", scale_depth=True,
                cfg: ReconConfig | None = None) -> Path:
    if cfg is not None:                                     # load from config
        images_dir, depth_dir, intrinsics_txt = \
            cfg.images_dir, cfg.depth_dir, cfg.intrinsics
        out_root, scale_depth = cfg.out_root, cfg.scale_depth

    images_dir = Path(images_dir).expanduser()
    depth_dir  = Path(depth_dir).expanduser()
    out_root   = Path(out_root).expanduser()

    poses_txt = run_colmap(images_dir, out_root, str(intrinsics_txt))
    poses = np.loadtxt(poses_txt)

    if scale_depth:
        s = metric_scale_mm(poses, depth_dir)
        poses[:, :3] *= s

    for i in range(len(poses)):                            # Y-flip
        t, q = flip_y_down_to_up(poses[i, :3], poses[i, 3:])
        poses[i, :3] = t
        poses[i, 3:] = q

    final_txt = out_root / "poses_mm_yup.txt"
    np.savetxt(final_txt, poses, fmt="%.6f")
    return final_txt
