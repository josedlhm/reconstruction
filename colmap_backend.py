from __future__ import annotations
import logging, shutil, subprocess
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def _run(cmd: list[str]):                 # simple wrapper
    logger.info("▶ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_colmap(images_dir: Path, out_dir: Path, intrinsics_txt: str,
               *, quality="high", use_gpu=True,
               merge_models=True, bundle_adjust=True) -> Path:
    sparse_dir = out_dir / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)

    _run([
        "colmap", "automatic_reconstructor",
        "--workspace_path", out_dir, "--image_path", images_dir,
        "--dense", "0", "--quality", quality,
        "--use_gpu", "1" if use_gpu else "0",
        "--camera_model", "PINHOLE",
        "--camera_params", intrinsics_txt,
        "--data_type", "video", "--single_camera", "1",
    ])

    models = sorted([d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name))
    if not models:
        raise RuntimeError("No sparse model from COLMAP")

    final = models[0]
    if merge_models and len(models) > 1:
        for i in range(1, len(models)):
            merged = sparse_dir / f"merged_{i}"
            _run(["colmap", "model_merger",
                  "--input_path1", final, "--input_path2", models[i],
                  "--output_path", merged])
            final = merged

    if bundle_adjust:
        refined = sparse_dir / "refined"
        _run(["colmap", "bundle_adjuster",
              "--input_path", final, "--output_path", refined])
        final = refined

    _run(["colmap", "model_converter",
          "--input_path", final, "--output_path", out_dir, "--output_type", "TXT"])

    poses = []
    with open(out_dir / "images.txt") as f:
        for ln in f:
            if ln.startswith("#"): continue
            qw,qx,qy,qz,tx,ty,tz = map(float, ln.split()[1:8])
            poses.append((tx,ty,tz,qx,qy,qz,qw))
    poses_path = out_dir / "poses_raw.txt"
    np.savetxt(poses_path, poses)
    (out_dir / "SUCCESS").touch()
    logger.info("✓ COLMAP done → %s", poses_path)
    return poses_path
