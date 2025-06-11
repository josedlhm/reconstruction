# cvrecon/colmap.py
from pathlib import Path
from utils import load_intrinsics, run
import numpy as np

def run_colmap(
    images_dir: Path,
    out_dir: Path,
    intrinsics: tuple[float, float, float, float],
    *,
    quality: str = "high",
    gpu: bool = True,
    merge: bool = True,
    bundle_adjust: bool = True,
) -> Path:
    """
    Launch COLMAP AutomaticReconstructor, optionally merges sub-models,
    runs BundleAdjuster, converts to TXT, and dumps poses_raw.txt.
    Returns path to that poses file.
    """
    images_dir = images_dir.expanduser()
    out_dir    = out_dir.expanduser()
    sparse_dir = out_dir / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy = intrinsics
    intrin_txt = f"{fx},{fy},{cx},{cy}"

    run([
        "colmap", "automatic_reconstructor",
        "--workspace_path", out_dir, "--image_path", images_dir,
        "--dense", "0", "--quality", quality,
        "--use_gpu", "1" if gpu else "0",
        "--camera_model", "PINHOLE",
        "--camera_params", intrin_txt,
        "--data_type", "video", "--single_camera", "1",
    ])

    models = sorted([d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name))
    if not models:
        raise RuntimeError("COLMAP produced no sparse model")

    final = models[0]
    if merge and len(models) > 1:
        for mdl in models[1:]:
            merged = sparse_dir / f"merged_{mdl.name}"
            run(["colmap", "model_merger",
                 "--input_path1", final, "--input_path2", mdl,
                 "--output_path", merged])
            final = merged

    if bundle_adjust:
        refined = sparse_dir / "refined"
        refined.mkdir(exist_ok=True)
        run(["colmap", "bundle_adjuster",
             "--input_path", final, "--output_path", refined])
        final = refined

    run(["colmap", "model_converter",
         "--input_path", final, "--output_path", out_dir, "--output_type", "TXT"])

    poses = []
    with open(out_dir / "images.txt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            qw,qx,qy,qz,tx,ty,tz = map(float, ln.split()[1:8])
            poses.append((tx,ty,tz,qx,qy,qz,qw))
    poses_path = out_dir / "poses_raw.txt"
    np.savetxt(poses_path, poses)
    (out_dir / "SUCCESS").touch()
    return poses_path
