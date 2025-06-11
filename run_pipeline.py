#!/usr/bin/env python3
# cvrecon/cli.py  –  minimal launcher

import sys
from pathlib import Path
import yaml

from utils  import load_intrinsics
from colmap import run_colmap
from scale  import rgbd_scale_and_export


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m cvrecon.cli <config.yaml>", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(sys.argv[1]).expanduser()
    cfg = yaml.safe_load(cfg_path.read_text())

    images_dir = Path(cfg["images_dir"]).expanduser()
    depth_dir  = Path(cfg["depth_dir"]).expanduser()
    out_root   = Path(cfg["out_root"]).expanduser()
    intrinsics = load_intrinsics(cfg["intrinsics"])

    print("Running COLMAP …")
    run_colmap(images_dir, out_root, intrinsics)

    print("Scaling with RGB-D …")
    poses_path = rgbd_scale_and_export(
        model_dir  = out_root,
        depth_dir  = depth_dir,
        intrinsics = intrinsics,
        out_path   = out_root / "poses_mm_yup.txt",
    )

    print(f"✓ Finished – poses written to {poses_path}")


if __name__ == "__main__":
    main()
