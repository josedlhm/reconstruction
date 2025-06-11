#!/usr/bin/env python3
import yaml
from recon import reconstruct

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)           # reads the 4 keys

poses_path = reconstruct(
    images_dir = cfg["images_dir"],
    depth_dir  = cfg["depth_dir"],
    intrinsics = cfg["intrinsics"],
    out_root   = cfg.get("out_root", "02_colmap"),
)

print(f"\n✓ Finished – poses written to {poses_path}")
