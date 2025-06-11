# reconlib/depth_scale.py
from pathlib import Path
import math, struct, random
from typing import Tuple, List, Dict
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
S_FLIP = np.diag([1., -1., 1.])         # Y-down  →  Y-up

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers: parsing COLMAP TXT model (much simpler than *.bin)
# ─────────────────────────────────────────────────────────────────────────────
def load_points3d_txt(path_txt: Path) -> Dict[int, np.ndarray]:
    """Return dict {point3D_id: XYZ[np.float64]}"""
    pts = {}
    with path_txt.open() as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            toks = ln.split()
            pid = int(toks[0])
            xyz = np.array(list(map(float, toks[1:4])), dtype=np.float64)
            pts[pid] = xyz
    return pts


class ImageRec:
    __slots__ = ("name", "qvec", "tvec", "points_xy", "points3d_ids")
    def __init__(self, name: str, qvec: np.ndarray, tvec: np.ndarray,
                 pts_xy: np.ndarray, pids: np.ndarray):
        self.name, self.qvec, self.tvec = name, qvec, tvec
        self.points_xy, self.points3d_ids = pts_xy, pids


def load_images_txt(path_txt: Path) -> List[ImageRec]:
    """Parse COLMAP images.txt (two-line format per image)."""
    images = []
    with path_txt.open() as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        ln1 = lines[i].strip()
        if ln1.startswith("#") or not ln1:
            i += 1
            continue
        ln2 = lines[i + 1].strip()
        i += 2

        toks = ln1.split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = toks[:9]
        name = toks[9]
        qvec = np.array([float(qx), float(qy), float(qz), float(qw)],
                        dtype=np.float64)
        tvec = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)

        # second line: x y point3D_id ...
        pts = ln2.split()
        xy = np.asarray(list(map(float, pts[0::3])), dtype=np.float64).reshape(-1, 2)
        pids = np.asarray(list(map(int, pts[2::3])), dtype=np.int64)
        images.append(ImageRec(name, qvec, tvec, xy, pids))
    return images


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """qx,qy,qz,qw  →  3×3"""
    return R.from_quat(q).as_matrix()


def bilinear(depth: np.ndarray, x: float, y: float) -> float | None:
    h, w = depth.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None
    xf, yf = int(math.floor(x)), int(math.floor(y))
    xi, yi = x - xf, y - yf
    ztl, ztr = depth[yf, xf], depth[yf, xf + 1]
    zbl, zbr = depth[yf + 1, xf], depth[yf + 1, xf + 1]
    if min(ztl, ztr, zbl, zbr) <= 0.0:
        return None
    zl = ztl * (1 - yi) + zbl * yi
    zr = ztr * (1 - yi) + zbr * yi
    return zl * (1 - xi) + zr * xi


# ─────────────────────────────────────────────────────────────────────────────
#  Core routine
# ─────────────────────────────────────────────────────────────────────────────
def rgbd_scale_and_export(model_dir: Path,
                          depth_dir: Path,
                          intrinsics: Tuple[float, float, float, float],
                          out_path: Path | None = None,
                          max_depth_m: float = 2.0,
                          samples_per_image: int = 2000) -> Path:
    """
    * Samples up to `samples_per_image` RGB-D correspondences per image.
    * Estimates global scale = mean( z_depth / z_colmap ).
    * Scales translations, converts to camera→world, flips Y, writes 3×4 poses.
    Returns path to written poses file.
    """
    model_dir = Path(model_dir)
    depth_dir = Path(depth_dir)
    fx, fy, cx, cy = intrinsics
    images = load_images_txt(model_dir / "images.txt")
    pts3d  = load_points3d_txt(model_dir / "points3D.txt")

    ratios = []
    rng = random.Random(0)

    for im in images:
        depth = cv2.imread(
            str(depth_dir / f"{Path(im.name).stem}.exr"),
            cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue

        idxs = [j for j, pid in enumerate(im.points3d_ids) if pid >= 0]
        rng.shuffle(idxs)
        idxs = idxs[:samples_per_image]

        Rwc = qvec_to_rotmat(im.qvec).T      # world→cam  →  cam→world
        tcw = -Rwc @ im.tvec

        for j in idxs:
            pid = im.points3d_ids[j]
            P_w = pts3d[pid]
            P_c = Rwc.T @ (P_w - tcw)        # world→cam
            if P_c[2] <= 0:
                continue
            x = fx * P_c[0] / P_c[2] + cx
            y = fy * P_c[1] / P_c[2] + cy
            z_d = bilinear(depth, x, y)
            if z_d is None or z_d > max_depth_m:
                continue
            ratios.append(z_d / P_c[2])

    if not ratios:
        print("[WARN] No valid RGB-D correspondences, falling back to 1.0 scale")
        scale = 1.0
    else:
        ratios = np.asarray(ratios, dtype=np.float64)
        scale = ratios.mean()
        print(f"[INFO] global scale = {scale:.6f} m/colmap "
              f"(±{ratios.std():.6f}, {len(ratios)} samples)")

    # ---------------------------------------------------------------- poses
    lines = []
    for im in sorted(images, key=lambda x: x.name):
        Rwc = qvec_to_rotmat(im.qvec).T
        tcw = -Rwc @ im.tvec * scale              # → millimetres later
        Rfl = S_FLIP @ Rwc @ S_FLIP
        tfl = S_FLIP @ tcw
        T = np.hstack([Rfl, tfl[:, None]]).reshape(-1) * 1000.0  # metres→mm
        lines.append(" ".join(f"{v:.6f}" for v in T))

    if out_path is None:
        out_path = model_dir / "poses_mm_yup.txt"
    Path(out_path).write_text("\n".join(lines))
    print(f"[INFO] wrote {len(lines)} poses → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrappers (keep the simple median-depth API too, if you like)
# ─────────────────────────────────────────────────────────────────────────────
def metric_scale_mm_median(poses: np.ndarray,
                           depth_dir: Path,
                           max_depth_mm: int = 2000) -> float:
    """
    (original quick-and-dirty method, kept for reference)
    """
    med_depths = []
    for p in depth_dir.iterdir():
        if p.suffix == ".png":
            d = np.frombuffer(p.read_bytes(), dtype=np.uint16).astype(np.float32)
        else:
            d = np.load(p).astype(np.float32)
        d = d[(d > 0) & (d < max_depth_mm)]
        if d.size:
            med_depths.append(np.median(d))
    if not med_depths:
        return 1.0
    depth_med = np.median(med_depths)
    trans_med = np.median(np.linalg.norm(poses[:, :3], axis=1))
    return depth_med / (trans_med + 1e-9)


def flip_y_down_to_up(t_xyz: np.ndarray, q_xyzw: np.ndarray):
    """Apply S_FLIP to translation and quaternion."""
    Rmat = R.from_quat(q_xyzw).as_matrix()
    R_fl = S_FLIP @ Rmat @ S_FLIP
    q_out = R.from_matrix(R_fl).as_quat()
    t_out = S_FLIP @ t_xyz
    return t_out, q_out
