from __future__ import annotations
"""
Dataset-level operations for neural network training pipelines.

Provides:
  - compute_class_means: scan a dataset tree and compute one global mean
    quaternion per alloy class (top-level directory)
  - encode_dream3d_global: encode using a pre-computed global mean instead
    of a per-sample mean (outputs 8-bit PNG or 16-bit TIFF)
  - decode_pixelwise: decode any RGB image (uint8 PNG or uint16 TIFF)
    using only a global mean (no per-sample labels or meta JSON required)
  - segment_grains: recover a grain label map from a decoded image via
    simple spatial colour clustering
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from orientation_codec.io_utils import (
    read_dream3d,
    save_16bit_tiff,
    load_16bit_tiff,
    save_png,
    load_png,
    write_dream3d,
    write_xdmf,
)
from orientation_codec.quaternion import (
    average_quaternion,
    quat_ensure_positive_w,
    stereographic_forward,
    stereographic_inverse,
)
from orientation_codec.symmetry import fold_to_fundamental_zone
from orientation_codec.unfolding import continuous_unfold


# ============================================================
# 1. Compute per-class global mean quaternion
# ============================================================

def _extract_mean_quat_from_file(dream3d_path: str) -> np.ndarray:
    """Worker: read one Dream3D file and return the FZ-folded per-grain mean quaternion."""
    data = read_dream3d(dream3d_path)
    euler_angles = data["euler_angles"]
    label_map = data["label_map"]

    unique_grains = np.unique(label_map)
    unique_grains = unique_grains[unique_grains != 0]

    # Collect one quaternion per grain, fold each into FZ for consistency
    quats = []
    for gid in unique_grains:
        coords = np.argwhere(label_map == gid)
        r, c = coords[0]
        q = R.from_euler('ZXZ', euler_angles[r, c]).as_quat()
        q_fz = fold_to_fundamental_zone(q)
        quats.append(q_fz)

    return average_quaternion(np.array(quats))


def compute_class_means(
    root_dir: str | Path,
    output_path: str | Path = None,
    samples_per_class: int = 50,
    workers: int = 8,
) -> dict[str, list[float]]:
    """
    Compute one global mean quaternion per alloy class (top-level subdirectory).

    Samples up to `samples_per_class` Dream3D files per class, extracts
    per-file mean quaternions, then averages them into a single class mean.

    Parameters
    ----------
    root_dir : root dataset directory (children = alloy class folders)
    output_path : JSON file to save the class means (default: root_dir/class_means.json)
    samples_per_class : how many files to sample per class (default: 50)
    workers : parallel processes

    Returns
    -------
    dict mapping class_name -> [qx, qy, qz, qw]
    """
    root_dir = Path(root_dir)
    if output_path is None:
        output_path = root_dir / "class_means.json"

    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    class_means = {}

    for class_dir in class_dirs:
        class_name = class_dir.name
        dream3d_files = sorted(class_dir.rglob("*.dream3d"))

        if not dream3d_files:
            print(f"[{class_name}] No .dream3d files found, skipping")
            continue

        # Sample a subset
        rng = np.random.default_rng(42)
        n_sample = min(samples_per_class, len(dream3d_files))
        sampled = rng.choice(dream3d_files, size=n_sample, replace=False)

        file_means = []
        effective_workers = min(workers, n_sample)

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_extract_mean_quat_from_file, str(f)): f
                for f in sampled
            }
            with tqdm(total=n_sample, desc=class_name, unit="file", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        q = future.result()
                        file_means.append(q)
                    except Exception as e:
                        pbar.write(f"  ERROR: {futures[future].name}: {e}")
                    pbar.update(1)

        if file_means:
            class_mean = average_quaternion(np.array(file_means))
            class_mean = quat_ensure_positive_w(class_mean)
            class_means[class_name] = class_mean.tolist()
            print(f"  [{class_name}] mean quaternion: {class_mean.round(6).tolist()}")

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(class_means, indent=2))
    print(f"\nSaved class means to {output_path}")

    return class_means


def load_class_means(path: str | Path) -> dict[str, np.ndarray]:
    """Load class_means.json and return dict mapping class_name -> (4,) quaternion."""
    data = json.loads(Path(path).read_text())
    return {k: np.array(v) for k, v in data.items()}


# ============================================================
# 2. Encode with global (class-level) mean quaternion
# ============================================================

def _extract_grain_quats(euler_angles, label_map):
    """Extract one quaternion per grain."""
    unique_grains = np.unique(label_map)
    unique_grains = unique_grains[unique_grains != 0]
    grain_quats = {}
    for gid in unique_grains:
        coords = np.argwhere(label_map == gid)
        r, c = coords[0]
        q = R.from_euler('ZXZ', euler_angles[r, c]).as_quat()
        grain_quats[int(gid)] = q
    return grain_quats


def _center_with_mean(grain_quats, mean_q):
    """Shift all quaternions so that mean_q becomes identity."""
    mean_rot_inv = R.from_quat(mean_q).inv()
    centered = {}
    for gid, q in grain_quats.items():
        q_c = (mean_rot_inv * R.from_quat(q)).as_quat()
        if q_c[3] < 0:
            q_c = -q_c
        centered[gid] = q_c
    return centered


def _build_rgb_image(label_map, grain_quats):
    """Build 16-bit RGB image from grain quaternions."""
    H, W = label_map.shape
    img = np.zeros((H, W, 3), dtype=np.float64)
    for gid, q in grain_quats.items():
        q = quat_ensure_positive_w(q)
        S = stereographic_forward(q)
        mask = label_map == gid
        img[mask] = S
    img_clipped = np.clip(img, -1.0, 1.0)
    return ((img_clipped + 1.0) / 2.0 * 65535.0).round().astype(np.uint16)


def encode_dream3d_global(
    dream3d_path: str | Path,
    global_mean_q: np.ndarray,
    output_dir: str | Path = None,
    name: str = None,
    fmt: str = "png",
) -> Path:
    """
    Encode a Dream3D file using a pre-computed global mean quaternion.

    Same pipeline as encode_dream3d but uses `global_mean_q` instead of
    computing a per-sample mean. The saved _meta.json stores the global
    mean, so ANY sample can be decoded with the same class_means.json.

    Parameters
    ----------
    dream3d_path : path to .dream3d file
    global_mean_q : (4,) quaternion [x, y, z, w] — the class-level mean
    output_dir : output directory (default: same as input)
    name : output base name (default: derived from input filename)
    fmt : output image format — ``"png"`` (8-bit, default, for pretrained
          vision models) or ``"tiff"`` (16-bit, lossless)

    Returns
    -------
    Path to the saved image
    """
    dream3d_path = Path(dream3d_path)
    if output_dir is None:
        output_dir = dream3d_path.parent
    if name is None:
        name = dream3d_path.stem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_dream3d(dream3d_path)
    euler_angles = data["euler_angles"]
    label_map = data["label_map"]
    spacing = data.get("spacing", np.array([1.0, 1.0, 1.0]))
    H, W = label_map.shape
    dimensions = data.get("dimensions", np.array([W, H, 1]))

    # Step 1: Extract per-grain quaternions
    grain_quats = _extract_grain_quats(euler_angles, label_map)

    # Step 2: Continuous unfolding anchored to the global mean
    unfolded = continuous_unfold(label_map, grain_quats, anchor_q=global_mean_q)

    # Step 3: Center with the GLOBAL mean (not per-sample)
    centered = _center_with_mean(unfolded, global_mean_q)

    # Step 4: Build 16-bit RGB via stereographic projection
    img_16bit = _build_rgb_image(label_map, centered)

    # Step 5: Save output (global mean is in class_means.json;
    #          grain labels are recovered via segmentation at decode time)
    fmt = fmt.lower().lstrip(".")
    if fmt == "tiff":
        out_path = output_dir / f"{name}.tiff"
        save_16bit_tiff(out_path, img_16bit)
    else:
        # Default: 8-bit PNG — compatible with pretrained vision models
        out_path = output_dir / f"{name}.png"
        save_png(out_path, img_16bit)

    return out_path


# ============================================================
# 3. Pixel-wise decoder (no labels required)
# ============================================================

def decode_pixelwise(
    img_16bit: np.ndarray,
    mean_q: np.ndarray,
) -> np.ndarray:
    """
    Decode a 16-bit RGB image to Euler angles — pixel by pixel, no grain labels needed.

    Parameters
    ----------
    img_16bit : (H, W, 3) uint16
    mean_q : (4,) global shift quaternion

    Returns
    -------
    euler_angles : (H, W, 3) float32, Bunge Euler in radians
    """
    H, W = img_16bit.shape[:2]

    # Unpack to stereographic coordinates [-1, 1]
    # Support both 16-bit TIFF (uint16) and 8-bit PNG (uint8)
    max_val = 65535.0 if img_16bit.dtype == np.uint16 else 255.0
    stereo = (img_16bit.astype(np.float64) / max_val) * 2.0 - 1.0  # (H, W, 3)

    # Vectorized inverse stereographic projection
    S_flat = stereo.reshape(-1, 3)                # (N, 3)
    S2 = np.sum(S_flat ** 2, axis=1, keepdims=True)  # (N, 1)
    denom = 1.0 + S2                              # (N, 1)
    qw = (1.0 - S2) / denom                       # (N, 1)
    q_xyz = 2.0 * S_flat / denom                  # (N, 3)
    quats = np.hstack([q_xyz, qw])                # (N, 4) [x,y,z,w]

    # Normalize
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= norms

    # Ensure w >= 0
    neg_w = quats[:, 3] < 0
    quats[neg_w] *= -1

    # Undo global frame centering (apply mean quaternion)
    global_shift = R.from_quat(mean_q)
    recovered = (global_shift * R.from_quat(quats)).as_quat()

    # Vectorized fold into HCP fundamental zone
    # Crystal symmetry acts on the RIGHT for an active crystal->sample
    # orientation (g * S). Left-multiplication (S * g) would rotate the crystal
    # in the sample frame and scramble the texture (see symmetry.py).
    from orientation_codec.symmetry import HCP_SYMMETRIES

    r_recovered = R.from_quat(recovered)
    best_w = np.full(len(recovered), -1.0)
    best_q = recovered.copy()

    for sym in HCP_SYMMETRIES:
        q_sym = (r_recovered * sym).as_quat()   # (N, 4) — vectorized, right-multiply
        neg = q_sym[:, 3] < 0
        q_sym[neg] *= -1
        better = q_sym[:, 3] > best_w
        best_w[better] = q_sym[better, 3]
        best_q[better] = q_sym[better]

    euler_flat = np.mod(
        R.from_quat(best_q).as_euler('ZXZ'), 2.0 * np.pi
    ).astype(np.float32)

    return euler_flat.reshape(H, W, 3)


def decode_image_pixelwise(
    image_path: str | Path,
    mean_q: np.ndarray,
    output_path: str | Path = None,
    label_map: np.ndarray = None,
    spacing: np.ndarray = None,
    phase_name: str = "AZ31",
) -> Path:
    """
    Decode a PNG or TIFF image to Dream3D using only the global mean quaternion.

    Accepts 8-bit PNG or 16-bit TIFF. No per-sample _meta.json or _labels.npy
    required. If label_map is not provided, grain segmentation is recovered
    from the decoded image.

    Parameters
    ----------
    image_path : path to .png (8-bit) or .tiff (16-bit) file
    mean_q : (4,) global mean quaternion (from class_means.json)
    output_path : output .dream3d path (default: same dir, _decoded suffix)
    label_map : optional grain label map; recovered via segmentation if None
    spacing : (3,) voxel spacing, default [1, 1, 1]
    phase_name : phase name written into CellEnsembleData/PhaseName

    Returns
    -------
    Path to the saved Dream3D file
    """
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_decoded.dream3d"
    output_path = Path(output_path)

    if spacing is None:
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Auto-detect format from extension
    if image_path.suffix.lower() == ".png":
        img = load_png(image_path)
    else:
        img = load_16bit_tiff(image_path)

    euler_angles = decode_pixelwise(img, mean_q)

    H, W = euler_angles.shape[:2]
    dimensions = np.array([W, H, 1])

    if label_map is None:
        label_map = segment_grains(img)

    write_dream3d(output_path, euler_angles, label_map, spacing, phase_name=phase_name)
    xdmf_path = output_path.with_suffix(".xdmf")
    write_xdmf(xdmf_path, output_path.name, dimensions, spacing)

    return output_path


# Backward compat alias
decode_tiff_pixelwise = decode_image_pixelwise


# ============================================================
# 4. Grain segmentation recovery from TIFF
# ============================================================

def segment_grains(img: np.ndarray, colour_tol: int = None) -> np.ndarray:
    """
    Recover a grain label map from an RGB image.

    Uses connected-component labelling: two adjacent pixels belong to the
    same grain if all three channels differ by less than `colour_tol`.

    For a clean image from the encoder, adjacent within-grain pixels are
    identical, so this perfectly recovers the original labels. For a
    network-generated image with some noise, `colour_tol` absorbs small
    deviations.

    Parameters
    ----------
    img : (H, W, 3) uint8 or uint16
    colour_tol : maximum per-channel difference for same-grain.
                 Default: 50 for uint16, 1 for uint8.

    Returns
    -------
    label_map : (H, W) int32, 1-based grain IDs (0 = none assigned, shouldn't happen)
    """
    from scipy.ndimage import label as ndlabel
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if colour_tol is None:
        colour_tol = 50 if img.dtype == np.uint16 else 1

    H, W = img.shape[:2]
    img = img.astype(np.int32)

    # Build same-neighbor masks (vectorized)
    h_same = np.abs(img[:, :-1, :] - img[:, 1:, :]).max(axis=2) <= colour_tol  # (H, W-1)
    v_same = np.abs(img[:-1, :, :] - img[1:, :, :]).max(axis=2) <= colour_tol  # (H-1, W)

    # Build sparse adjacency graph (fully vectorized, no Python loops)
    N = H * W

    # Horizontal edges: (r, c) <-> (r, c+1)
    hr, hc = np.where(h_same)
    h_idx1 = hr * W + hc
    h_idx2 = hr * W + hc + 1

    # Vertical edges: (r, c) <-> (r+1, c)
    vr, vc = np.where(v_same)
    v_idx1 = vr * W + vc
    v_idx2 = (vr + 1) * W + vc

    # Symmetric edges
    row = np.concatenate([h_idx1, h_idx2, v_idx1, v_idx2])
    col = np.concatenate([h_idx2, h_idx1, v_idx2, v_idx1])
    data = np.ones(len(row), dtype=np.int8)

    graph = csr_matrix((data, (row, col)), shape=(N, N))
    n_comp, labels = connected_components(graph, directed=False)

    # Convert from 0-based to 1-based labels
    label_map = (labels + 1).astype(np.int32).reshape(H, W)
    return label_map


# ============================================================
# 5. Batch encode with global mean
# ============================================================

def _process_one_global(args):
    """Worker for batch_encode_global. Returns (out_path | None, error_str | None)."""
    import io, sys
    fpath_str, out_dir_str, mean_q_list, fmt = args
    fpath = Path(fpath_str)
    out_dir = Path(out_dir_str)
    mean_q = np.array(mean_q_list)
    _devnull = io.StringIO()
    try:
        sys.stdout = _devnull
        sys.stderr = _devnull
        out_path = encode_dream3d_global(fpath, mean_q, out_dir, fmt=fmt)
        return str(out_path), None
    except Exception as e:
        return None, f"{fpath.name}: {e}"
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def _is_done(fpath, out_dir, fmt="png"):
    """Check if the output image already exists (checks both PNG and TIFF)."""
    ext = fmt.lower().lstrip(".")
    if (out_dir / f"{fpath.stem}.{ext}").exists():
        return True
    # Also accept the other format so we don't re-encode files from a prior run
    alt = "tiff" if ext == "png" else "png"
    return (out_dir / f"{fpath.stem}.{alt}").exists()


def batch_encode_global(
    root_dir: str | Path,
    class_means_path: str | Path,
    output_dir: str | Path = None,
    flat_output: bool = False,
    workers: int = 8,
    skip_existing: bool = True,
    fmt: str = "png",
) -> list[Path]:
    """
    Batch encode all Dream3D files using per-class global mean quaternions.

    The class is determined by the top-level subdirectory name in root_dir.

    Parameters
    ----------
    root_dir : root directory (children = alloy class folders)
    class_means_path : path to class_means.json from compute_class_means()
    output_dir : base output directory (default: in-place next to inputs)
    flat_output : if True, flatten all outputs into output_dir
    workers : parallel worker processes
    skip_existing : skip files with existing outputs
    fmt : output image format — ``"png"`` (8-bit, default) or ``"tiff"`` (16-bit)

    Returns
    -------
    List of paths to generated image files
    """
    root_dir = Path(root_dir)
    class_means = load_class_means(class_means_path)

    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    results = []
    total_errors = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_means:
            print(f"[{class_name}] No mean quaternion found, skipping")
            continue

        mean_q = class_means[class_name]
        mean_q_list = mean_q.tolist()

        files = sorted(class_dir.rglob("*.dream3d"))
        if not files:
            continue

        # Build work list
        work = []
        skipped = 0
        for fpath in files:
            if output_dir is None:
                out_dir = fpath.parent
            elif flat_output:
                out_dir = Path(output_dir)
            else:
                rel = fpath.parent.relative_to(root_dir)
                out_dir = Path(output_dir) / rel

            if skip_existing and _is_done(fpath, out_dir, fmt):
                skipped += 1
            else:
                work.append((str(fpath), str(out_dir), mean_q_list, fmt))

        total = len(work)
        if skipped:
            print(f"[{class_name}] {skipped} already done, {total} remaining")
        if not work:
            continue

        errors = []
        effective_workers = min(workers, total)

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_process_one_global, item): item for item in work}
            with tqdm(total=total, desc=class_name, unit="file", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    tiff_str, err = future.result()
                    if tiff_str:
                        results.append(Path(tiff_str))
                    else:
                        errors.append(err)
                    pbar.update(1)
                    if err:
                        pbar.write(f"  ERROR: {err}")

        total_errors.extend(errors)

    print(f"\nBatch complete: {len(results)} files encoded")
    if total_errors:
        print(f"{len(total_errors)} errors total")
    return results
