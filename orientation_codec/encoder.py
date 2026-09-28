from __future__ import annotations
"""
Encoder: Dream3D -> 16-bit RGB TIFF image.

Pipeline:
  1. Read Euler angles and grain labels from Dream3D HDF5
  2. Convert Euler angles (Bunge ZXZ, radians) to quaternions
  3. Continuous Unfolding: assign symmetry-equivalent quaternions for spatial smoothness
  4. Global Frame Centering: shift mean orientation to identity
  5. Stereographic Projection: quaternion -> 3D vector in [-1, 1]
  6. Pack to 16-bit [0, 65535] and save as TIFF
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from orientation_codec.quaternion import stereographic_forward, quat_ensure_positive_w
from orientation_codec.unfolding import continuous_unfold, global_frame_center
from orientation_codec.io_utils import read_dream3d, save_16bit_tiff, save_metadata


def _extract_grain_quats(euler_angles: np.ndarray, label_map: np.ndarray) -> dict[int, np.ndarray]:
    """
    Extract one representative quaternion per grain from Euler angle field.

    Parameters
    ----------
    euler_angles : (H, W, 3) Bunge Euler in radians
    label_map : (H, W) grain IDs

    Returns
    -------
    dict mapping grain_id -> (4,) quaternion [x, y, z, w]
    """
    unique_grains = np.unique(label_map)
    unique_grains = unique_grains[unique_grains != 0]

    grain_quats = {}
    for gid in unique_grains:
        coords = np.argwhere(label_map == gid)
        r, c = coords[0]
        euler_rad = euler_angles[r, c]
        quat = R.from_euler('ZXZ', euler_rad).as_quat()  # [x,y,z,w]
        grain_quats[int(gid)] = quat

    return grain_quats


def _build_rgb_image(
    label_map: np.ndarray,
    grain_quats: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Build a 16-bit RGB image from grain quaternions via stereographic projection.

    Parameters
    ----------
    label_map : (H, W) grain IDs
    grain_quats : dict grain_id -> (4,) quaternion

    Returns
    -------
    (H, W, 3) uint16 image
    """
    H, W = label_map.shape

    # Pre-compute stereographic projections for all grains
    grain_stereo = {}
    for gid, q in grain_quats.items():
        q = quat_ensure_positive_w(q)
        S = stereographic_forward(q)
        grain_stereo[gid] = S

    # Build image using vectorized lookup
    img = np.zeros((H, W, 3), dtype=np.float64)
    for gid, S in grain_stereo.items():
        mask = label_map == gid
        img[mask] = S

    # Map [-1, 1] -> [0, 65535]
    img_clipped = np.clip(img, -1.0, 1.0)
    img_16bit = ((img_clipped + 1.0) / 2.0 * 65535.0).round().astype(np.uint16)
    return img_16bit


def encode(
    euler_angles: np.ndarray,
    label_map: np.ndarray,
    output_dir: str | Path,
    name: str = "encoded",
    spacing: np.ndarray = None,
    dimensions: np.ndarray = None,
    source_path: str | Path = None,
) -> Path:
    """
    Full encoding pipeline: Euler angles + labels -> 16-bit TIFF + metadata.

    Parameters
    ----------
    euler_angles : (H, W, 3) Bunge Euler in radians
    label_map : (H, W) int grain IDs
    output_dir : directory for output files
    name : base filename (without extension)
    spacing : (3,) voxel spacing
    dimensions : (3,) grid dimensions
    source_path : path to original Dream3D file (saved in metadata for decoding)

    Returns
    -------
    Path to the saved TIFF image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if spacing is None:
        spacing = np.array([1.0, 1.0, 1.0])
    if dimensions is None:
        H, W = label_map.shape
        dimensions = np.array([W, H, 1])

    # Step 1: Extract per-grain quaternions
    grain_quats = _extract_grain_quats(euler_angles, label_map)
    print(f"  Extracted {len(grain_quats)} grain orientations")

    # Step 2: Continuous unfolding for spatial smoothness
    unfolded = continuous_unfold(label_map, grain_quats)
    print("  Continuous unfolding complete")

    # Step 3: Global frame centering
    centered, mean_q = global_frame_center(unfolded)
    print("  Global frame centering complete")

    # Step 4: Build 16-bit RGB via stereographic projection
    img_16bit = _build_rgb_image(label_map, centered)

    # Step 5: Save outputs
    tiff_path = output_dir / f"{name}.tiff"
    meta_path = output_dir / f"{name}_meta.json"
    labels_path = output_dir / f"{name}_labels.npy"

    save_16bit_tiff(tiff_path, img_16bit)
    save_metadata(meta_path, mean_q, spacing, dimensions,
                  source_path=str(source_path) if source_path else None)
    np.save(labels_path, label_map)

    print(f"  Saved: {tiff_path.name}, {meta_path.name}, {labels_path.name}")
    return tiff_path


def encode_dream3d(
    dream3d_path: str | Path,
    output_dir: str | Path = None,
    name: str = None,
) -> Path:
    """
    Encode a Dream3D file to a 16-bit TIFF image.

    Parameters
    ----------
    dream3d_path : path to .dream3d file
    output_dir : output directory (default: same as input)
    name : output base name (default: derived from input filename)

    Returns
    -------
    Path to the saved TIFF image
    """
    dream3d_path = Path(dream3d_path)
    if output_dir is None:
        output_dir = dream3d_path.parent
    if name is None:
        name = dream3d_path.stem

    print(f"Encoding: {dream3d_path.name}")
    data = read_dream3d(dream3d_path)
    return encode(
        data["euler_angles"],
        data["label_map"],
        output_dir,
        name,
        data["spacing"],
        data["dimensions"],
        source_path=dream3d_path,
    )
