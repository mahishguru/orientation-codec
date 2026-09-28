from __future__ import annotations
"""
Decoder: 16-bit RGB TIFF image -> Dream3D file with Euler angles.

Pipeline:
  1. Load 16-bit TIFF and metadata JSON
  2. Unpack [0, 65535] -> stereographic coordinates [-1, 1]
  3. Grain-wise median pooling (using saved label map) to reject boundary blur
  4. Inverse stereographic projection -> unit quaternion (no square roots)
  5. Undo global frame centering (apply saved mean quaternion)
  6. Fold into HCP fundamental zone
  7. Convert to Bunge Euler angles and write Dream3D HDF5
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from orientation_codec.quaternion import stereographic_inverse
from orientation_codec.symmetry import fold_to_fundamental_zone
from orientation_codec.io_utils import (
    load_16bit_tiff, load_metadata,
    write_dream3d, write_xdmf, copy_dream3d_with_new_eulers,
)


def decode(
    img_16bit: np.ndarray,
    label_map: np.ndarray,
    mean_q: np.ndarray,
) -> np.ndarray:
    """
    Decode a 16-bit RGB image back to Euler angles.

    Parameters
    ----------
    img_16bit : (H, W, 3) uint16
    label_map : (H, W) int grain IDs
    mean_q : (4,) global shift quaternion from encoding

    Returns
    -------
    euler_angles : (H, W, 3) float32, Bunge Euler in radians
    """
    H, W = label_map.shape

    # Step 1: Unpack to stereographic coordinates [-1, 1]
    stereo = (img_16bit.astype(np.float64) / 65535.0) * 2.0 - 1.0

    # Step 2: Grain-wise median pooling
    unique_grains = np.unique(label_map)
    unique_grains = unique_grains[unique_grains != 0]

    global_shift = R.from_quat(mean_q)
    euler_field = np.zeros((H, W, 3), dtype=np.float32)

    for gid in unique_grains:
        mask = label_map == gid

        # Median of pixels for this grain (ignores blurred boundaries)
        grain_pixels = stereo[mask]
        median_S = np.median(grain_pixels, axis=0)

        # Step 3: Inverse stereographic projection (rational, no square roots)
        q_recovered = stereographic_inverse(median_S)

        # Step 4: Undo global frame centering
        original_rot = global_shift * R.from_quat(q_recovered)

        # Step 5: Fold to HCP fundamental zone
        q_fz = fold_to_fundamental_zone(original_rot.as_quat())

        # Step 6: Convert to Bunge Euler (ZXZ) in radians
        euler_rad = R.from_quat(q_fz).as_euler('ZXZ')
        euler_rad = np.mod(euler_rad, 2.0 * np.pi)  # Ensure [0, 2pi)

        euler_field[mask] = euler_rad.astype(np.float32)

    return euler_field


def decode_rgb_to_dream3d(
    tiff_path: str | Path,
    meta_path: str | Path = None,
    labels_path: str | Path = None,
    output_path: str | Path = None,
) -> Path:
    """
    Decode a 16-bit TIFF back to a Dream3D file with Euler angles.

    If the metadata contains a reference to the original Dream3D file and
    that file still exists, the original is copied and only the EulerAngles
    dataset is replaced (preserving all other metadata). Otherwise, a new
    Dream3D file is created from scratch.

    Parameters
    ----------
    tiff_path : path to 16-bit .tiff file
    meta_path : path to metadata JSON (default: inferred from tiff_path)
    labels_path : path to label map .npy (default: inferred from tiff_path)
    output_path : output .dream3d path (default: same dir, same stem + _decoded)

    Returns
    -------
    Path to the saved Dream3D file
    """
    tiff_path = Path(tiff_path)
    stem = tiff_path.stem
    parent = tiff_path.parent

    if meta_path is None:
        meta_path = parent / f"{stem}_meta.json"
    if labels_path is None:
        labels_path = parent / f"{stem}_labels.npy"
    if output_path is None:
        output_path = parent / f"{stem}_decoded.dream3d"

    meta_path = Path(meta_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)

    print(f"Decoding: {tiff_path.name}")

    # Load inputs
    img_16bit = load_16bit_tiff(tiff_path)
    meta = load_metadata(meta_path)
    label_map = np.load(labels_path)
    mean_q = np.array(meta["global_shift_quaternion"])
    spacing = np.array(meta.get("spacing", [1.0, 1.0, 1.0]))
    dimensions = np.array(meta.get("dimensions", [label_map.shape[1], label_map.shape[0], 1]))

    # Decode orientations
    euler_angles = decode(img_16bit, label_map, mean_q)

    # Write Dream3D output
    source_path = meta.get("source_dream3d")
    if source_path and Path(source_path).exists():
        # Copy original and replace EulerAngles (preserves all metadata)
        copy_dream3d_with_new_eulers(source_path, output_path, euler_angles)
        print(f"  Saved (from original template): {output_path.name}")
    else:
        # Create new Dream3D file from scratch
        write_dream3d(output_path, euler_angles, label_map, spacing)
        print(f"  Saved (new file): {output_path.name}")

    # Write companion XDMF for ParaView
    xdmf_path = output_path.with_suffix(".xdmf")
    write_xdmf(xdmf_path, output_path.name, dimensions, spacing)
    print(f"  Saved: {xdmf_path.name}")

    return output_path
