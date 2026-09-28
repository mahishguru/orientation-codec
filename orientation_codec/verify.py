from __future__ import annotations
"""
Round-trip verification: encode -> decode -> compare misorientations.

Both the original and decoded orientations are folded to the fundamental zone
before comparison, which gives a direct, unambiguous angular distance.
"""

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from orientation_codec.symmetry import fold_to_fundamental_zone
from orientation_codec.encoder import encode_dream3d
from orientation_codec.decoder import decode_rgb_to_dream3d
from orientation_codec.io_utils import read_dream3d


def fz_misorientation_deg(euler1_rad: np.ndarray, euler2_rad: np.ndarray) -> float:
    """
    Compute misorientation (degrees) between two orientations by folding
    both into the HCP fundamental zone and measuring the direct angle.

    This is the correct comparison when orientations may have been encoded
    with different symmetry equivalents.
    """
    q1 = R.from_euler('ZXZ', euler1_rad).as_quat()
    q2 = R.from_euler('ZXZ', euler2_rad).as_quat()

    q1_fz = fold_to_fundamental_zone(q1)
    q2_fz = fold_to_fundamental_zone(q2)

    # Direct angle between FZ quaternions
    dot = np.clip(np.abs(np.dot(q1_fz, q2_fz)), 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return np.degrees(angle_rad)


def verify_round_trip(dream3d_path: str | Path, verbose: bool = True) -> dict:
    """
    Verify the encode -> decode round-trip for a Dream3D file.

    Parameters
    ----------
    dream3d_path : path to .dream3d file
    verbose : print detailed results

    Returns
    -------
    dict with keys: max_misori, mean_misori, median_misori, n_grains (all in degrees)
    """
    dream3d_path = Path(dream3d_path)
    if verbose:
        print(f"Verifying round-trip: {dream3d_path.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Encode
        tiff_path = encode_dream3d(dream3d_path, tmpdir)

        # Decode
        dream3d_out = decode_rgb_to_dream3d(tiff_path)

        # Load both
        orig = read_dream3d(dream3d_path)
        decoded = read_dream3d(dream3d_out)

    # Compare per-grain
    label_map = orig["label_map"]
    unique_grains = np.unique(label_map)
    unique_grains = unique_grains[unique_grains != 0]

    misori_angles = []
    for gid in unique_grains:
        coords = np.argwhere(label_map == gid)
        r, c = coords[0]
        e_orig = orig["euler_angles"][r, c]
        e_dec = decoded["euler_angles"][r, c]
        angle = fz_misorientation_deg(e_orig, e_dec)
        misori_angles.append(angle)

    misori = np.array(misori_angles)

    results = {
        "max_misori": float(misori.max()),
        "mean_misori": float(misori.mean()),
        "median_misori": float(np.median(misori)),
        "n_grains": len(misori),
    }

    if verbose:
        print(f"\n  Round-trip verification results:")
        print(f"  Grains:             {results['n_grains']}")
        print(f"  Max misorientation: {results['max_misori']:.4f} deg")
        print(f"  Mean misorientation:{results['mean_misori']:.4f} deg")
        print(f"  Median misori:      {results['median_misori']:.4f} deg")

        if results["max_misori"] < 0.1:
            print(f"  Status: EXCELLENT (sub-0.1 deg)")
        elif results["max_misori"] < 1.0:
            print(f"  Status: GOOD (sub-1 deg)")
        else:
            print(f"  Status: WARNING (>1 deg error detected)")

    return results
