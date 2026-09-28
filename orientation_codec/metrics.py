"""
Round-trip fidelity metrics for HCP orientation fields.

- hcp_disorientation_deg: symmetry-aware disorientation angle between two
  orientation sets (minimum over the 12 proper HCP rotations)
- grain_disorientations: one disorientation per grain between two Euler fields
- boundary_f1: agreement of grain-boundary pixels between two label maps
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from orientation_codec.symmetry import HCP_SYMMETRIES


def hcp_disorientation_deg(euler_a: np.ndarray, euler_b: np.ndarray) -> np.ndarray:
    """
    Disorientation angle (degrees) between paired Bunge Euler orientations.

    min over s in S of angle(g_a^-1 * g_b * s), with S the 12 proper HCP rotations.

    Parameters
    ----------
    euler_a, euler_b : (..., 3) Bunge ZXZ Euler angles in radians

    Returns
    -------
    (...,) array of angles in degrees
    """
    shape = np.shape(euler_a)[:-1]
    ra = R.from_euler("ZXZ", np.reshape(euler_a, (-1, 3)))
    rb = R.from_euler("ZXZ", np.reshape(euler_b, (-1, 3)))
    delta = ra.inv() * rb
    best_w = np.zeros(len(delta))
    for sym in HCP_SYMMETRIES:
        w = np.abs((delta * sym).as_quat()[:, 3])
        best_w = np.maximum(best_w, w)
    angles = np.degrees(2.0 * np.arccos(np.clip(best_w, 0.0, 1.0)))
    return angles.reshape(shape)


def grain_disorientations(
    euler_ref: np.ndarray,
    euler_test: np.ndarray,
    label_map: np.ndarray,
) -> np.ndarray:
    """
    Per-grain disorientation (degrees) between a reference and a test Euler field.

    Each grain is represented by its first voxel in the reference label map,
    matching the per-grain constant orientation of Dream3D RVEs.
    """
    gids, first = np.unique(label_map.ravel(), return_index=True)
    first = first[gids != 0]
    ea = euler_ref.reshape(-1, 3)[first]
    eb = euler_test.reshape(-1, 3)[first]
    return hcp_disorientation_deg(ea, eb)


def _boundary_mask(label_map: np.ndarray) -> np.ndarray:
    b = np.zeros(label_map.shape, dtype=bool)
    h = label_map[:, :-1] != label_map[:, 1:]
    v = label_map[:-1, :] != label_map[1:, :]
    b[:, :-1] |= h
    b[:, 1:] |= h
    b[:-1, :] |= v
    b[1:, :] |= v
    return b


def boundary_f1(label_ref: np.ndarray, label_test: np.ndarray) -> float:
    """Pixel-level F1 score of grain-boundary masks (1.0 = identical boundaries)."""
    a, b = _boundary_mask(label_ref), _boundary_mask(label_test)
    tp = np.count_nonzero(a & b)
    fp = np.count_nonzero(~a & b)
    fn = np.count_nonzero(a & ~b)
    if tp == 0:
        return 1.0 if (fp == 0 and fn == 0) else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
