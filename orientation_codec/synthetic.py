"""
Synthetic 2D microstructures for tests, examples and quick sanity checks.

Generates a periodic-free Voronoi grain map with one orientation per grain,
so the codec can be exercised end to end without any external Dream3D data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from orientation_codec.io_utils import write_dream3d


def make_voronoi_microstructure(
    size: int | tuple[int, int] = 128,
    n_grains: int = 60,
    texture_spread_deg: float | None = 25.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 2D Voronoi grain map and a per-voxel Bunge Euler field.

    Parameters
    ----------
    size : image side length, or (H, W)
    n_grains : number of Voronoi seeds
    texture_spread_deg : if given, orientations are drawn as random rotations
        of at most this angle about a common reference (a crude fibre texture);
        if None, orientations are uniformly random on SO(3)
    seed : RNG seed

    Returns
    -------
    euler_angles : (H, W, 3) float32, Bunge ZXZ Euler angles in radians
    label_map    : (H, W) int32, 1-based grain IDs
    """
    H, W = (size, size) if np.isscalar(size) else size
    rng = np.random.default_rng(seed)

    seeds = rng.uniform([0, 0], [H, W], size=(n_grains, 2))
    rr, cc = np.mgrid[0:H, 0:W]
    _, nearest = cKDTree(seeds).query(np.column_stack([rr.ravel(), cc.ravel()]))
    label_map = (nearest.reshape(H, W) + 1).astype(np.int32)

    # Relabel so that IDs are contiguous (a seed may own no pixel)
    _, label_map = np.unique(label_map, return_inverse=True)
    label_map = (label_map.reshape(H, W) + 1).astype(np.int32)
    n_present = int(label_map.max())

    if texture_spread_deg is None:
        rots = R.random(n_present, random_state=rng.integers(2**31))
    else:
        axes = rng.normal(size=(n_present, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        angles = np.radians(texture_spread_deg) * rng.uniform(0, 1, size=(n_present, 1))
        rots = R.from_rotvec(axes * angles)

    grain_euler = np.mod(rots.as_euler("ZXZ"), 2.0 * np.pi).astype(np.float32)
    euler_angles = grain_euler[label_map - 1]
    return euler_angles, label_map


def write_voronoi_dream3d(path: str | Path, **kwargs) -> Path:
    """Generate a Voronoi microstructure (see above) and write it as a .dream3d file."""
    euler_angles, label_map = make_voronoi_microstructure(**kwargs)
    path = Path(path)
    write_dream3d(path, euler_angles, label_map)
    return path
