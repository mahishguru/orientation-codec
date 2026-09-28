"""
Continuous Unfolding algorithm for grain orientations.

Instead of forcing quaternions into a fundamental zone (which creates artificial
color discontinuities), this algorithm traverses the grain adjacency graph via BFS
and picks the symmetry-equivalent quaternion for each grain that is closest to its
already-assigned neighbor. The result is a spatially smooth orientation field that
compresses well through neural networks.
"""

import numpy as np
from collections import deque, defaultdict
from scipy.spatial.transform import Rotation as R

from orientation_codec.symmetry import HCP_SYMMETRIES
from orientation_codec.quaternion import average_quaternion


def build_grain_adjacency(label_map: np.ndarray) -> dict[int, list[int]]:
    """
    Build an adjacency graph from a 2D label map.

    Two grains are adjacent if they occupy horizontally or vertically
    neighboring pixels. Background (label 0) is excluded.

    Returns
    -------
    dict mapping grain_id -> list of neighboring grain_ids
    """
    H, W = label_map.shape
    edges = set()

    # Horizontal neighbors
    mask_h = label_map[:, :-1] != label_map[:, 1:]
    rows_h, cols_h = np.where(mask_h)
    for r, c in zip(rows_h, cols_h):
        u, v = int(label_map[r, c]), int(label_map[r, c + 1])
        if u != 0 and v != 0:
            edges.add((min(u, v), max(u, v)))

    # Vertical neighbors
    mask_v = label_map[:-1, :] != label_map[1:, :]
    rows_v, cols_v = np.where(mask_v)
    for r, c in zip(rows_v, cols_v):
        u, v = int(label_map[r, c]), int(label_map[r + 1, c])
        if u != 0 and v != 0:
            edges.add((min(u, v), max(u, v)))

    adjacency = defaultdict(list)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    return dict(adjacency)


def continuous_unfold(
    label_map: np.ndarray,
    grain_quats: dict[int, np.ndarray],
    anchor_q: np.ndarray = None,
) -> dict[int, np.ndarray]:
    """
    Assign symmetry-equivalent quaternions to grains so that neighbors
    have the closest possible orientations (maximum quaternion dot product).

    Parameters
    ----------
    label_map : (H, W) int array of grain IDs (0 = background)
    grain_quats : dict mapping grain_id -> (4,) quaternion [x,y,z,w]
    anchor_q : optional (4,) reference quaternion. If provided, the BFS root
               is aligned to this anchor first, so the entire unfolded tree
               ends up in the same region of quaternion space. Used by
               global-mean encoding.

    Returns
    -------
    dict mapping grain_id -> unfolded quaternion [x,y,z,w]
    """
    adjacency = build_grain_adjacency(label_map)
    unique_grains = sorted(set(grain_quats.keys()))

    unfolded = {}
    visited = set()

    # Find the largest grain as BFS root for the main cluster
    grain_sizes = {}
    for gid in unique_grains:
        grain_sizes[gid] = np.count_nonzero(label_map == gid)
    sorted_grains = sorted(unique_grains, key=lambda g: grain_sizes[g], reverse=True)

    for start_grain in sorted_grains:
        if start_grain in visited:
            continue

        queue = deque([start_grain])
        q_start = grain_quats[start_grain]

        if anchor_q is not None:
            # Align root grain to the anchor quaternion
            rot_raw = R.from_quat(q_start)
            best_q = None
            best_dot = -1.0
            for sym in HCP_SYMMETRIES:
                # Right-multiply: crystal symmetry for an active crystal->sample
                # orientation is g * S (preserves the physical orientation).
                q_sym = (rot_raw * sym).as_quat()
                for sign in (1.0, -1.0):
                    q_test = sign * q_sym
                    dot = np.dot(anchor_q, q_test)
                    if dot > best_dot:
                        best_dot = dot
                        best_q = q_test
            q_start = best_q
        else:
            q_start = q_start.copy()
            if q_start[3] < 0:
                q_start = -q_start

        unfolded[start_grain] = q_start
        visited.add(start_grain)

        while queue:
            current = queue.popleft()
            q_current = unfolded[current]

            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                q_raw = grain_quats[neighbor]
                rot_raw = R.from_quat(q_raw)

                best_q = None
                best_dot = -1.0

                for sym in HCP_SYMMETRIES:
                    # Right-multiply for crystal symmetry (g * S).
                    rot_sym = rot_raw * sym
                    q_sym = rot_sym.as_quat()
                    for sign in (1.0, -1.0):
                        q_test = sign * q_sym
                        dot = np.dot(q_current, q_test)
                        if dot > best_dot:
                            best_dot = dot
                            best_q = q_test

                unfolded[neighbor] = best_q
                visited.add(neighbor)
                queue.append(neighbor)

    # Handle isolated grains with no adjacency edges
    for gid in unique_grains:
        if gid not in unfolded:
            q = grain_quats[gid].copy()
            if q[3] < 0:
                q = -q
            unfolded[gid] = q

    return unfolded


def global_frame_center(
    grain_quats: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Shift all orientations so the population mean sits at identity q=(0,0,0,1).

    This prevents the stereographic projection from hitting the equator (qw=0)
    where sign flips cause discontinuities.

    Parameters
    ----------
    grain_quats : dict mapping grain_id -> (4,) quaternion [x,y,z,w]

    Returns
    -------
    centered_quats : dict mapping grain_id -> centered quaternion
    mean_q : (4,) the mean quaternion used for shifting (needed for decoding)
    """
    quat_list = np.array(list(grain_quats.values()))
    mean_q = average_quaternion(quat_list)

    mean_rot_inv = R.from_quat(mean_q).inv()

    centered = {}
    for gid, q in grain_quats.items():
        q_centered = (mean_rot_inv * R.from_quat(q)).as_quat()
        if q_centered[3] < 0:
            q_centered = -q_centered
        centered[gid] = q_centered

    return centered, mean_q
