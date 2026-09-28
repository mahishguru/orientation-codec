"""
HCP crystal symmetry operations.

The hexagonal close-packed crystal system has 12 proper rotations (point group 6/mmm
restricted to proper rotations = group 622, order 12). These are:
  - 6 rotations about the c-axis (z): 0, 60, 120, 180, 240, 300 degrees
  - 6 two-fold rotations: each c-axis rotation composed with a 180-degree flip about x
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_hcp_symmetries() -> list[R]:
    """Return the 12 proper rotation operators for HCP (D6 / 622)."""
    syms = []
    for i in range(6):
        theta = i * (np.pi / 3.0)
        Rz = R.from_euler('z', theta).as_matrix()
        syms.append(Rz)
        Rx_pi = R.from_euler('x', np.pi).as_matrix()
        syms.append(Rz @ Rx_pi)
    return [R.from_matrix(m) for m in syms]


HCP_SYMMETRIES = get_hcp_symmetries()


def fold_to_fundamental_zone(quat: np.ndarray) -> np.ndarray:
    """
    Find the symmetry-equivalent quaternion closest to identity (highest qw).

    This places the orientation into the standard fundamental zone that
    Dream3D expects for HCP crystals.

    Parameters
    ----------
    quat : (4,) array, scipy convention [x, y, z, w]

    Returns
    -------
    (4,) array in fundamental zone, qw >= 0
    """
    best_q = None
    max_qw = -1.0
    rot_raw = R.from_quat(quat)

    for sym in HCP_SYMMETRIES:
        # Crystal symmetry acts on the right for an active crystal->sample
        # orientation (g * S). Left-multiplication (S * g) would rotate the
        # crystal in the sample frame and scramble the texture.
        rot_sym = rot_raw * sym
        q_sym = rot_sym.as_quat()
        if q_sym[3] < 0:
            q_sym = -q_sym
        if q_sym[3] > max_qw:
            max_qw = q_sym[3]
            best_q = q_sym

    return best_q
