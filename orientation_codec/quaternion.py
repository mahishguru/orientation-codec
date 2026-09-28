"""
Quaternion math and stereographic projection utilities.

Stereographic projection maps a unit quaternion q = [x, y, z, w] (with w >= 0)
to a 3D vector S = [x, y, z] / (1 + w). This mapping:
  - Is smooth and continuous (no discontinuities)
  - Maps to the range approximately [-1, 1]
  - Degrades gracefully under noise (no square-root domain errors)
  - Has a clean inverse via rational functions (no square roots needed)
"""

import numpy as np


def quat_ensure_positive_w(q: np.ndarray) -> np.ndarray:
    """Flip quaternion to the w >= 0 hemisphere."""
    if q[3] < 0:
        return -q
    return q.copy()


def stereographic_forward(q: np.ndarray) -> np.ndarray:
    """
    Stereographic projection: unit quaternion -> 3D vector.

    Parameters
    ----------
    q : (4,) array, scipy convention [x, y, z, w], must have w >= 0

    Returns
    -------
    S : (3,) array, values approximately in [-1, 1]
    """
    w = q[3]
    return q[:3] / (1.0 + w)


def stereographic_inverse(S: np.ndarray) -> np.ndarray:
    """
    Inverse stereographic projection: 3D vector -> unit quaternion.

    Uses rational reconstruction (no square roots):
      S2 = Sx^2 + Sy^2 + Sz^2
      qw = (1 - S2) / (1 + S2)
      qxyz = 2 * S / (1 + S2)

    Parameters
    ----------
    S : (3,) array

    Returns
    -------
    q : (4,) array [x, y, z, w], unit quaternion with w >= 0
    """
    S2 = np.dot(S, S)
    denom = 1.0 + S2
    qw = (1.0 - S2) / denom
    q_xyz = 2.0 * S / denom
    q = np.array([q_xyz[0], q_xyz[1], q_xyz[2], qw])
    # Normalize to guard against accumulated float error
    q /= np.linalg.norm(q)
    return quat_ensure_positive_w(q)


def average_quaternion(quaternions: np.ndarray) -> np.ndarray:
    """
    Compute the mean quaternion via eigenvalue decomposition.

    Parameters
    ----------
    quaternions : (N, 4) array, scipy convention [x, y, z, w]

    Returns
    -------
    (4,) mean quaternion
    """
    Q = np.asarray(quaternions)
    A = Q.T @ Q / len(Q)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvectors[:, np.argmax(eigenvalues)]
