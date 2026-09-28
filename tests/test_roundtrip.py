"""
End-to-end round-trip tests on synthetic Voronoi microstructures.

These tests need no external data: a Voronoi RVE is generated, written as
.dream3d, encoded, decoded, and compared with symmetry-aware metrics.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from orientation_codec import (
    boundary_f1,
    decode_image_pixelwise,
    decode_pixelwise,
    decode_rgb_to_dream3d,
    encode_dream3d,
    encode_dream3d_global,
    grain_disorientations,
    hcp_disorientation_deg,
    load_16bit_tiff,
    load_png,
    read_dream3d,
    segment_grains,
)
from orientation_codec.symmetry import HCP_SYMMETRIES, fold_to_fundamental_zone
from orientation_codec.synthetic import write_voronoi_dream3d


@pytest.fixture(params=[25.0, None], ids=["fibre", "random"])
def rve(tmp_path, request):
    path = write_voronoi_dream3d(
        tmp_path / "voronoi.dream3d", size=96, n_grains=40,
        texture_spread_deg=request.param, seed=1,
    )
    return path, read_dream3d(path)


def _class_mean(data):
    """Stand-in for class_means.json: the mean of the sample's own orientations."""
    from orientation_codec.quaternion import average_quaternion, quat_ensure_positive_w
    q = R.from_euler("ZXZ", data["euler_angles"].reshape(-1, 3)).as_quat()
    q = np.array([fold_to_fundamental_zone(x) for x in q[::97]])
    return quat_ensure_positive_w(average_quaternion(q))


def test_symmetry_group_is_closed():
    mats = [s.as_matrix() for s in HCP_SYMMETRIES]
    assert len(mats) == 12
    for a in mats:
        for b in mats:
            assert any(np.allclose(a @ b, c, atol=1e-9) for c in mats)


def test_disorientation_is_symmetry_invariant():
    rng = np.random.default_rng(0)
    g = R.random(50, random_state=1)
    euler = g.as_euler("ZXZ")
    for sym in HCP_SYMMETRIES:
        e_equiv = (g * sym).as_euler("ZXZ")
        assert np.all(hcp_disorientation_deg(euler, e_equiv) < 1e-4)
    # disorientation is bounded by the HCP maximum (~93.8 deg)
    other = R.random(50, random_state=2).as_euler("ZXZ")
    assert np.all(hcp_disorientation_deg(euler, other) <= 93.9)


def test_per_sample_tiff_roundtrip(rve, tmp_path):
    path, orig = rve
    tiff = encode_dream3d(path, tmp_path / "enc")
    decoded = read_dream3d(decode_rgb_to_dream3d(tiff))
    mis = grain_disorientations(orig["euler_angles"], decoded["euler_angles"], orig["label_map"])
    assert mis.max() < 0.01


@pytest.mark.parametrize("fmt,max_mean,max_max", [("tiff", 0.01, 0.02), ("png", 1.0, 2.0)])
def test_global_mean_pixelwise_roundtrip(rve, tmp_path, fmt, max_mean, max_max):
    path, orig = rve
    mean_q = _class_mean(orig)
    img_path = encode_dream3d_global(path, mean_q, tmp_path / "enc", fmt=fmt)
    img = load_png(img_path) if fmt == "png" else load_16bit_tiff(img_path)
    euler = decode_pixelwise(img, mean_q)
    mis = grain_disorientations(orig["euler_angles"], euler, orig["label_map"])
    assert mis.mean() < max_mean
    assert mis.max() < max_max


def test_self_segmentation_recovers_grains(rve, tmp_path):
    """
    8-bit self-segmentation never splits a grain, and only merges neighbours
    across low-angle boundaries (disorientation below the 5 deg Read-Shockley
    threshold), which are physically the same grain.
    """
    from orientation_codec.unfolding import build_grain_adjacency

    path, orig = rve
    true_labels, euler = orig["label_map"], orig["euler_angles"]
    mean_q = _class_mean(orig)
    img = load_png(encode_dream3d_global(path, mean_q, tmp_path / "enc", fmt="png"))
    labels = segment_grains(img)

    # No grain is split: each 4-connected piece of a true grain maps to exactly
    # one recovered segment (pixelated Voronoi cells can have diagonal-only pixels)
    from scipy.ndimage import label as ndlabel
    for gid in np.unique(true_labels):
        pieces, n = ndlabel(true_labels == gid)
        for k in range(1, n + 1):
            assert len(np.unique(labels[pieces == k])) == 1

    # Any two adjacent grains that were merged are separated by a low-angle boundary
    seg_of = {g: labels[true_labels == g][0] for g in np.unique(true_labels)}
    first = {g: euler[true_labels == g][0] for g in np.unique(true_labels)}
    for a, nbrs in build_grain_adjacency(true_labels).items():
        for b in nbrs:
            if a < b and seg_of[a] == seg_of[b]:
                assert hcp_disorientation_deg(first[a], first[b]) < 5.0

    # Boundaries that survive are the true ones
    assert boundary_f1(true_labels, labels) > 0.9


def test_decoded_dream3d_has_damask_groups(rve, tmp_path):
    import h5py
    path, orig = rve
    mean_q = _class_mean(orig)
    png = encode_dream3d_global(path, mean_q, tmp_path / "enc", fmt="png")
    out = decode_image_pixelwise(png, mean_q, tmp_path / "dec.dream3d", phase_name="Mg5Gd")
    dc = "DataContainers/SyntheticVolumeDataContainer"
    with h5py.File(out, "r") as f:
        for key in ("CellData/EulerAngles", "CellData/FeatureIds", "CellData/Phases",
                    "Grain Data/EulerAngles", "Grain Data/Phases", "Grain Data/Volumes",
                    "CellEnsembleData/CrystalStructures", "CellEnsembleData/PhaseName",
                    "_SIMPL_GEOMETRY/DIMENSIONS"):
            assert f"{dc}/{key}" in f, key
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f[f"{dc}/CellEnsembleData/PhaseName"][:]]
        assert names[1] == "Mg5Gd"
    assert out.with_suffix(".xdmf").exists()
