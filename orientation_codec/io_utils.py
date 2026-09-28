from __future__ import annotations
"""
I/O utilities for Dream3D (.dream3d), TIFF, and metadata files.
"""

import json
from pathlib import Path

import h5py
import numpy as np


# ============================================================
# Dream3D (.dream3d) reader
# ============================================================

_CELL_DATA = "DataContainers/SyntheticVolumeDataContainer/CellData"


def read_dream3d(path: str | Path) -> dict:
    """
    Read a Dream3D HDF5 file and extract orientation + label data.

    Returns
    -------
    dict with keys:
        euler_angles : (H, W, 3) float32, Bunge Euler in radians
        label_map    : (H, W) int32, grain IDs (1-based)
        dimensions   : (3,) int array [X, Y, Z]
        spacing      : (3,) float array
        phases       : (H, W) int32, phase IDs (optional, if present)
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        cell_data_path = _find_cell_data_group(f)

        eulers = f[f"{cell_data_path}/EulerAngles"][:]  # (Z, Y, X, 3)
        fids = f[f"{cell_data_path}/FeatureIds"][:]     # (Z, Y, X, 1)

        # Squeeze Z dimension (2D slices have Z=1)
        eulers = np.squeeze(eulers, axis=0)    # (Y, X, 3)
        fids = np.squeeze(fids, axis=0)        # (Y, X, 1)
        fids = np.squeeze(fids, axis=-1)       # (Y, X)

        # Read geometry
        geom_path = cell_data_path.rsplit("/", 1)[0] + "/_SIMPL_GEOMETRY"
        dims = f[f"{geom_path}/DIMENSIONS"][:]
        spacing = f[f"{geom_path}/SPACING"][:]

        # Read phases if available
        phases = None
        phases_key = f"{cell_data_path}/Phases"
        if phases_key in f:
            phases = f[phases_key][:]
            phases = np.squeeze(phases, axis=0)
            phases = np.squeeze(phases, axis=-1)

        # Read crystal structures if available
        crystal_structures = None
        ensemble_paths = [
            cell_data_path.rsplit("/", 1)[0] + "/CellEnsembleData/CrystalStructures",
        ]
        for ep in ensemble_paths:
            if ep in f:
                crystal_structures = f[ep][:]
                break

    result = {
        "euler_angles": eulers.astype(np.float32),
        "label_map": fids.astype(np.int32),
        "dimensions": dims,
        "spacing": spacing,
    }
    if phases is not None:
        result["phases"] = phases.astype(np.int32)
    if crystal_structures is not None:
        result["crystal_structures"] = crystal_structures
    return result


def _find_cell_data_group(f: h5py.File) -> str:
    """Search for the CellData group inside the HDF5 file."""
    if _CELL_DATA in f:
        return _CELL_DATA

    result = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Group) and name.endswith("CellData"):
            result.append(name)

    f.visititems(_visit)
    if result:
        return result[0]
    raise KeyError("Could not find CellData group in Dream3D file")


# ============================================================
# Dream3D (.dream3d) writer
# ============================================================

def write_dream3d(
    path: str | Path,
    euler_angles: np.ndarray,
    label_map: np.ndarray,
    spacing: np.ndarray = None,
    phases: np.ndarray = None,
    crystal_structures: np.ndarray = None,
    phase_name: str = "AZ31",
):
    """
    Write orientation data to a Dream3D-compatible HDF5 file.

    Creates the standard Dream3D file structure that can be opened
    directly in Dream3D / SIMPL / ParaView **and** read by
    ``damask.ConfigMaterial().load_DREAM3D(grain_data="Grain Data", ...)``.

    Parameters
    ----------
    path : output .dream3d file path
    euler_angles : (H, W, 3) Bunge Euler angles in radians, float32
    label_map : (H, W) int32 grain IDs (1-based)
    spacing : (3,) voxel spacing [X, Y, Z], default [1, 1, 1]
    phases : (H, W) int32 phase IDs, default all 1
    crystal_structures : (N_phases, 1) uint32, default [[999], [0]] (unknown + hexagonal)
    phase_name : phase name written into CellEnsembleData/PhaseName
    """
    path = Path(path)
    H, W = label_map.shape
    Z = 1

    if spacing is None:
        spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)

    if phases is None:
        phases = np.ones((H, W), dtype=np.int32)

    if crystal_structures is None:
        # Default: index 0 = unknown (999), index 1 = hexagonal (0)
        crystal_structures = np.array([[999], [0]], dtype=np.uint32)

    # ---- Per-grain data from label_map + euler_angles ----
    grain_ids = np.unique(label_map)
    grain_ids = grain_ids[grain_ids > 0]  # skip 0 if present
    n_grains = len(grain_ids)

    # Row 0 = dummy (grain IDs are 1-based); rows 1..n_grains = real grains
    gd_eulers = np.zeros((n_grains + 1, 3), dtype=np.float32)
    gd_phases = np.zeros((n_grains + 1, 1), dtype=np.int32)
    gd_volumes = np.zeros((n_grains + 1, 1), dtype=np.float32)

    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    for i, gid in enumerate(grain_ids):
        mask = label_map == gid
        gd_eulers[i + 1] = np.median(euler_angles[mask], axis=0)
        gd_phases[i + 1] = 1  # single-phase
        gd_volumes[i + 1] = float(np.count_nonzero(mask)) * voxel_vol

    # Reshape to Dream3D convention: (Z, Y, X, components)
    euler_4d = euler_angles.reshape(Z, H, W, 3).astype(np.float32)
    fids_4d = label_map.reshape(Z, H, W, 1).astype(np.int32)
    phases_4d = phases.reshape(Z, H, W, 1).astype(np.int32)

    dc = "DataContainers/SyntheticVolumeDataContainer"

    with h5py.File(str(path), "w") as f:
        # File-level attributes
        f.attrs["FileVersion"] = "7.0"
        f.attrs["DREAM3D Version"] = "orientation_codec_v0.2"

        # ---- Geometry ----
        geom = f.create_group(f"{dc}/_SIMPL_GEOMETRY")
        geom.create_dataset("DIMENSIONS", data=np.array([W, H, Z], dtype=np.int64))
        geom.create_dataset("ORIGIN", data=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        geom.create_dataset("SPACING", data=spacing)
        geom.attrs["GeometryType"] = np.uint32(0)
        geom.attrs["GeometryTypeName"] = "ImageGeometry"
        geom.attrs["GeometryName"] = "ImageGeometry"
        geom.attrs["SpatialDimensionality"] = np.uint32(3)
        geom.attrs["UnitDimensionality"] = np.uint32(3)

        # ---- CellData (per-voxel) ----
        cell = f.create_group(f"{dc}/CellData")
        cell.attrs["AttributeMatrixType"] = np.array([3], dtype=np.uint32)
        cell.attrs["TupleDimensions"] = np.array([Z, H, W], dtype=np.uint64)

        _write_ds(cell, "EulerAngles", euler_4d, comp=3,
                  dtype_str="DataArray<float>", tuple_dims=[Z, H, W])
        _write_ds(cell, "FeatureIds", fids_4d, comp=1,
                  dtype_str="DataArray<int32_t>", tuple_dims=[Z, H, W])
        _write_ds(cell, "Phases", phases_4d, comp=1,
                  dtype_str="DataArray<int32_t>", tuple_dims=[Z, H, W])

        # ---- Grain Data (per-grain) — required by DAMASK ----
        gd = f.create_group(f"{dc}/Grain Data")
        gd.attrs["AttributeMatrixType"] = np.array([7], dtype=np.uint32)
        gd.attrs["TupleDimensions"] = np.array([n_grains + 1], dtype=np.uint64)

        _write_ds(gd, "EulerAngles", gd_eulers, comp=3,
                  dtype_str="DataArray<float>", tuple_dims=[n_grains + 1])
        _write_ds(gd, "Phases", gd_phases, comp=1,
                  dtype_str="DataArray<int32_t>", tuple_dims=[n_grains + 1])
        _write_ds(gd, "Volumes", gd_volumes, comp=1,
                  dtype_str="DataArray<float>", tuple_dims=[n_grains + 1])

        # ---- CellEnsembleData ----
        ensemble = f.create_group(f"{dc}/CellEnsembleData")
        n_phases = len(crystal_structures)
        ensemble.attrs["AttributeMatrixType"] = np.array([11], dtype=np.uint32)
        ensemble.attrs["TupleDimensions"] = np.array([n_phases], dtype=np.uint64)

        _write_ds(ensemble, "CrystalStructures", crystal_structures, comp=1,
                  dtype_str="DataArray<uint32_t>", tuple_dims=[n_phases])

        # NumFeatures: [0, n_grains] (index 0 = unknown phase with 0 grains)
        num_features = np.zeros((n_phases, 1), dtype=np.int32)
        if n_phases > 1:
            num_features[1, 0] = n_grains
        _write_ds(ensemble, "NumFeatures", num_features, comp=1,
                  dtype_str="DataArray<int32_t>", tuple_dims=[n_phases])

        # PhaseName — stored as variable-length strings (h5py special_dtype)
        phase_names = ["Unknown Phase Type", phase_name][:n_phases]
        dt = h5py.special_dtype(vlen=str)
        ds_pn = ensemble.create_dataset("PhaseName", data=phase_names, dtype=dt)
        ds_pn.attrs["ComponentDimensions"] = np.array([1], dtype=np.uint64)
        ds_pn.attrs["ObjectType"] = "StringDataArray"
        ds_pn.attrs["TupleDimensions"] = np.array([n_phases], dtype=np.uint64)

        # ---- DataContainerBundles (empty, but Dream3D expects it) ----
        f.create_group("DataContainerBundles")


def _write_ds(grp: h5py.Group, name: str, data: np.ndarray,
              comp: int, dtype_str: str, tuple_dims: list) -> None:
    """Create a dataset with standard Dream3D-style attributes."""
    ds = grp.create_dataset(name, data=data)
    ds.attrs["ComponentDimensions"] = np.array([comp], dtype=np.uint64)
    ds.attrs["ObjectType"] = dtype_str
    ds.attrs["TupleDimensions"] = np.array(tuple_dims, dtype=np.uint64)
    ds.attrs["DataArrayVersion"] = np.array([2], dtype=np.int32)


def copy_dream3d_with_new_eulers(
    source_path: str | Path,
    output_path: str | Path,
    new_euler_angles: np.ndarray,
):
    """
    Copy a Dream3D file but replace the EulerAngles dataset.

    This preserves all original metadata, pipeline info, grain statistics,
    etc. while only swapping the orientation data.

    Parameters
    ----------
    source_path : original .dream3d file
    output_path : output .dream3d file
    new_euler_angles : (H, W, 3) Bunge Euler angles in radians
    """
    import shutil

    source_path = Path(source_path)
    output_path = Path(output_path)
    shutil.copy2(source_path, output_path)

    with h5py.File(str(output_path), "r+") as f:
        cell_data_path = _find_cell_data_group(f)
        euler_key = f"{cell_data_path}/EulerAngles"

        H, W = new_euler_angles.shape[:2]
        euler_4d = new_euler_angles.reshape(1, H, W, 3).astype(np.float32)

        del f[euler_key]
        ds = f.create_dataset(euler_key, data=euler_4d)
        ds.attrs["ComponentDimensions"] = np.array([3], dtype=np.uint64)
        ds.attrs["ObjectType"] = "DataArray<float>"
        ds.attrs["TupleDimensions"] = np.array([1, H, W], dtype=np.uint64)


# ============================================================
# XDMF writer (companion for Dream3D HDF5)
# ============================================================

def write_xdmf(
    path: str | Path,
    h5_filename: str,
    dimensions: np.ndarray,
    spacing: np.ndarray,
):
    """
    Write an XDMF file pointing to a Dream3D HDF5 file for ParaView.

    Parameters
    ----------
    path : output .xdmf file path
    h5_filename : name of the companion HDF5 file (just filename, not full path)
    dimensions : (3,) [X, Y, Z] cell counts
    spacing : (3,) voxel spacing
    """
    X, Y, Z = int(dimensions[0]), int(dimensions[1]), int(dimensions[2])
    dc = "DataContainers/SyntheticVolumeDataContainer"

    xdmf = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="SyntheticVolumeDataContainer" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{Z+1} {Y+1} {X+1}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
          0.0 0.0 0.0
        </DataItem>
        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
          {spacing[2]} {spacing[1]} {spacing[0]}
        </DataItem>
      </Geometry>
      <Attribute Name="EulerAngles" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="{Z} {Y} {X} 3" NumberType="Float" Precision="4" Format="HDF">
          {h5_filename}:/{dc}/CellData/EulerAngles
        </DataItem>
      </Attribute>
      <Attribute Name="FeatureIds" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{Z} {Y} {X} 1" NumberType="Int" Format="HDF">
          {h5_filename}:/{dc}/CellData/FeatureIds
        </DataItem>
      </Attribute>
      <Attribute Name="Phases" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{Z} {Y} {X} 1" NumberType="Int" Format="HDF">
          {h5_filename}:/{dc}/CellData/Phases
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    Path(path).write_text(xdmf)


# ============================================================
# Metadata (JSON) helpers
# ============================================================

def save_metadata(
    path: str | Path,
    mean_q: np.ndarray,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    source_path: str = None,
):
    """Save encoding metadata needed for decoding."""
    meta = {
        "global_shift_quaternion": mean_q.tolist(),
        "spacing": spacing.tolist(),
        "dimensions": dimensions.tolist(),
        "note": "Apply inverse of global_shift_quaternion before folding to FZ.",
    }
    if source_path is not None:
        meta["source_dream3d"] = str(source_path)
    Path(path).write_text(json.dumps(meta, indent=2))


def load_metadata(path: str | Path) -> dict:
    """Load encoding metadata."""
    return json.loads(Path(path).read_text())


# ============================================================
# 16-bit TIFF I/O
# ============================================================

def save_16bit_tiff(path: str | Path, img: np.ndarray):
    """Save a 16-bit 3-channel image as TIFF."""
    import tifffile
    tifffile.imwrite(str(path), img.astype(np.uint16))


def load_16bit_tiff(path: str | Path) -> np.ndarray:
    """Load a 16-bit TIFF image."""
    import tifffile
    return tifffile.imread(str(path))


# ============================================================
# 8-bit PNG I/O  (for use with pretrained vision models)
# ============================================================

def save_png(path: str | Path, img: np.ndarray):
    """
    Save a 3-channel float/uint16 image as an 8-bit RGB PNG.

    Accepts either:
      - uint16 (H, W, 3) — values 0-65535, rescaled to 0-255
      - float  (H, W, 3) — values in [0, 1], scaled to 0-255
    """
    from PIL import Image
    if img.dtype == np.uint16:
        img8 = (img.astype(np.float32) * (255.0 / 65535.0)).round().astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        img8 = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    Image.fromarray(img8, mode="RGB").save(str(path))


def load_png(path: str | Path) -> np.ndarray:
    """Load an 8-bit RGB PNG.  Returns (H, W, 3) uint8."""
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
