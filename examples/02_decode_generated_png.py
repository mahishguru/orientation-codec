#!/usr/bin/env python3
"""
Decode an encoded/generated orientation PNG into a DAMASK-ready Dream3D RVE.

Uses the bundled per-class mean quaternions (data/class_means.json) and one of
the example images in data/examples/. Run from the repository root:

    python examples/02_decode_generated_png.py
    python examples/02_decode_generated_png.py path/to/generated.png --class-name Mg-5Gd_extruded
"""

import argparse
from pathlib import Path

import numpy as np

from orientation_codec import decode_image_pixelwise, load_class_means, read_dream3d

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", type=Path,
                default=Path("data/examples/AZ31_extruded_orientation_1000.png"))
ap.add_argument("--class-name", default="AZ31_extruded")
ap.add_argument("--class-means", type=Path, default=Path("data/class_means.json"))
ap.add_argument("--phase-name", default="Mg")
ap.add_argument("--spacing", type=float, default=1.0, help="voxel size (um)")
args = ap.parse_args()

means = load_class_means(args.class_means)
out = Path("example_output") / f"{args.image.stem}.dream3d"
out.parent.mkdir(exist_ok=True)

dream3d = decode_image_pixelwise(
    args.image,
    means[args.class_name],
    out,
    spacing=np.full(3, args.spacing, dtype=np.float32),
    phase_name=args.phase_name,
)
rve = read_dream3d(dream3d)
print(f"wrote {dream3d} ({rve['label_map'].max()} grains, {rve['label_map'].shape} voxels)")
print("load in DAMASK with:")
print(f"  damask.ConfigMaterial.load_DREAM3D('{dream3d}', grain_data='Grain Data')")
