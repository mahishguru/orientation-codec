#!/usr/bin/env python3
"""
Self-contained demo: synthetic Voronoi RVE -> PNG -> DAMASK-ready Dream3D.

No external data is needed. Run from the repository root:

    python examples/01_synthetic_roundtrip.py
"""

from pathlib import Path

import numpy as np

from orientation_codec import (
    decode_image_pixelwise,
    encode_dream3d_global,
    grain_disorientations,
    read_dream3d,
)
from orientation_codec.synthetic import write_voronoi_dream3d

out = Path("example_output")
out.mkdir(exist_ok=True)

# 1. A 200x200 RVE with 150 grains and a ~25 deg fibre-like texture spread
src = write_voronoi_dream3d(out / "voronoi.dream3d", size=200, n_grains=150, seed=0)

# 2. Encode to an 8-bit PNG. For a real dataset, use the per-class mean from
#    data/class_means.json; here the texture is centred on the identity.
mean_q = np.array([0.0, 0.0, 0.0, 1.0])
png = encode_dream3d_global(src, mean_q, out, name="voronoi_encoded", fmt="png")

# 3. Decode the PNG back to a Dream3D file. Grain labels are recovered from the
#    image itself; no metadata besides mean_q is needed.
dec = decode_image_pixelwise(png, mean_q, out / "voronoi_decoded.dream3d")

# 4. Compare
orig, rec = read_dream3d(src), read_dream3d(dec)
mis = grain_disorientations(orig["euler_angles"], rec["euler_angles"], orig["label_map"])
print(f"encoded image : {png}")
print(f"decoded RVE   : {dec}")
print(f"grains        : {len(mis)} original, {rec['label_map'].max()} recovered")
print(f"disorientation: mean {mis.mean():.3f} deg, max {mis.max():.3f} deg")
