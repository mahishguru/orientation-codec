#!/usr/bin/env python3
"""
Round-trip fidelity of the global-mean workflow on real Dream3D RVEs.

For each input file: Dream3D -> {8-bit PNG, 16-bit TIFF} -> pixel-wise decode,
then report per-grain HCP disorientation, grain-boundary F1 and grain-size
Pearson r of the self-segmented label map. This is the protocol behind the
codec round-trip table of the papers (Acta Materialia 2026, Sec. 4.4;
NeurIPS 2026, App. C).

Example
-------
    python scripts/verify_roundtrip.py \
        /path/to/rve/AZ31_extruded/7345/AZ31_extruded_7345.dream3d \
        --class-means data/class_means.json --class-name AZ31_extruded \
        --out roundtrip_results.json
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from orientation_codec import (
    boundary_f1,
    decode_pixelwise,
    encode_dream3d_global,
    grain_disorientations,
    load_16bit_tiff,
    load_class_means,
    load_png,
    read_dream3d,
    segment_grains,
)


def _stats(x: np.ndarray) -> dict:
    return {
        "mean_deg": float(x.mean()),
        "median_deg": float(np.median(x)),
        "max_deg": float(x.max()),
        "p95_deg": float(np.percentile(x, 95)),
    }


def _grain_size_pearson(true_labels: np.ndarray, rec_labels: np.ndarray) -> float:
    """Pearson r between each true grain's area and the area of its recovered segment."""
    rec_area = np.bincount(rec_labels.ravel())
    gids, first, true_area = np.unique(true_labels.ravel(), return_index=True, return_counts=True)
    keep = gids != 0
    rec = rec_area[rec_labels.ravel()[first[keep]]]
    return float(np.corrcoef(true_area[keep], rec)[0, 1])


def verify_file(path: Path, mean_q: np.ndarray) -> dict:
    orig = read_dream3d(path)
    labels, euler = orig["label_map"], orig["euler_angles"]
    result = {"file": path.name, "n_grains": int(len(np.unique(labels[labels > 0])))}

    with tempfile.TemporaryDirectory() as tmp:
        for fmt, loader in (("png", load_png), ("tiff", load_16bit_tiff)):
            img = loader(encode_dream3d_global(path, mean_q, tmp, name=f"rt_{fmt}", fmt=fmt))
            decoded = decode_pixelwise(img, mean_q)
            rec_labels = segment_grains(img)
            result[fmt] = {
                "disorientation": _stats(grain_disorientations(euler, decoded, labels)),
                "boundary_f1": boundary_f1(labels, rec_labels),
                "grain_size_pearson_r": _grain_size_pearson(labels, rec_labels),
                "n_segments": int(rec_labels.max()),
            }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", type=Path, help=".dream3d files")
    ap.add_argument("--class-means", type=Path, required=True, help="class_means.json")
    ap.add_argument("--class-name", required=True, help="key in class_means.json, e.g. AZ31_extruded")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    means = load_class_means(args.class_means)
    if args.class_name not in means:
        ap.error(f"class '{args.class_name}' not in {args.class_means}; available: {sorted(means)}")
    mean_q = means[args.class_name]

    results = []
    for path in args.inputs:
        r = verify_file(path, mean_q)
        results.append(r)
        print(f"{r['file']}  ({r['n_grains']} grains)")
        for fmt in ("png", "tiff"):
            d = r[fmt]["disorientation"]
            print(f"  {fmt:4s}  mean {d['mean_deg']:.4f} deg  max {d['max_deg']:.4f} deg  "
                  f"boundary F1 {r[fmt]['boundary_f1']:.4f}  "
                  f"grain-size r {r[fmt]['grain_size_pearson_r']:.4f}")

    if args.out:
        args.out.write_text(json.dumps({"class": args.class_name, "results": results}, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
