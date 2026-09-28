"""
Command-line interface for orientation_codec.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="orientation-codec",
        description="Encode crystallographic orientations from Dream3D files to neural-network-friendly RGB images and decode them back.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- encode ---
    enc = subparsers.add_parser("encode", help="Encode a single Dream3D file to 16-bit TIFF")
    enc.add_argument("input", type=str, help="Path to .dream3d file")
    enc.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory (default: same as input)")
    enc.add_argument("-n", "--name", type=str, default=None, help="Output base name (default: input stem)")

    # --- decode ---
    dec = subparsers.add_parser("decode", help="Decode a 16-bit TIFF back to Dream3D")
    dec.add_argument("input", type=str, help="Path to 16-bit .tiff file")
    dec.add_argument("-m", "--meta", type=str, default=None, help="Path to metadata JSON")
    dec.add_argument("-l", "--labels", type=str, default=None, help="Path to labels .npy file")
    dec.add_argument("-o", "--output", type=str, default=None, help="Output .dream3d path")

    # --- batch ---
    bat = subparsers.add_parser("batch", help="Recursively encode all Dream3D files in a directory")
    bat.add_argument("root_dir", type=str, help="Root directory to search")
    bat.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    bat.add_argument("--flat", action="store_true", help="Put all outputs in one flat directory")
    bat.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    bat.add_argument("--no-skip", action="store_true", help="Re-process files even if outputs already exist")

    # --- verify ---
    ver = subparsers.add_parser("verify", help="Verify round-trip accuracy for a Dream3D file")
    ver.add_argument("input", type=str, help="Path to .dream3d file")

    # --- compute-means ---
    cm = subparsers.add_parser("compute-means", help="Compute per-class global mean quaternions")
    cm.add_argument("root_dir", type=str, help="Root directory (children = alloy classes)")
    cm.add_argument("-o", "--output", type=str, default=None, help="Output JSON path (default: root_dir/class_means.json)")
    cm.add_argument("-s", "--samples", type=int, default=50, help="Samples per class (default: 50)")
    cm.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers (default: 8)")

    # --- batch-global ---
    bg = subparsers.add_parser("batch-global", help="Batch encode using per-class global mean quaternions")
    bg.add_argument("root_dir", type=str, help="Root directory (children = alloy classes)")
    bg.add_argument("class_means", type=str, help="Path to class_means.json")
    bg.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    bg.add_argument("--flat", action="store_true", help="Flatten output directory")
    bg.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers (default: 8)")
    bg.add_argument("--no-skip", action="store_true", help="Re-process existing files")
    bg.add_argument("--fmt", type=str, default="png", choices=["png", "tiff"],
                    help="Output image format: png (8-bit, default) or tiff (16-bit)")

    # --- decode-pixelwise ---
    dp = subparsers.add_parser("decode-pixelwise", help="Decode a PNG/TIFF using only a global mean quaternion (no labels needed)")
    dp.add_argument("input", type=str, help="Path to 8-bit .png or 16-bit .tiff file")
    dp.add_argument("class_means", type=str, help="Path to class_means.json")
    dp.add_argument("--class-name", type=str, required=True, help="Alloy class name (key in class_means.json)")
    dp.add_argument("-o", "--output", type=str, default=None, help="Output .dream3d path")

    args = parser.parse_args()

    if args.command == "encode":
        from orientation_codec.encoder import encode_dream3d
        encode_dream3d(args.input, args.output_dir, args.name)

    elif args.command == "decode":
        from orientation_codec.decoder import decode_rgb_to_dream3d
        decode_rgb_to_dream3d(args.input, args.meta, args.labels, args.output)

    elif args.command == "batch":
        from orientation_codec.batch import batch_encode
        batch_encode(
            args.root_dir,
            args.output_dir,
            (".dream3d",),
            args.flat,
            workers=args.workers,
            skip_existing=not args.no_skip,
        )

    elif args.command == "verify":
        from orientation_codec.verify import verify_round_trip
        verify_round_trip(args.input)

    elif args.command == "compute-means":
        from orientation_codec.dataset import compute_class_means
        compute_class_means(args.root_dir, args.output, args.samples, args.workers)

    elif args.command == "batch-global":
        from orientation_codec.dataset import batch_encode_global
        batch_encode_global(
            args.root_dir,
            args.class_means,
            args.output_dir,
            args.flat,
            workers=args.workers,
            skip_existing=not args.no_skip,
            fmt=args.fmt,
        )

    elif args.command == "decode-pixelwise":
        from orientation_codec.dataset import load_class_means, decode_image_pixelwise
        means = load_class_means(args.class_means)
        if args.class_name not in means:
            print(f"Error: class '{args.class_name}' not found in {args.class_means}")
            print(f"Available: {list(means.keys())}")
            sys.exit(1)
        decode_image_pixelwise(args.input, means[args.class_name], args.output)


if __name__ == "__main__":
    main()
