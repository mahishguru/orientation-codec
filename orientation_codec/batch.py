from __future__ import annotations
"""
Batch processing: recursively find and convert all Dream3D files.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from orientation_codec.encoder import encode_dream3d

# Default number of parallel worker processes
DEFAULT_WORKERS = 8


def _out_dir_for(fpath: Path, root_dir: Path, output_dir, flat_output: bool) -> Path:
    """Resolve the output directory for a given input file."""
    if output_dir is None:
        return fpath.parent
    if flat_output:
        return Path(output_dir)
    rel = fpath.parent.relative_to(root_dir)
    return Path(output_dir) / rel


def _is_done(fpath: Path, out_dir: Path) -> bool:
    """Return True if all three output files already exist for this source."""
    stem = fpath.stem
    return (
        (out_dir / f"{stem}.tiff").exists()
        and (out_dir / f"{stem}_meta.json").exists()
        and (out_dir / f"{stem}_labels.npy").exists()
    )


def _process_one(args):
    """Worker function: encode a single file. Returns (tiff_path | None, error_str | None)."""
    import io, sys
    fpath_str, out_dir_str = args
    fpath = Path(fpath_str)
    out_dir = Path(out_dir_str)
    # Suppress all stdout/stderr from the encoder so only the tqdm bar is visible
    _devnull = io.StringIO()
    try:
        sys.stdout = _devnull
        sys.stderr = _devnull
        tiff_path = encode_dream3d(fpath, out_dir)
        return str(tiff_path), None
    except Exception as e:
        return None, f"{fpath.name}: {e}"
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def batch_encode(
    root_dir: str | Path,
    output_dir: str | Path = None,
    file_types: tuple[str, ...] = (".dream3d",),
    flat_output: bool = False,
    workers: int = DEFAULT_WORKERS,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Recursively find all Dream3D files and encode them to TIFF.

    Parameters
    ----------
    root_dir : directory to search recursively
    output_dir : base output directory. Default (None) saves outputs next to each
                 .dream3d file where it was found. If flat_output=True, all outputs
                 go directly into output_dir. If flat_output=False, the input
                 directory structure is mirrored under output_dir.
    file_types : tuple of extensions to search for
    flat_output : if True, all outputs go into output_dir without subdirectories
    workers : number of parallel worker processes (default: 8)
    skip_existing : if True, skip files whose three output files already exist

    Returns
    -------
    List of paths to generated TIFF files
    """
    root_dir = Path(root_dir)

    files = []
    for ext in file_types:
        files.extend(sorted(root_dir.rglob(f"*{ext}")))

    if not files:
        print(f"No files found with extensions {file_types} under {root_dir}")
        return []

    print(f"Found {len(files)} files total")

    # Build work list, filtering out already-processed files
    work = []
    skipped = 0
    for fpath in files:
        out_dir = _out_dir_for(fpath, root_dir, output_dir, flat_output)
        if skip_existing and _is_done(fpath, out_dir):
            skipped += 1
        else:
            work.append((str(fpath), str(out_dir)))

    if skipped:
        print(f"Skipping {skipped} already-processed files ({len(work)} remaining)")

    if not work:
        print("Nothing to do.")
        return []

    results = []
    errors = []
    effective_workers = min(workers, len(work))

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_process_one, item): item for item in work}
        with tqdm(total=len(work), unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                tiff_str, err = future.result()
                if tiff_str:
                    results.append(Path(tiff_str))
                else:
                    errors.append(err)
                pbar.update(1)
                if err:
                    pbar.write(f"  ERROR: {err}")

    print(f"\nBatch complete: {len(results)}/{len(work)} files encoded successfully")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    return results
