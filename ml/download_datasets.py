"""
Public dataset downloader for LoLvlance ML training.

Usage:
    python ml/download_datasets.py --datasets musan --output-root /data/lolvlance
    python ml/download_datasets.py --datasets musan fsd50k openmic --output-root /data/lolvlance
    python ml/download_datasets.py --list

Datasets:
    musan   - MUSAN (Music/Speech/Noise), ~11 GB, public domain (OpenSLR)
    fsd50k  - FSD50K (general audio events), ~24 GB, CC (Zenodo)
    openmic - OpenMIC-2018 (instrument recognition), ~13 GB, CC (Zenodo)

After downloading, train:
    python -m ml.train \\
      --audio-root /data/lolvlance/musan \\
      --musan-root /data/lolvlance/musan \\
      --fsd50k-root /data/lolvlance/fsd50k \\
      --openmic-root /data/lolvlance/openmic \\
      --rebuild-manifest --epochs 20 --export-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "musan": {
        "description": "MUSAN – Music, Speech, Noise (~11 GB, public domain)",
        "files": [
            {
                "url": "https://www.openslr.org/resources/17/musan.tar.gz",
                "filename": "musan.tar.gz",
                "md5": None,  # large file – skip md5 by default
                "extract": "tar",
                "subfolder": "musan",
            }
        ],
        "notes": (
            "Contains music, speech, and noise segments. "
            "Folder names are used to infer source labels (music/speech/noise). "
            "After extraction the root you pass to --musan-root should be the 'musan/' subfolder."
        ),
    },
    "fsd50k": {
        "description": "FSD50K – Freesound Dataset 50K (~24 GB, CC BY 4.0)",
        "files": [
            {
                "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01",
                "filename": "FSD50K.dev_audio.z01",
                "md5": None,
                "extract": None,
                "subfolder": None,
            },
            {
                "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02",
                "filename": "FSD50K.dev_audio.z02",
                "md5": None,
                "extract": None,
                "subfolder": None,
            },
            {
                "url": "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip",
                "filename": "FSD50K.dev_audio.zip",
                "md5": None,
                "extract": "zip_multipart",
                "subfolder": "fsd50k",
            },
            {
                "url": "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip",
                "filename": "FSD50K.ground_truth.zip",
                "md5": None,
                "extract": "zip",
                "subfolder": "fsd50k",
            },
            {
                "url": "https://zenodo.org/record/4060432/files/FSD50K.metadata.zip",
                "filename": "FSD50K.metadata.zip",
                "md5": None,
                "extract": "zip",
                "subfolder": "fsd50k",
            },
        ],
        "notes": (
            "Large multi-part zip. All parts must be in the same directory before extraction. "
            "This script downloads and extracts them automatically. "
            "Pass FSD50K.dev_audio/ as --fsd50k-root."
        ),
    },
    "openmic": {
        "description": "OpenMIC-2018 – instrument recognition (~13 GB, CC BY 4.0)",
        "files": [
            {
                "url": "https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz",
                "filename": "openmic-2018-v1.0.0.tgz",
                "md5": None,
                "extract": "tar",
                "subfolder": "openmic",
            }
        ],
        "notes": (
            "Contains 20,000 10-second music clips with instrument labels. "
            "Pass openmic-2018/ as --openmic-root."
        ),
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        print(f"\r  {block_count * block_size / 1_048_576:.1f} MB downloaded...", end="", flush=True)
        return
    downloaded = block_count * block_size
    pct = min(100.0, downloaded / total_size * 100)
    downloaded_mb = downloaded / 1_048_576
    total_mb = total_size / 1_048_576
    print(f"\r  {pct:5.1f}%  {downloaded_mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def _download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress


def _verify_md5(path: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive.name} ...")
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest_dir)
    print("  Done.")


def _extract_zip(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive.name} ...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    print("  Done.")


def _extract_zip_multipart(primary_zip: Path, dest_dir: Path) -> None:
    """
    Recombine multi-part zip (z01, z02, ..., zip) using the system zip tool,
    or fall back to direct Python extraction if the parts are already merged.
    """
    print(f"  Assembling multi-part zip from {primary_zip.parent} ...")
    out_dir = dest_dir

    # Try system 'zip -FF' or '7z' first (handles split archives properly)
    if shutil.which("7z"):
        cmd = ["7z", "x", str(primary_zip), f"-o{out_dir}", "-y"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  Done (7z).")
            return
        print(f"  7z failed: {result.stderr.strip()}")

    if shutil.which("zip"):
        merged = primary_zip.parent / "merged_dev.zip"
        fix_cmd = ["zip", "-FF", str(primary_zip), "--out", str(merged)]
        result = subprocess.run(fix_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            _extract_zip(merged, out_dir)
            merged.unlink(missing_ok=True)
            return
        print(f"  zip -FF failed: {result.stderr.strip()}")

    # Last resort: try Python zipfile directly (works if no split)
    try:
        _extract_zip(primary_zip, out_dir)
    except zipfile.BadZipFile as exc:
        print(
            f"\n  ERROR: Cannot extract multi-part zip automatically.\n"
            f"  Please install 7-Zip (7z) or the system 'zip' utility, then re-run.\n"
            f"  Original error: {exc}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-dataset downloader
# ---------------------------------------------------------------------------

def download_dataset(name: str, output_root: Path, keep_archives: bool) -> None:
    spec = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Dataset: {name.upper()}")
    print(f"  {spec['description']}")
    print(f"  Note: {spec['notes']}")
    print(f"{'='*60}")

    dataset_dir = output_root / name
    archive_dir = output_root / "_archives" / name
    archive_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for file_spec in spec["files"]:
        url: str = file_spec["url"]
        filename: str = file_spec["filename"]
        archive_path = archive_dir / filename
        extract_mode: str | None = file_spec["extract"]

        if not archive_path.exists():
            _download_file(url, archive_path)
        else:
            print(f"  Already downloaded: {filename}")

        if file_spec.get("md5") and archive_path.exists():
            if not _verify_md5(archive_path, file_spec["md5"]):
                print(f"  WARNING: MD5 mismatch for {filename}. File may be corrupt.")

        if extract_mode is None:
            continue  # part file — extraction triggered by primary zip entry

        if extract_mode == "tar":
            _extract_tar(archive_path, dataset_dir)
        elif extract_mode == "zip":
            _extract_zip(archive_path, dataset_dir)
        elif extract_mode == "zip_multipart":
            _extract_zip_multipart(archive_path, dataset_dir)

    if not keep_archives:
        print(f"  Removing archive directory {archive_dir} ...")
        shutil.rmtree(archive_dir, ignore_errors=True)

    print(f"\n  {name.upper()} ready at: {dataset_dir.resolve()}")


# ---------------------------------------------------------------------------
# Training hint
# ---------------------------------------------------------------------------

def print_training_hint(downloaded: list[str], output_root: Path) -> None:
    flags: list[str] = ["python -m ml.train \\", "  --rebuild-manifest \\", "  --epochs 20 \\", "  --export-onnx \\"]
    if "musan" in downloaded:
        flags.append(f"  --audio-root {(output_root / 'musan').resolve()} \\")
        flags.append(f"  --musan-root {(output_root / 'musan').resolve()} \\")
    if "fsd50k" in downloaded:
        flags.append(f"  --fsd50k-root {(output_root / 'fsd50k').resolve()} \\")
    if "openmic" in downloaded:
        flags.append(f"  --openmic-root {(output_root / 'openmic').resolve()} \\")

    # Remove trailing backslash from last flag
    if flags:
        flags[-1] = flags[-1].rstrip(" \\")

    print("\n" + "="*60)
    print("All downloads complete. To train:")
    print()
    print("\n".join(flags))
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public audio datasets for LoLvlance ML training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=None,
        metavar="NAME",
        help="Datasets to download. Choices: " + ", ".join(DATASETS.keys()),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory where datasets will be saved (default: data/datasets).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files after extraction.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, spec in DATASETS.items():
            print(f"  {name:12s}  {spec['description']}")
            print(f"              {spec['notes']}\n")
        return

    if not args.datasets:
        print("Specify at least one dataset with --datasets. Use --list to see options.")
        sys.exit(1)

    output_root: Path = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")

    downloaded: list[str] = []
    for name in args.datasets:
        download_dataset(name, output_root, keep_archives=args.keep_archives)
        downloaded.append(name)

    print_training_hint(downloaded, output_root)


if __name__ == "__main__":
    main()
