"""Download the two public datasets from their official sources.

The repository never stores the data. This script fetches it, records a
manifest (URL, licence, checksum, shapes and library versions) and leaves the
files under ``data/`` -- ignored by git.

Examples
--------
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --force --size 64
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uq import __version__  # noqa: E402
from uq.data import credit, pneumonia  # noqa: E402
from uq.settings import data_dir, ensure_dir  # noqa: E402

PACKAGES = ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "medmnist")


def _versions() -> dict[str, str]:
    installed = {}
    for package in PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - defensive
            installed[package] = "not installed"
    return installed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=data_dir(),
        help="where the datasets are stored (default: %(default)s)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=pneumonia.DEFAULT_SIZE,
        choices=pneumonia.SUPPORTED_SIZES,
        help="MedMNIST image size (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if the files are already present",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = ensure_dir(args.data_dir)

    print(f"[credit] downloading from {credit.SOURCE_URL}")
    frame = credit.load(root, force_download=args.force)
    print(f"[credit] {len(frame):,} rows x {frame.shape[1] - 1} features")

    print(f"[pneumonia] downloading PneumoniaMNIST ({args.size}x{args.size})")
    shapes = pneumonia.download(root, size=args.size)
    for split, shape in shapes.items():
        print(f"[pneumonia] {split:>5}: {shape[0]:,} images {tuple(shape[1:])}")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "project_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _versions(),
        "data_dir": str(root),
        "datasets": {
            "credit": {
                "name": credit.DATASET_NAME,
                "source": credit.SOURCE_URL,
                "doi": credit.DOI,
                "licence": credit.LICENCE,
                "archive_sha256": credit.sha256(credit.dataset_dir(root) / credit.ARCHIVE_NAME),
                "n_rows": len(frame),
                "n_features": int(frame.shape[1] - 1),
                "target": credit.TARGET,
                "positive_rate": round(float(frame[credit.TARGET].mean()), 6),
            },
            "pneumonia": {
                "name": pneumonia.DATASET_NAME,
                "source": "https://medmnist.com/",
                "doi": "10.1038/s41597-022-01721-8",
                "licence": "CC BY 4.0",
                "size": args.size,
                "splits": {split: int(shape[0]) for split, shape in shapes.items()},
                "image_shape": list(next(iter(shapes.values()))[1:]),
            },
        },
    }

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"[done] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
