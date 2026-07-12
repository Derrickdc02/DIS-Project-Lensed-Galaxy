"""Preprocess raw PROBES FITS images for diffusion-prior training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from tqdm import tqdm


BAND = "g"
CROP_SIZE = 256
A = 5.5
LOWER = 0.0
MIN_FILESIZE = 5_000


def check_for_corruption(ar: np.ndarray) -> bool:
    """Return whether an image fails the PROBES data-quality checks.

    Images are rejected when they contain non-finite values or when more than
    30 percent of their pixels are exactly zero.
    """

    if not np.isfinite(ar).all():
        return True
    zero_fraction = np.count_nonzero(ar == 0) / ar.size
    return zero_fraction > 0.30


def center_crop(ar: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    """Crop a two-dimensional image about its centre.

    Parameters
    ----------
    ar
        Input image with shape (height, width).
    crop_size
        Requested even side length in pixels.

    Returns
    -------
    numpy.ndarray
        Centred square crop with shape (crop_size, crop_size).
    """

    if ar.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {ar.shape}")
    if crop_size <= 0 or crop_size % 2:
        raise ValueError(f"crop_size must be a positive even integer, got {crop_size}")

    height, width = ar.shape
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image too small for crop: got {ar.shape}, need at least "
            f"{crop_size}x{crop_size}"
        )

    centre_y, centre_x = height // 2, width // 2
    half = crop_size // 2
    return ar[centre_y - half : centre_y + half, centre_x - half : centre_x + half]


def normalize_probes(x: np.ndarray, A: float = A, lower: float = LOWER) -> np.ndarray:
    """Clip a PROBES image and map its flux values to [-1, 1].

    The transform follows the preprocessing used by Smith et al. and the
    strong-lensing experiments: 2 * clip(x, lower, A) / A - 1.
    """

    if A <= 0:
        raise ValueError(f"A must be positive, got {A}")
    if lower < 0 or lower >= A:
        raise ValueError(f"lower must satisfy 0 <= lower < A, got lower={lower}, A={A}")

    clipped = np.clip(x, lower, A)
    return (2.0 * clipped / A - 1.0).astype(np.float32)


def process_one_fits(path: str | Path) -> np.ndarray:
    """Load, validate, crop, and normalize one FITS image."""

    path = Path(path)
    if path.stat().st_size < MIN_FILESIZE:
        raise ValueError("File appears empty or too small")

    with fits.open(path, memmap=False) as hdul:
        image = hdul[0].data

    if image is None:
        raise ValueError("FITS data is None")

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    if check_for_corruption(image):
        raise ValueError("Corrupted image (NaN/Inf or too many zero pixels)")

    return normalize_probes(center_crop(image, CROP_SIZE), A=A, lower=LOWER)


def _discover_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing this project's pyproject.toml."""

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and "DIS-Project-Lensed-Galaxy" in pyproject.read_text(
            encoding="utf-8"
        ):
            return candidate
    return start


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for PROBES preprocessing."""

    project_root = _discover_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root / "data" / "raws",
        help="directory containing raw FITS files (default: data/raws)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "data" / "gals_gband_norm",
        help="directory for normalized .npy files (default: data/gals_gband_norm)",
    )
    parser.add_argument(
        "--band",
        default=BAND,
        help=f"FITS band suffix to select (default: {BAND})",
    )
    return parser


def main() -> int:
    """Run the preprocessing CLI and return a process exit status."""

    parser = build_arg_parser()
    args = parser.parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    files = sorted(raw_dir.glob(f"*_{args.band}.fits"))

    if not files:
        pattern = raw_dir / f"*_{args.band}.fits"
        parser.error(f"No files found for pattern: {pattern}")

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0

    for path in tqdm(files, desc=f"Processing {args.band}-band FITS"):
        try:
            image = process_one_fits(path)[None, :, :]
            out_name = path.name.removesuffix(f"_{args.band}.fits") + ".npy"
            np.save(out_dir / out_name, image)
            kept += 1
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")
            skipped += 1

    print("Done.")
    print(f"Saved:   {kept}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {out_dir}")

    if kept == 0:
        print("No usable FITS files were produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
