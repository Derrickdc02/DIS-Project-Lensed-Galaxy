import numpy as np
from tqdm import tqdm
from astropy.io import fits
from glob import glob
from os.path import basename, getsize, join
import os

# =========================
# Settings for reproduction
# =========================
RAW_DIR = "./raws"
OUT_DIR = "./gals_gband_norm"

# Article-aligned settings
BAND = "g"          # The strong-lensing paper shows g-band results
CROP_SIZE = 256     # 256 x 256 images
A = 5.5             # PROBES upper truncation / normalization constant
LOWER = 0.0         # lower truncation
MIN_FILESIZE = 5000 # same practical empty-file check as astroddpm

os.makedirs(OUT_DIR, exist_ok=True)


def check_for_corruption(ar: np.ndarray) -> bool:
    """
    Practical data-quality check inherited from astroddpm.
    Returns True if the image should be skipped.

    Criteria:
    1. contains NaN or inf
    2. more than 30% of pixels are exactly zero
    """
    nanned = np.any(~np.isfinite(ar))
    zeroed = np.sum(ar == 0) > (ar.size * 0.3)
    return bool(nanned or zeroed)


def center_crop(ar: np.ndarray, crop_size: int = 256) -> np.ndarray:
    """
    Crop a 2D image about its center to shape (crop_size, crop_size).
    """
    h, w = ar.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image too small for crop: got {ar.shape}, need at least {crop_size}x{crop_size}")

    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return ar[cy - half: cy + half, cx - half: cx + half]


def normalize_probes(x: np.ndarray, A: float = 5.5, lower: float = 0.0) -> np.ndarray:
    """
    Smith et al. Section 3.1 style preprocessing for PROBES:
      1) clip to [lower, A]
      2) min-max normalize to [-1, 1]

    Formula in the paper:
        x_bar = 2 * max(0, min(x, A)) / A - 1
    """
    x = np.clip(x, lower, A)
    x = 2.0 * x / A - 1.0
    return x.astype(np.float32)


def process_one_fits(fi: str) -> np.ndarray:
    """
    Load one FITS image, run quality checks, center crop, and normalize.
    Returns a (256, 256) float32 array in [-1, 1].
    """
    if getsize(fi) < MIN_FILESIZE:
        raise ValueError("File appears empty or too small")

    with fits.open(fi, memmap=False) as hdul:
        img = hdul[0].data

    if img is None:
        raise ValueError("FITS data is None")

    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    if check_for_corruption(img):
        raise ValueError("Corrupted image (NaN/Inf or too many zero pixels)")

    img = center_crop(img, CROP_SIZE)
    img = normalize_probes(img, A=A, lower=LOWER)
    return img


def main():
    pattern = join(RAW_DIR, f"*_{BAND}.fits")
    files = sorted(glob(pattern))

    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    kept = 0
    skipped = 0

    for fi in tqdm(files, desc=f"Processing {BAND}-band FITS"):
        try:
            img = process_one_fits(fi)

            # Save as single-channel array with explicit channel axis: (1, H, W)
            img = img[None, :, :]

            out_name = basename(fi).replace(f"_{BAND}.fits", ".npy")
            out_path = join(OUT_DIR, out_name)
            np.save(out_path, img)

            kept += 1

        except Exception as e:
            print(f"Skipping {fi}: {e}")
            skipped += 1

    print(f"\nDone.")
    print(f"Saved:   {kept}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()