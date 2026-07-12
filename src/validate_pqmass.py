"""PQMass two-sample validation for trained prior draws and PROBES images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_prior_samples(path: str | Path) -> torch.Tensor:
    """Load prior draws from a merged tensor file or resumable chunk directory."""

    path = Path(path)
    if path.is_file():
        samples = torch.load(path, map_location="cpu", weights_only=False)
    elif path.is_dir():
        merged = path / "prior_samples.pt"
        if merged.is_file():
            samples = torch.load(merged, map_location="cpu", weights_only=False)
        else:
            chunks = sorted((path / "chunks").glob("chunk_*.pt"))
            if not chunks:
                chunks = sorted(path.glob("chunk_*.pt"))
            if not chunks:
                raise FileNotFoundError(f"No prior_samples.pt or chunk_*.pt files under {path}")
            samples = torch.cat(
                [
                    torch.load(chunk, map_location="cpu", weights_only=False)
                    for chunk in chunks
                ],
                dim=0,
            )
    else:
        raise FileNotFoundError(f"Prior sample path does not exist: {path}")

    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"Expected a tensor of prior samples, got {type(samples).__name__}")
    if samples.ndim == 3:
        samples = samples.unsqueeze(1)
    if samples.ndim != 4 or samples.shape[1] != 1:
        raise ValueError(f"Expected prior shape (N, 1, H, W), got {tuple(samples.shape)}")
    if not torch.isfinite(samples).all():
        raise ValueError("Prior samples contain NaN or Inf values")
    return samples.float().clamp(-1.0, 1.0)


def load_real_samples(
    data_dir: str | Path,
    *,
    image_size: int,
    max_samples: int | None,
    seed: int,
) -> torch.Tensor:
    """Load a deterministic random subset of normalized PROBES arrays."""

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(files))
    if max_samples is not None:
        order = order[:max_samples]

    arrays = []
    for index in order:
        array = np.load(files[int(index)]).astype(np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D image in {files[int(index)]}, got {array.shape}")
        arrays.append(array)

    real = torch.from_numpy(np.stack(arrays)).unsqueeze(1)
    if real.shape[-2:] != (image_size, image_size):
        real = F.interpolate(
            real,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    if not torch.isfinite(real).all():
        raise ValueError("Real PROBES samples contain NaN or Inf values")
    return real.float()


def prepare_two_sample_arrays(
    prior: torch.Tensor,
    real: torch.Tensor,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match sample counts and flatten prior and real images for PQMass."""

    sample_count = min(prior.shape[0], real.shape[0])
    if max_samples is not None:
        sample_count = min(sample_count, max_samples)
    if sample_count < 16:
        raise ValueError(f"PQMass requires at least 16 samples per set, got {sample_count}")

    generator = torch.Generator().manual_seed(seed)
    prior_order = torch.randperm(prior.shape[0], generator=generator)[:sample_count]
    real_order = torch.randperm(real.shape[0], generator=generator)[:sample_count]
    prior_flat = prior[prior_order].reshape(sample_count, -1).numpy().astype(np.float32)
    real_flat = real[real_order].reshape(sample_count, -1).numpy().astype(np.float32)
    return prior_flat, real_flat


def pca_scores(
    real: np.ndarray,
    prior: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project both sample sets into an exact joint PCA score space.

    A dual Gram-matrix eigendecomposition avoids materialising the much larger
    pixel covariance matrix.
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    combined = np.concatenate([real, prior], axis=0).astype(np.float64)
    centered = combined - combined.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = order[eigenvalues[order] > 0]
    count = min(n_components, len(positive))
    if count == 0:
        raise ValueError("PCA input has no positive-variance components")

    selected = positive[:count]
    scores = eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])
    retained = float(eigenvalues[selected].sum() / eigenvalues[positive].sum())
    split = real.shape[0]
    return (
        scores[:split].astype(np.float32),
        scores[split:].astype(np.float32),
        retained,
    )


def pqmass_statistics(
    prior: np.ndarray,
    real: np.ndarray,
    *,
    num_refs: int,
    re_tessellation: int,
    seed: int,
) -> dict[str, Any]:
    """Run PQMass and return distributions plus summary statistics."""

    from pqm import pqm_chi2, pqm_pvalue

    if prior.shape != real.shape:
        raise ValueError(f"PQMass arrays must have matching shapes, got {prior.shape} and {real.shape}")

    sample_count = prior.shape[0]
    effective_refs = min(num_refs, max(2, sample_count // 8))
    if effective_refs >= sample_count // 2:
        raise ValueError("num_refs must be smaller than each real-vs-real sanity subset")

    np.random.seed(seed)
    torch.manual_seed(seed)
    options = {
        "num_refs": effective_refs,
        "re_tessellation": re_tessellation,
        "z_score_norm": True,
        "gauss_frac": 0.05,
    }
    chi2_values = np.asarray(pqm_chi2(prior, real, **options), dtype=np.float64)
    pvalues = np.asarray(pqm_pvalue(prior, real, **options), dtype=np.float64)

    half = sample_count // 2
    sanity = np.asarray(
        pqm_pvalue(real[:half], real[half : 2 * half], **options),
        dtype=np.float64,
    )
    degrees_of_freedom = effective_refs - 1
    return {
        "num_refs": effective_refs,
        "degrees_of_freedom": degrees_of_freedom,
        "chi2_over_dof_mean": float(chi2_values.mean() / degrees_of_freedom),
        "pvalue_mean": float(pvalues.mean()),
        "real_vs_real_pvalue_mean": float(sanity.mean()),
        "chi2_values": chi2_values,
        "pvalues": pvalues,
        "real_vs_real_pvalues": sanity,
    }


def _summary_without_arrays(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in statistics.items()
        if not isinstance(value, np.ndarray)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the PQMass command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True, help="prior_samples.pt or its directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="normalized PROBES .npy directory")
    parser.add_argument("--output", type=Path, default=Path("pqm_results.json"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--num-refs", type=int, default=100)
    parser.add_argument("--re-tessellation", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=21)
    return parser


def main() -> int:
    """Run pixel-space and PCA-space PQMass validation."""

    args = build_arg_parser().parse_args()
    if args.max_samples < 16:
        raise SystemExit("--max-samples must be at least 16")
    if args.num_refs < 2:
        raise SystemExit("--num-refs must be at least 2")
    if args.re_tessellation < 1:
        raise SystemExit("--re-tessellation must be positive")

    prior = load_prior_samples(args.prior)
    real = load_real_samples(
        args.data_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    prior_flat, real_flat = prepare_two_sample_arrays(
        prior,
        real,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    pixel = pqmass_statistics(
        prior_flat,
        real_flat,
        num_refs=args.num_refs,
        re_tessellation=args.re_tessellation,
        seed=args.seed,
    )
    real_pca, prior_pca, retained = pca_scores(
        real_flat,
        prior_flat,
        args.pca_components,
    )
    pca = pqmass_statistics(
        prior_pca,
        real_pca,
        num_refs=args.num_refs,
        re_tessellation=args.re_tessellation,
        seed=args.seed,
    )

    result = {
        "sample_count": int(prior_flat.shape[0]),
        "pixel_dimension": int(prior_flat.shape[1]),
        "pca_components": int(real_pca.shape[1]),
        "pca_retained_variance": retained,
        "seed": args.seed,
        "pixel": _summary_without_arrays(pixel),
        "pca": _summary_without_arrays(pca),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
