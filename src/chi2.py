"""Image-plane chi-squared diagnostic for saved posterior source draws.

Runs post-hoc on an existing samples/posterior_draws.pt; no resampling.
Usage: python src/chi2.py --output_dir ./outputs/probes_final/sample_srcfov

The injected true source provides the direct chi-squared-per-pixel noise
reference. Posterior-draw values are reported as a distribution rather than
being required individually to equal one.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from lensing import build_lens_sim, SOURCE_PIXELSCALE
from sample import lens_forward, pixelate_image

FLUX_A = 5.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--max_draws",
        type=int,
        default=160,
        help="Use at most this many saved posterior draws (default: 160)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(Path(args.output_dir) / "samples" / "posterior_draws.pt",
                   map_location=device, weights_only=False)
    post = d["post"][:args.max_draws].to(device)
    obs, sigma = d["obs"].to(device), d["noise_sigma"]
    src, N = d["src"].to(device), obs.numel()
    pool = d.get("image_pool", 2)
    sim = build_lens_sim(device=device, source_pixelscale=SOURCE_PIXELSCALE)

    chi2 = torch.empty(post.shape[0], dtype=torch.float64)
    with torch.no_grad():
        true_prediction = pixelate_image(
            lens_forward(sim, src.squeeze()), pool
        )
        chi2_true = (((obs - true_prediction) / sigma) ** 2).sum().item() / N
        image_flux = (
            (FLUX_A / 2.0) * (true_prediction + 1.0)
        ).sum().item()

        for i in range(post.shape[0]):
            pred = pixelate_image(lens_forward(sim, post[i, 0]), pool)
            chi2[i] = (((obs - pred) / sigma) ** 2).sum() / N

        mean_pred = pixelate_image(lens_forward(sim, post.mean(0)[0]), pool)
        chi2_mean = (((obs - mean_pred) / sigma) ** 2).sum().item() / N

    values = chi2.numpy()
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])

    print(f"draws={post.shape[0]}  N={N}  sigma_y={sigma}")
    print(f"true-source    chi^2/N : {chi2_true:.3f}   (noise reference)")
    print(f"posterior-mean chi^2/N : {chi2_mean:.3f}")
    print(
        "posterior draws chi^2/N : "
        f"median {median:.3f}  [16th {q16:.3f}, 84th {q84:.3f}]"
    )
    print(
        "posterior draws auxiliary: "
        f"mean {values.mean():.3f} +/- SD {values.std(ddof=1):.3f}"
        f"  [min {values.min():.3f}, max {values.max():.3f}]"
    )
    print(f"clean image-plane flux : {image_flux:.6g}")


if __name__ == "__main__":
    main()
