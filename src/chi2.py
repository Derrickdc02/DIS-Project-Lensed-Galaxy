"""Posterior-predictive chi^2 check (Adam et al. 2022): each posterior source
draw, pushed through the lens+pool forward model, should fit the observation to
within the noise -> per-sample chi^2/N ~ 1. The posterior mean fits better
(chi^2/N < 1) because averaging smooths the residual.

Runs post-hoc on an existing samples/posterior_draws.pt; no resampling.
Usage: python src/chi2.py --output_dir ./outputs/probes_final/sample_srcfov
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from lensing import build_lens_sim


def lens_forward(sim, x):
    return sim({"source": {"image": x + 1.0}}) - 1.0


def pool(img, factor=2):
    return F.avg_pool2d(img[None, None], factor).squeeze(0).squeeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(Path(args.output_dir) / "samples" / "posterior_draws.pt",
                   map_location=device, weights_only=False)
    post, obs, sigma = d["post"].to(device), d["obs"].to(device), d["noise_sigma"]
    src, N = d["src"].to(device), obs.numel()
    sim = build_lens_sim(device=device, source_pixelscale=0.028)

    chi2 = torch.empty(post.shape[0])
    with torch.no_grad():
        chi2_true = (((obs - pool(lens_forward(sim, src.squeeze()), 2)) / sigma)
                     ** 2).sum().item() / N
        for i in range(post.shape[0]):
            pred = pool(lens_forward(sim, post[i, 0]), 2)
            chi2[i] = (((obs - pred) / sigma) ** 2).sum() / N
        mean_pred = pool(lens_forward(sim, post.mean(0)[0]), 2)
        chi2_mean = (((obs - mean_pred) / sigma) ** 2).sum().item() / N

    print(f"draws={post.shape[0]}  N={N}  sigma_y={sigma}")
    print(f"true-source    chi^2/N : {chi2_true:.3f}   (consistency check, must be ~1.0)")
    print(f"posterior-mean chi^2/N : {chi2_mean:.3f}")
    print(f"per-sample     chi^2/N : {chi2.mean():.3f} +/- {chi2.std():.3f}"
          f"  [min {chi2.min():.3f}, max {chi2.max():.3f}]   (target ~1.0)")


if __name__ == "__main__":
    main()
