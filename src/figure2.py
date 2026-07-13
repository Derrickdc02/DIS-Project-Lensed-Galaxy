"""Reproduce Adam et al. 2022 Figure 2: out-of-distribution source + noise sweep.

The ground-truth source is a non-galaxy ("7"). For each noise level we lens it,
add noise, and draw ONE posterior sample. At low noise the likelihood dominates
and the sample recovers the "7"; as the noise grows the likelihood goes
uninformative and the galaxy prior takes over, so the sample looks like a galaxy.

Cheap (one draw per noise level) -> good interactive-GPU job.
Usage: python src/figure2.py --ckpt ../latest.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from sample import (
    load_model,
    posterior_sample,
    lens_forward,
    pixelate_image,
    to_display_flux,
    FLUX_A,
    FLUX_VMIN,
)
from lensing import build_lens_sim, SOURCE_PIXELSCALE


def make_ood_source(size: int = 256, char: str = "7", img_path: str | None = None) -> torch.Tensor:
    """Build a [-1,1] OOD source (background -1, bright +1).

    img_path given -> load image, grayscale, resize. Else render `char`.
    Flipped vertically so it shows upright under imshow(origin="lower").
    """
    if img_path is not None:
        from PIL import Image

        g = Image.open(img_path).convert("L").resize((size, size), Image.LANCZOS)
        buf = np.asarray(g, dtype=np.float32) / 255.0
    else:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = plt.figure(figsize=(1, 1), dpi=size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="black"))
        ax.text(
            0.5, 0.46, char, fontsize=60, fontweight="bold", ha="center", va="center", color="white"
        )
        FigureCanvasAgg(fig).draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float32) / 255.0
        plt.close(fig)
    return torch.from_numpy(2.0 * buf[::-1].copy() - 1.0)  # (size, size) in [-1,1]


def render_figure(src, clean, samples, obs_list, noises: list[float], out: str) -> None:
    """Paper-style filmstrip: row0 = source/samples, row1 = clean/noisy obs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import LogNorm

    ncol = len(noises) + 1
    L, R, TOP, BOT = 0.07, 0.995, 0.90, 0.10  # figure margins (fractions)
    H = 5.6  # width -> square panels (they touch)
    W = H * (TOP - BOT) / 2 * ncol / (R - L)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        2, ncol, left=L, right=R, top=TOP, bottom=BOT, wspace=0.0, hspace=0.0
    )  # no gutters -> filmstrip
    kw = dict(cmap="magma", norm=LogNorm(vmin=FLUX_VMIN, vmax=FLUX_A))

    def show(r, c, img):
        a = fig.add_subplot(gs[r, c])
        a.imshow(to_display_flux(img), origin="lower", **kw)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("white")
            sp.set_linewidth(0.8)
        return a

    show(0, 0, src)
    show(1, 0, clean)
    for k, s in enumerate(noises):
        show(0, k + 1, samples[k])
        a = show(1, k + 1, obs_list[k])
        a.text(
            0.93,
            0.06,
            rf"$\sigma_\mathcal{{N}}={s:g}$",
            transform=a.transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontsize=15,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    x0 = L + 0.5 * (R - L) / ncol  # col-0 center
    x1 = L + (R - L) / ncol + 0.5 * (R - L) * (ncol - 1) / ncol  # cols 1.. center
    fig.text(x0, TOP + 0.015, "Ground Truth", ha="center", fontsize=15)
    fig.text(x1, TOP + 0.015, r"Samples from the posterior $p(x\mid y)$", ha="center", fontsize=17)
    fig.text(
        L - 0.045,
        BOT + 0.75 * (TOP - BOT),
        "Background\nsource (x)",
        va="center",
        ha="center",
        rotation=90,
        fontsize=13,
    )
    fig.text(
        L - 0.045,
        BOT + 0.25 * (TOP - BOT),
        "Distorted\nimage (y)",
        va="center",
        ha="center",
        rotation=90,
        fontsize=13,
    )
    xa = L + (R - L) / ncol
    fig.add_artist(
        FancyArrowPatch(
            (xa, BOT - 0.05),
            (R, BOT - 0.05),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.4,
            color="black",
        )
    )
    fig.text(
        (xa + R) / 2, BOT - 0.085, "Data with increasing levels of noise", ha="center", fontsize=12
    )
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    """Reproduce the out-of-distribution source and noise-level comparison."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to score-model checkpoint")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--noises", type=str, default="0.001,0.1,0.8,1.0,5.0")
    p.add_argument(
        "--source_img",
        type=str,
        default=None,
        help="OOD source image; if omitted, render a plain '7'",
    )
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--out", type=str, default="./outputs/figure2.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noises = [float(s) for s in args.noises.split(",")]
    model, _ = load_model(args.ckpt, device)
    sim = build_lens_sim(device=device, source_pixelscale=SOURCE_PIXELSCALE)

    src = make_ood_source(img_path=args.source_img).to(device)
    with torch.no_grad():
        clean = pixelate_image(lens_forward(sim, src), factor=2)  # (128,128)

    # Each level is checkpointed; re-running skips finished ones (resume after a wall).
    parts = Path(args.out).with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    samples, obs_list = [], []
    for k, s in enumerate(noises):
        part = parts / f"level_{k}.pt"
        if part.exists():
            d = torch.load(part, map_location="cpu", weights_only=False)
            samples.append(d["x"])
            obs_list.append(d["y"])
            print(f"  sigma_N={s:<6g} on disk, skipping")
            continue
        torch.manual_seed(args.seed + k)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + k)
        y = clean + s * torch.randn_like(clean)
        x = posterior_sample(
            model,
            sim,
            y=y,
            sigma_y=s,
            steps=args.steps,
            n_samples=1,
            source_size=src.shape[-1],
            image_pool=2,
        )
        torch.save({"x": x[0, 0].cpu(), "y": y.cpu(), "sigma": s}, part)
        samples.append(x[0, 0].cpu())
        obs_list.append(y.cpu())
        print(f"  sigma_N={s:<6g} done")

    render_figure(src.cpu(), clean.cpu(), samples, obs_list, noises, args.out)


if __name__ == "__main__":
    main()
