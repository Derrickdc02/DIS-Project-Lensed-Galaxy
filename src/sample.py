"""Posterior source reconstruction from a strong-lensing observation.

Standalone HPC-batch version of notebooks/full_sample.ipynb: load a trained
NCSN++ score prior, build the caustics SIE + external-shear forward model, make a
mock observation from a real PROBES galaxy, and draw x ~ p(x | y) with the
convolved-likelihood sampler (Adam et al. 2022); save the draws and diagnostic
figures. Defaults reproduce the full run: 8000 steps, 160 draws, chunks of 32.
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from score_models import NCSNpp, ScoreModel

from lensing import build_lens_sim  # src/lensing.py: SIE + external-shear forward model


# ---- Display scaling ----
# preprocess.py maps flux -> [-1,1] via x = 2*clip(flux,0,A)/A - 1, so flux = A*(x+1)/2.
# Adam et al. show intensity on a log scale; invert to flux for LogNorm (display only).
FLUX_A = 5.5        # PROBES normalization constant (must match data/preprocess.py)
FLUX_VMIN = 1e-2    # lower end of the log intensity colorbar, in flux units


def to_display_flux(img, floor=1e-3):
    """[-1,1] intensity -> PROBES flux [0, FLUX_A], floored >0 so LogNorm has no masked pixels."""
    if isinstance(img, torch.Tensor):
        return (FLUX_A * (img.detach().cpu() + 1.0) / 2.0).clamp(min=floor).numpy()
    return np.clip(FLUX_A * (np.asarray(img) + 1.0) / 2.0, floor, None)


# ---- Model + forward operator ----
def load_model(ckpt_path, device):
    """Load the EMA score model from a train_prior.py checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["score_model_hyperparameters"]

    net = NCSNpp(**hp)
    net.load_state_dict(ckpt["ema_model"])
    model = ScoreModel(
        model=net,
        sigma_min=hp["sigma_min"],
        sigma_max=hp["sigma_max"],
        device=device,
    ).to(device)
    print(
        f"Loaded EMA at step {ckpt['step']}, "
        f"sigma_max={model.sde.sigma_max}, sigma_min={model.sde.sigma_min}"
    )
    return model, int(ckpt["step"])


# ---- Posterior sampler (Adam et al. 2022, convolved likelihood, eqs. 19, 20) ----
def pixelate_image(img, factor=2):
    """Average-pool image-plane outputs by factor, preserving leading dims."""
    if factor == 1:
        return img
    if img.ndim == 2:
        return F.avg_pool2d(img[None, None], kernel_size=factor).squeeze(0).squeeze(0)
    if img.ndim == 3:
        return F.avg_pool2d(img[:, None], kernel_size=factor).squeeze(1)
    raise ValueError(f"Expected image shape (H, W) or (N, H, W), got {tuple(img.shape)}")


def lens_forward(sim, x):
    """Lens source image x. The shift makes out-of-FOV rays (which source_pixelscale
    leaves outside the source grid) map to -1 = empty sky, not caustics' 0-pad =
    mid-flux. In-bounds is unchanged since the interpolation is linear in x."""
    return sim({"source": {"image": x + 1.0}}) - 1.0


@torch.no_grad()
def posterior_sample(
    model, sim, y, sigma_y, steps=8000, n_samples=32,
    source_size=256, image_pool=2,
):
    """Reverse-diffusion posterior sampler with a convolved Gaussian likelihood."""
    device = next(model.parameters()).device
    sde = model.sde

    t_grid = torch.linspace(1.0, 1e-3, steps + 1, device=device)
    x = sde.prior([n_samples, 1, source_size, source_size]).sample().to(device)

    for i in range(steps):
        t_scalar = t_grid[i]
        t_batch = t_scalar.expand(n_samples)
        h = t_grid[i] - t_grid[i + 1]
        var = sigma_y ** 2 + sde.sigma(t_scalar) ** 2
        g = sde.diffusion(t_batch, x)

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            preds_256 = torch.stack([lens_forward(sim, x_req[b, 0])
                                     for b in range(n_samples)])
            preds = pixelate_image(preds_256, image_pool)
            log_lik = -0.5 * ((y - preds) ** 2).sum() / var
            grad_ll = torch.autograd.grad(log_lik, x_req)[0]

        score_post = model.score(t_batch, x) + grad_ll
        z = torch.randn_like(x)
        x = x + (g ** 2) * score_post * h + g * z * h.sqrt()

    return x


# ---- Ground-truth source ----
def load_source(data_dir, pick, device):
    """Load one preprocessed PROBES image -> (H, W) float32 source in [-1, 1]."""
    src_files = sorted(glob.glob(str(Path(data_dir) / "*.npy")))
    assert src_files, f"No .npy files found in {data_dir}"
    arr = np.load(src_files[pick]).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    name = Path(src_files[pick]).stem
    src = torch.from_numpy(arr).to(device)
    print(f"Ground truth: [{pick}] {name}  shape={tuple(src.shape)}  "
          f"range=[{src.min():.3f}, {src.max():.3f}]")
    return src, name


# ---- Diagnostics ----
def plot_mean_std(src, obs, post_mean, post_std, n_post, out_path):
    """True | obs | mean | std | residual | z-score, with calibration numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    residual = post_mean - src
    zscore = residual / post_std.clamp(min=1e-6)

    # Intensity panels: physical flux on a log scale (Adam et al. convention).
    flux_kw = dict(cmap="magma", norm=LogNorm(vmin=FLUX_VMIN, vmax=FLUX_A))
    panels = [
        ("True source", to_display_flux(src), flux_kw),
        ("Observation", to_display_flux(obs), flux_kw),
        (f"Posterior mean (N={n_post})", to_display_flux(post_mean), flux_kw),
        ("Posterior std", post_std.numpy(), dict(cmap="viridis")),
        ("Mean - True", residual.numpy(), dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5)),
        ("z-score (resid / std)", zscore.numpy(), dict(cmap="RdBu_r", vmin=-3, vmax=3)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.8))
    for ax, (title, img, kw) in zip(axes, panels):
        h = ax.imshow(img, origin="lower", **kw)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    rmse_mean = (residual ** 2).mean().sqrt().item()
    mean_std = post_std.mean().item()
    mean_abs_z = zscore.abs().mean().item()
    print(f"RMSE(posterior mean, true source) = {rmse_mean:.4f}")
    print(f"Mean per-pixel posterior std      = {mean_std:.4f}")
    print(f"Mean |z-score|                    = {mean_abs_z:.3f}   "
          f"(well-calibrated Gaussian -> sqrt(2/pi) ~ 0.798)")


def plot_grid(
    sim, post, src, obs, post_mean, post_std, noise_sigma, device, out_path,
    image_pool=2,
):
    """Source- and image-plane summary grid (mirrors the notebook's final cell)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with torch.no_grad():
        A_post_256 = torch.stack([
            lens_forward(sim, post[i, 0].to(device))
            for i in range(post.shape[0])
        ]).cpu()
        A_post = pixelate_image(A_post_256, image_pool)
    A_post_mean = A_post.mean(dim=0)
    A_post_std = A_post.std(dim=0)

    torch.manual_seed(0)
    idx = torch.randperm(post.shape[0])[:2].tolist()
    s1, s2 = post[idx[0], 0], post[idx[1], 0]
    i1, i2 = A_post[idx[0]], A_post[idx[1]]

    src_resid = (src - post_mean) / noise_sigma
    img_resid = (obs - A_post_mean) / noise_sigma

    std_vmax = max(post_std.max().item(), A_post_std.max().item())
    res_vmax = 5.0

    # Both planes share one physical-flux log scale (Adam et al. intensity colorbar);
    # surface brightness is conserved, so source and image plane are directly comparable.
    intensity_kw = dict(cmap="magma", norm=LogNorm(vmin=FLUX_VMIN, vmax=FLUX_A), origin="lower")
    std_kw = dict(cmap="hot", vmin=0, vmax=std_vmax, origin="lower")
    res_kw = dict(cmap="RdBu_r", vmin=-res_vmax, vmax=res_vmax, origin="lower")

    fig, axes = plt.subplots(2, 6, figsize=(20, 6.6),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.04})

    axes[0, 0].imshow(to_display_flux(src), **intensity_kw)
    axes[0, 1].imshow(to_display_flux(s1), **intensity_kw)
    axes[0, 2].imshow(to_display_flux(s2), **intensity_kw)
    im_int = axes[0, 3].imshow(to_display_flux(post_mean), **intensity_kw)
    im_std = axes[0, 4].imshow(post_std, **std_kw)
    im_res = axes[0, 5].imshow(src_resid, **res_kw)

    axes[1, 0].imshow(to_display_flux(obs), **intensity_kw)
    axes[1, 1].imshow(to_display_flux(i1), **intensity_kw)
    axes[1, 2].imshow(to_display_flux(i2), **intensity_kw)
    axes[1, 3].imshow(to_display_flux(A_post_mean), **intensity_kw)
    axes[1, 4].imshow(A_post_std, **std_kw)
    axes[1, 5].imshow(img_resid, **res_kw)

    titles = ["Ground Truth", r"$x \sim p(x \mid y)$", r"$x \sim p(x \mid y)$",
              r"$\mu$", r"$\sigma$", r"$(\mathrm{GT}-\mu)/\sigma_{\mathcal{N}}$"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=14)
    axes[1, 0].text(0.04, 0.96, fr"$\sigma_{{\mathcal{{N}}}}={noise_sigma:g}$",
                    transform=axes[1, 0].transAxes, color="white", fontsize=11, va="top")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(right=0.9)
    cax_int = fig.add_axes([0.91, 0.55, 0.012, 0.34])
    cax_std = fig.add_axes([0.91, 0.32, 0.012, 0.20])
    cax_res = fig.add_axes([0.91, 0.08, 0.012, 0.20])
    fig.colorbar(im_int, cax=cax_int).set_label("Intensity", fontsize=11)
    fig.colorbar(im_std, cax=cax_std).set_label("Dispersion", fontsize=11)
    fig.colorbar(im_res, cax=cax_res).set_label("Residuals", fontsize=11)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Main ----
def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=str, default="./outputs/probes_final",
                   help="train_prior.py output dir; checkpoint + samples live here")
    p.add_argument("--data_dir", type=str, default="./data/gals_gband_norm",
                   help="dir of preprocessed PROBES *.npy images")
    p.add_argument("--ckpt", type=str, default="latest.pt",
                   help="checkpoint filename inside output_dir (or an absolute path)")

    # Sampling (notebook full-run defaults)
    p.add_argument("--steps", type=int, default=8000, help="reverse-diffusion steps")
    p.add_argument("--n_post", type=int, default=160, help="total posterior draws")
    p.add_argument("--chunk", type=int, default=32, help="draws per posterior_sample call")
    p.add_argument("--seed", type=int, default=21)

    # Observation
    p.add_argument("--pick", type=int, default=15, help="index into sorted data_dir/*.npy")
    p.add_argument("--noise_sigma", type=float, default=0.02, help="observation noise std")

    # Forward-model geometry is fixed in src/lensing.py's build_lens_sim defaults,
    # not exposed as CLI flags. Edit the build_lens_sim(...) call below to vary it.
    return p


def atomic_save(obj, path):
    """Save via a temp file then atomic rename, so a walltime kill can't leave a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    args = build_arg_parser().parse_args()
    assert args.n_post % args.chunk == 0, "--n_post must be a multiple of --chunk"
    n_chunks = args.n_post // args.chunk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = output_dir / args.ckpt
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(ckpt_path, device)

    # Geometry from src/lensing.py defaults: SIE (q=0.7, phi=pi/6, Rein=1.2) + shear (0.03, 0.04).
    # source_pixelscale=0.028 shrinks the source FOV inside the lens footprint -> uniform posterior std.
    sim = build_lens_sim(device=device, source_pixelscale=0.028)

    # --- Resumable observation: persist y so every chunk (and any resume) uses the
    # same noisy observation; refuse to mix chunks made with different settings. ---
    chunks_dir = samples_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    obs_path = samples_dir / "observation.pt"

    if obs_path.exists():
        meta = torch.load(obs_path, map_location="cpu", weights_only=False)
        mismatch = {k: (meta.get(k), getattr(args, k))
                    for k in ("pick", "noise_sigma", "seed")
                    if meta.get(k) != getattr(args, k)}
        if mismatch:
            raise SystemExit(
                f"{obs_path} was made with different settings {mismatch}.\n"
                f"Use a fresh --output_dir, or delete {chunks_dir} and {obs_path}."
            )
        src = meta["src"].to(device)
        obs = meta["obs"].to(device)
        src_name = meta["src_name"]
        print(f"Resuming: reusing observation [{args.pick}] {src_name} from {obs_path}")
    else:
        src, src_name = load_source(args.data_dir, args.pick, device)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            lensed_256 = lens_forward(sim, src)
            lensed = pixelate_image(lensed_256, factor=2)
        obs = lensed + args.noise_sigma * torch.randn_like(lensed)
        atomic_save({"src": src.cpu(), "obs": obs.cpu(), "pick": args.pick,
                     "src_name": src_name, "noise_sigma": args.noise_sigma,
                     "seed": args.seed}, obs_path)

    # --- Posterior draws, one resumable chunk at a time (chunk c seeded args.seed+c,
    # so a resumed job reproduces an uninterrupted run exactly). ---
    print(f"\nDrawing {args.n_post} posterior samples "
          f"({n_chunks} x {args.chunk}) at {args.steps} steps...")
    t0 = time.time()
    for c in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{c:03d}.pt"
        if chunk_path.exists():
            print(f"  chunk {c + 1}/{n_chunks} already on disk -> {chunk_path.name}, skipping")
            continue
        torch.manual_seed(args.seed + c)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + c)
        chunk = posterior_sample(
            model, sim,
            y=obs.to(device),
            sigma_y=args.noise_sigma,
            steps=args.steps,
            n_samples=args.chunk,
            source_size=src.shape[-1],
            image_pool=2,
        ).cpu()
        atomic_save(chunk, chunk_path)
        elapsed = time.time() - t0
        print(f"  chunk {c + 1}/{n_chunks} done -> {chunk_path.name} ({elapsed:.1f}s elapsed)")

    # --- Assemble all chunks (freshly drawn + previously saved) ----------- #
    post = torch.cat(
        [torch.load(chunks_dir / f"chunk_{c:03d}.pt", map_location="cpu")
         for c in range(n_chunks)],
        dim=0,
    )                                           # (N_POST, 1, H, W)
    post_mean = post.mean(dim=0).squeeze(0)     # (H, W)
    post_std = post.std(dim=0).squeeze(0)       # (H, W)

    draws_path = samples_dir / "posterior_draws.pt"
    atomic_save(
        {"post": post, "src": src.cpu(), "obs": obs.cpu(),
         "pick": args.pick, "src_name": src_name,
         "noise_sigma": args.noise_sigma, "steps": args.steps,
         "image_pool": 2},
        draws_path,
    )
    print(f"Saved {post.shape[0]} draws -> {draws_path}")

    # Diagnostics
    mean_std_png = samples_dir / "posterior_mean_std.png"
    grid_png = samples_dir / "posterior_grid.png"
    plot_mean_std(
        src.cpu(), obs.cpu(), post_mean, post_std, args.n_post, mean_std_png,
    )
    plot_grid(
        sim, post, src.cpu(), obs.cpu(), post_mean, post_std,
        args.noise_sigma, device, grid_png, image_pool=2,
    )
    print(f"Wrote figures -> {mean_std_png}, {grid_png}")


if __name__ == "__main__":
    main()
