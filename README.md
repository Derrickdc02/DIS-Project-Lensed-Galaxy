# DIS-Project-Lensed-Galaxy

We apply score-based diffusion models to strong gravitational lensing, reconstructing posterior distributions of background galaxies from noisy, contaminated observations. This Bayesian approach enables image-level uncertainty quantification and offers insights for both dark matter studies and trustworthy AI image generation.

Method follows Adam et al. 2022 ([arXiv:2211.03812](https://arxiv.org/abs/2211.03812)): an NCSN++ score model trained on PROBES galaxies serves as the prior, and posterior sources are drawn with a convolved-Gaussian-likelihood reverse-diffusion sampler through a differentiable SIE + external-shear lens model (caustics).

## Layout

```
src/
  lensing.py             SIE + external-shear forward model (caustics)
  train_prior.py         stage 3: 256x256 prior, multi-GPU DDP training
  lowres_sample_train.py stage 2: 128x128 prior, single GPU
  sample.py              posterior sampling from a mock lensed observation
  chi2.py                posterior-predictive chi^2 check on saved draws
  figure2.py             Adam et al. Figure 2 reproduction (OOD source + noise sweep)
  backfill_wandb.py      rebuild W&B history from SLURM logs
data/preprocess.py       raw PROBES FITS -> normalized 256x256 .npy
notebooks/               exploratory prototypes of the above
run_stage2.sh, run_stage3.sh, run_sample.sh   SLURM job scripts (Wilkes3)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"   # or pick extras: preprocess, notebooks, dev
```

The SLURM scripts activate `$HOME/rds/hpc-work/venv/dis_proj` by default; override with `VENV=/path/to/venv sbatch ...`.

## Pipeline

1. **Preprocess** — `cd data && python preprocess.py` (expects raw FITS in `data/raws/`, writes `data/gals_gband_norm/`).
2. **Train the prior** — `sbatch run_stage2.sh` (low-res pilot) or `sbatch run_stage3.sh` (full 256x256, 4x A100, resumable via `--resume auto`).
3. **Sample the posterior** — `sbatch run_sample.sh`; chunked and resumable, writes draws + diagnostic figures under `outputs/.../samples/`.
4. **Validate** — `python src/chi2.py --output_dir <run dir>` (per-draw chi^2/N ~ 1).
5. **Figure 2** — `python src/figure2.py --ckpt <checkpoint>` (OOD "7" source across noise levels).

## Tests

```bash
pip install -e ".[dev]"
pytest
```
