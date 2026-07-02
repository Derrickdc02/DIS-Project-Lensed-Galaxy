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
  sample_prior.py        unconditional prior draws (PQMass prior check)
  chi2.py                posterior-predictive chi^2 check on saved draws
  figure2.py             Adam et al. Figure 2 reproduction (OOD source + noise sweep)
  backfill_wandb.py      rebuild W&B history from SLURM logs
data/preprocess.py       raw PROBES FITS -> normalized 256x256 .npy
notebooks/               exploratory prototypes of the above
run_stage2.sh, run_stage3.sh, run_sample.sh, run_sample_prior.sh   SLURM job scripts (Wilkes3)
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
4. **Validate the posterior** — `python src/chi2.py --output_dir <run dir>` (per-draw chi^2/N ~ 1).
5. **Validate the prior** — `sbatch run_sample_prior.sh` for unconditional draws, then run `notebooks/PQMassPriorCheck.ipynb` (PQMass two-sample test vs PROBES).
6. **Figure 2** — `python src/figure2.py --ckpt <checkpoint>` (OOD "7" source across noise levels).

## Data & attribution

This project reproduces the method of Adam et al. (2022) (see the top of this
README); it does **not** build on the `astroddpm` codebase. `astroddpm` is used
only to *fetch the raw data*: its script

> https://github.com/Smith42/astroddpm/blob/master/data/probes/get_probes.sh
> (accessed 2026-07-01)

downloads the full PROBES dataset (raw multi-band FITS) into `data/raws/`. The
g-band selection, center-crop, and normalization to 256x256 `.npy` are done by
this repository's own `data/preprocess.py`. None of the PROBES imaging is
redistributed here. The underlying data is the PROBES compilation (Photometry
and Rotation curve OBservations from Extragalactic Surveys; Stone & Courteau et
al.), which carries its own terms.

`astroddpm`'s source code is licensed under the **GNU Affero General Public
License v3.0 (AGPL-3.0)**, but this repository reuses none of that code — only
the raw PROBES data its script downloads. AGPL applies to code, not to the data,
so this project remains under its own MIT license.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
