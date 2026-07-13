# DIS Project: Lensed Galaxy

A reproducible open-science pipeline for score-based Bayesian reconstruction of
background galaxies in strong gravitational lenses. An NCSN++ prior learned from
PROBES g-band galaxies is combined with a differentiable SIE plus external-shear lens
and a convolved Gaussian likelihood to generate posterior source samples.

## Documentation

The complete scientific workflow, CLI/API reference, Wilkes3 instructions, notebook
catalogue and limitations are published at:

<https://derrickdc02.github.io/DIS-Project-Lensed-Galaxy/>

The source documentation is in [`docs/`](docs/), and can be built locally with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Quick start

Python 3.11 is the supported environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
pytest -q
preprocess-probes --help
```

GPU workflows are submitted from the repository root:

```bash
sbatch scripts/run_stage2.sh
sbatch scripts/run_stage3.sh
sbatch scripts/run_sample.sh
sbatch scripts/run_sample_prior.sh
```

Override runtime paths without editing scientific parameters:

```bash
sbatch --export=ALL,VENV=/path/to/venv,CKPT=/absolute/path/latest.pt   scripts/run_sample.sh
```

Stage 3 requests 36 hours and uses `--max_hours 35` to save and exit before Slurm's
hard limit. Training uses `--resume auto`; sampling resumes at completed chunk
boundaries. Outputs are written below `outputs/` by default and Slurm logs below
`slurm_logs/`.

## Repository layout

- `src/`: installable training, sampling, preprocessing and validation modules.
- `scripts/`: strict Wilkes3/Slurm launchers and shared shell functions.
- `notebooks/`: eight curated HPC/Colab workflows that call shared `src/` code.
- `docs/`: English Sphinx/MyST/Furo documentation.
- `tests/`: CPU unit, CLI utility and notebook-structure tests.

## Data and attribution

This project follows Adam et al. (2022),
[arXiv:2211.03812](https://arxiv.org/abs/2211.03812), but does not copy the
`astroddpm` implementation. Its PROBES download helper is used only to obtain the raw
multi-band FITS data; this repository performs its own g-band selection, centre crop
and normalisation. PROBES imaging is not redistributed here and remains subject to
its own terms. Repository code is MIT licensed; see [`LICENSE`](LICENSE).
