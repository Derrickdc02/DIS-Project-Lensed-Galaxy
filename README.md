# DIS Project: Lensed Galaxy

A reproducible open-science pipeline for score-based Bayesian reconstruction of
background galaxies in strong gravitational lenses. An NCSN++ prior learned from
PROBES g-band galaxies is combined with a differentiable SIE plus external-shear lens
and a convolved Gaussian likelihood to generate posterior source samples.

## Documentation

The scientific workflow, CLI/API reference, Wilkes3 instructions, notebook catalogue
and limitations are published at:

<https://derrickdc02.github.io/DIS-Project-Lensed-Galaxy/>

GitHub Pages is deployed from `main`. To build and preview the same documentation
locally:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>. A successful strict build ends with `build succeeded`.

## Installation

Python 3.11 is the supported reproducibility target:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

For the exact CPU versions exercised by CI and Docker:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
python -m pip install -c requirements/constraints-py311-cpu.txt -e ".[all]"
```

## Verify the installation

Run the CPU-safe test and smoke-test suite from the repository root:

```bash
pytest -q
ruff check src tests
nbqa ruff notebooks
preprocess-probes --help
validate-mira --smoke-test --device cpu --num-runs 4 --num-bootstrap 4
```

The current suite contains 33 tests. GPU training and full diffusion sampling are not
part of these CPU checks.

## CPU Docker verification

Docker provides an independent Python 3.11 CPU installation check:

```bash
docker build --tag dis-lensed-galaxy:verify .
docker run --rm dis-lensed-galaxy:verify
```

The image runs all 33 tests, checks all nine CLI entry points and runs a small MIRA
smoke test. It does not emulate Wilkes3, Slurm, CUDA or multi-GPU training. Remove the
local image after verification if desired:

```bash
docker image rm dis-lensed-galaxy:verify
```

## GitLab CI submission

The root `.gitlab-ci.yml` mirrors the CPU-safe GitHub checks for GitLab merge
requests, branches and tags. It runs Python 3.11 tests, Ruff, notebook lint,
wheel/CLI smoke tests, HPC shell checks, a strict Sphinx build and the CPU Docker
verification. It never submits Slurm jobs or runs GPU training and sampling.

After pushing to a GitLab project, check `Settings > CI/CD > Runners` before
opening `Build > Pipelines`. The container job uses Docker-in-Docker and therefore
requires a compatible runner. If that job cannot connect to a Docker daemon, contact
the GitLab administrator rather than weakening the Python or shell checks.

The `deploy-pages` job publishes Sphinx output only from the GitLab default branch
and only when GitLab Pages is enabled by the server. Its URL is shown under
`Deploy > Pages`. Checkpoints, private Drive links and GPU credentials are not
required by any CI job.

## Data and preprocessing

Raw PROBES imaging is not distributed with this repository. After obtaining the FITS
files, place them below `data/raws/` and run:

```bash
preprocess-probes --raw-dir data/raws --out-dir data/gals_gband_norm
```

See the [preprocessing documentation](docs/preprocessing.md) for selection,
normalisation and data-quality details.

## Wilkes3 workflows

Submit jobs from the repository root:

```bash
sbatch scripts/run_stage2.sh
sbatch scripts/run_stage3.sh
sbatch scripts/run_sample.sh
sbatch scripts/run_sample_prior.sh
```

Override runtime paths without changing version-controlled scientific parameters:

```bash
sbatch --export=ALL,VENV=/path/to/venv,CKPT=/absolute/path/latest.pt   scripts/run_sample.sh
```

Stage 3 requests 36 hours and uses `--max_hours 35` to save before Slurm's hard
limit. Training uses `--resume auto`; sampling resumes at completed chunk boundaries.
Outputs are written below `outputs/` and job logs below `slurm_logs/`. Full resources,
expected runtimes and metadata logging are documented in [docs/hpc.md](docs/hpc.md).

## Reproducibility artifacts

The audited artifact inventory is in
[`artifacts/manifest.json`](artifacts/manifest.json). Checkpoints and large generated
outputs remain private and are available for academic reproducibility upon reasonable
request. Email **derricktang02@gmail.com** with your name, affiliation and intended
use. Private Google Drive links and file IDs are not published; see
[`artifacts/README.md`](artifacts/README.md) for the access policy.

## Repository layout

- `src/`: installable training, sampling, preprocessing and validation modules.
- `scripts/`: strict Wilkes3/Slurm launchers and shared shell functions.
- `notebooks/`: eight curated HPC/Colab workflows that call shared `src/` code.
- `docs/`: English Sphinx/MyST/Furo documentation.
- `tests/`: CPU unit, CLI utility and repository-structure tests.
- `artifacts/`: metadata and private-on-request access policy for large results.

## Data and attribution

This project follows Adam et al. (2022),
[arXiv:2211.03812](https://arxiv.org/abs/2211.03812), but does not copy the
`astroddpm` implementation. Its PROBES download helper is used only to obtain the raw
multi-band FITS data; this repository performs its own g-band selection, centre crop
and normalisation. PROBES imaging is not redistributed here and remains subject to
its own terms. Repository code is MIT licensed; see [`LICENSE`](LICENSE).
