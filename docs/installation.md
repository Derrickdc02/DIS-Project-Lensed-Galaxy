# Installation

Python 3.11 is the supported reproducibility target. From a fresh clone:

```bash
git clone https://github.com/Derrickdc02/DIS-Project-Lensed-Galaxy.git
cd DIS-Project-Lensed-Galaxy
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

For a smaller environment, install only the relevant extras:

```bash
python -m pip install -e ".[preprocess]"   # FITS preprocessing
python -m pip install -e ".[validation]"   # PQMass and MIRA
python -m pip install -e ".[dev,docs]"     # tests, lint and documentation
```

For the exact Python 3.11 CPU versions exercised by CI and Docker, apply the
recorded constraints after installing CPU PyTorch:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
python -m pip install -c requirements/constraints-py311-cpu.txt -e ".[all]"
```

Check the installation without a GPU:

```bash
pytest -q
preprocess-probes --help
validate-mira --smoke-test --device cpu --num-runs 4 --num-bootstrap 4
```

GPU training depends on a CUDA-compatible PyTorch installation and is intended for
Wilkes3. The CPU container is a validation environment, not a training environment.
