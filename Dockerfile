FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY requirements/constraints-py311-cpu.txt requirements/constraints-py311-cpu.txt
COPY src/ src/

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==2.13.0+cpu" \
    && python -m pip install -c requirements/constraints-py311-cpu.txt ".[validation,dev]"

COPY tests/ tests/
COPY notebooks/ notebooks/
COPY scripts/ scripts/
COPY .gitignore .gitignore
COPY slurm_logs/.gitkeep slurm_logs/.gitkeep
COPY artifacts/ artifacts/

CMD ["sh", "-c", "pytest -q && for command in preprocess-probes train-prior train-prior-lowres sample-posterior sample-prior validate-chi2 validate-pqmass validate-mira reproduce-figure2; do $command --help >/dev/null; done && validate-mira --smoke-test --device cpu --num-runs 4 --num-bootstrap 4 >/dev/null"]
