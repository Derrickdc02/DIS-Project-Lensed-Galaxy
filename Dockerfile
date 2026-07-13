FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /workspace
COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.1,<3.0" \
    && python -m pip install ".[preprocess,figures,validation,dev,docs]"

CMD ["sh", "-c", "pytest -q && validate-mira --smoke-test --device cpu --num-runs 4 --num-bootstrap 4 && preprocess-probes --help >/dev/null && validate-pqmass --help >/dev/null"]
