# syntax=docker/dockerfile:1
# Official tags: NVIDIA container-images/cuda supported-tags; astral-sh/uv release 0.13.4.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    PYTHON=/opt/venv/bin/python \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    PYTHONUNBUFFERED=1
WORKDIR /workspace
ARG BASELINES=0

COPY pyproject.toml uv.lock .python-version ./
RUN if [ "$BASELINES" = 1 ]; then uv sync --locked --extra baselines --no-install-project --no-cache; \
    else uv sync --locked --no-install-project --no-cache; fi

# Keep source paths intact: model loaders locate vendored code relative to moonseg.
COPY moonseg/ moonseg/
COPY third_party/ third_party/
COPY scripts/ scripts/
COPY configs/ configs/
COPY train.py predict.py LICENSE README.md ./
RUN if [ "$BASELINES" = 1 ]; then uv sync --locked --extra baselines --no-build-isolation --no-cache; \
    else uv sync --locked --no-build-isolation --no-cache; fi
RUN FORCE_CUDA=1 MAX_JOBS=4 bash scripts/build_ops.sh \
    && mkdir -p data ckpts runs

# Dataset, pretrained weights and run outputs are supplied as runtime bind mounts.
CMD ["bash"]
