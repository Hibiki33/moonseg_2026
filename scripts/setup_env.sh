#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_dir/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$repo_dir/.cache/uv-python}"
export UV_PYTHON_PREFERENCE=only-managed
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
python_version="$(tr -d '\r\n' < .python-version)"
uv python install --no-bin "$python_version"
if [[ -f uv.lock ]]; then
  uv sync --locked "$@"
else
  uv lock
  uv sync --locked "$@"
fi
uv run --no-sync python -c 'import torch; print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda, "visible GPUs:", torch.cuda.device_count())'
