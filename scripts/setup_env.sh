#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
storage_root="${MOONSEG_STORAGE_ROOT:-/home/disk02/wjj/moonseg}"
mkdir -p "$storage_root/.cache"
storage_root="$(cd -- "$storage_root" && pwd)"
for name in .cache .venv; do
  target="$storage_root/$name"
  if [[ -L "$repo_dir/$name" ]]; then
    if [[ "$(readlink -f -- "$repo_dir/$name")" != "$target" ]]; then
      printf 'Existing %s link points elsewhere; use the matching MOONSEG_STORAGE_ROOT.\n' "$name" >&2
      exit 1
    fi
  elif [[ -e "$repo_dir/$name" ]]; then
    printf 'Existing %s is a real directory; move it to %s before linking.\n' "$name" "$target" >&2
    exit 1
  else
    ln -s "$target" "$repo_dir/$name"
  fi
done
export UV_CACHE_DIR="${UV_CACHE_DIR:-$storage_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$storage_root/.cache/uv-python}"
export UV_PROJECT_ENVIRONMENT="$storage_root/.venv"
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
