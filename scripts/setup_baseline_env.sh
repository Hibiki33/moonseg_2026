#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_dir"
storage_root="${MOONSEG_STORAGE_ROOT:-/home/disk02/wjj/moonseg}"
mkdir -p "$storage_root/.cache"
storage_root="$(cd -- "$storage_root" && pwd -P)"
baseline_env="$storage_root/.venv-baselines"
if [[ -L "$repo_dir/.venv-baselines" ]]; then
  [[ "$(readlink -f -- "$repo_dir/.venv-baselines")" == "$baseline_env" ]] || {
    printf '%s\n' '.venv-baselines points elsewhere; check MOONSEG_STORAGE_ROOT.' >&2
    exit 1
  }
elif [[ -e "$repo_dir/.venv-baselines" ]]; then
  printf '%s\n' '.venv-baselines must be a link to the selected storage root.' >&2
  exit 1
else
  ln -s "$baseline_env" "$repo_dir/.venv-baselines"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$storage_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$storage_root/.cache/uv-python}"
export UV_PROJECT_ENVIRONMENT="$baseline_env"
export UV_PYTHON_PREFERENCE=only-managed
uv sync --locked --extra baselines --python "$(tr -d '\r\n' < .python-version)"
# decord's published py3-none Linux wheel mistakenly retains a cp36 tag in
# WHEEL. It loads libdecord through ctypes, not the CPython extension ABI.
# Correct only this metadata in the isolated env; replace atomically so uv's
# hardlinked cache and other environments remain untouched.
"$baseline_env/bin/python" - <<'PY'
from importlib.metadata import distribution
import decord
package = distribution("decord")
assert package.version == "0.6.0"
wheel = next(package.locate_file(p) for p in package.files if str(p).endswith(".dist-info/WHEEL"))
content = wheel.read_text()
old = "Tag: cp36-cp36m-manylinux2010_x86_64"
if old in content:
    temporary = wheel.with_suffix(".moonseg-tmp")
    temporary.write_text(content.replace(old, "Tag: py3-none-manylinux2010_x86_64"))
    temporary.replace(wheel)
PY
uv pip check --python "$baseline_env/bin/python"
