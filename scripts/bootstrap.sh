#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ -n "${PYTHON:-}" ]]; then runner=("$PYTHON"); else runner=(uv run --no-sync python); fi
exec "${runner[@]}" tools/download_assets.py "$@"
