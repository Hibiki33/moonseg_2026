#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -f .git/config && ! -f .git ]]; then
  git init --initial-branch=main
fi
git config --local user.name jiajun
git config --local user.email jiajun9264@gmail.com
git status --short
