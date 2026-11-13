#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.."
# Physical GPU 0,7 map to local cuda:0,cuda:1 inside torchrun workers.
# Build the compiled deformable attention once (bash scripts/build_ops.sh); otherwise
# model.deformable_attention=auto warns and falls back to the slower PyTorch path.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Unset -> GPUs 0,7; explicitly empty (CUDA_VISIBLE_DEVICES=) means CPU only; -1 also works.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
config="${1:-configs/dnv3_va_m2f_vitl16_default.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
# Resolve the environment directory, not bin/python's final symlink: Python must
# still discover pyvenv.cfg. Launch directly so uv need not inspect a NAS checkout.
python="${PYTHON:-.venv/bin/python}"
if [[ "$python" != */* ]]; then python="$(command -v -- "$python")"; fi
python="$(cd -P -- "$(dirname -- "$python")" && pwd)/$(basename -- "$python")"
runner=("$python")
exec "${runner[@]}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m moonseg.train --config "$config" "$@"
