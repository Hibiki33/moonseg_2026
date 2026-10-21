#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Physical GPU 0,7 map to local cuda:0,cuda:1 inside torchrun workers.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
config="${1:-configs/moonseg_vitb16.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -n "${PYTHON:-}" ]]; then runner=("$PYTHON"); else runner=(uv run --no-sync python); fi
exec "${runner[@]}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m moonseg.train --config "$config" "$@"
