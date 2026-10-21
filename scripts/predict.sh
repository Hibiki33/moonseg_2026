#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,7}"
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT TEST_MANIFEST [--output runs/predictions] [other predict options]" >&2
  exit 2
fi
config="$1"
checkpoint="$2"
manifest="$3"
shift 3
if [[ -n "${PYTHON:-}" ]]; then runner=("$PYTHON"); else runner=(uv run --no-sync python); fi
exec "${runner[@]}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m moonseg.predict --config "$config" --checkpoint "$checkpoint" \
  --manifest "$manifest" --output runs/predictions "$@"
