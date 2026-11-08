#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Unset -> GPUs 0,7; explicitly empty (CUDA_VISIBLE_DEVICES=) means CPU only; -1 also works.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,7}"
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT TEST_MANIFEST --official-test-allowlist OFFICIAL_TEST_LIST [--output runs/predictions] [--weights-key auto|model|ema_model] [other predict options]" >&2
  echo "  --official-test-allowlist is required: the organizer's complete fixed test sample list." >&2
  exit 2
fi
config="$1"
checkpoint="$2"
manifest="$3"
shift 3
python="${PYTHON:-.venv/bin/python}"
if [[ "$python" != */* ]]; then python="$(command -v -- "$python")"; fi
python="$(cd -P -- "$(dirname -- "$python")" && pwd)/$(basename -- "$python")"
runner=("$python")
exec "${runner[@]}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m moonseg.predict --config "$config" --checkpoint "$checkpoint" \
  --manifest "$manifest" --output runs/predictions "$@"
