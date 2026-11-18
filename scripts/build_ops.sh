#!/usr/bin/env bash
# Build the official DINOv3 (Deformable-DETR) MultiScaleDeformableAttention CUDA extension for this
# project's uv environment. Sources are copied out of third_party (no vendor edits); the .so is written
# to build/ms_deform_attn/ (gitignored), where moonseg.models loads it when
# model.deformable_attention is auto or cuda. Rebuild after any torch / CUDA / Python change.
#
#   bash scripts/build_ops.sh                      # default: CUDA 12.6 toolkit, sm_86 + sm_89
#   CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST="8.6;8.9" bash scripts/build_ops.sh
#   FORCE_CUDA=1 bash scripts/build_ops.sh        # toolkit-only build (e.g. Docker, no GPU)
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
src="$repo_dir/third_party/dinov3/dinov3/eval/segmentation/models/utils/ops"
out="${MOONSEG_MSDA_DIR:-$repo_dir/build/ms_deform_attn}"
# Match torch.version.cuda (cu126 wheels). /usr/local/cuda may point at a different toolkit (12.8 here).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
# 8.6 = RTX A6000 (Ampere), 8.9 = RTX 6000 Ada. Add 8.0 (A100) / 9.0 (H100) if the target host differs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}"
export MAX_JOBS="${MAX_JOBS:-16}"
# Normal local builds may select a GPU; FORCE_CUDA builds only need the toolkit.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${BUILD_GPU:-0}"
[[ -d "$src/src" ]] || { echo "Official ops sources missing: $src" >&2; exit 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "nvcc not found under CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
# Resolve the project interpreter here: uv run from the temporary build directory would not find the project.
if [[ -n "${PYTHON:-}" ]]; then runner=("$PYTHON"); else runner=(uv run --no-sync python); fi
py="$("${runner[@]}" -c 'import sys; print(sys.executable)')"
"$py" - "$CUDA_HOME" <<'PY'
import os, re, subprocess, sys, torch
if torch.version.cuda is None:
    sys.exit("A CUDA-enabled PyTorch wheel is required")
nvcc = subprocess.run([sys.argv[1] + "/bin/nvcc", "--version"], capture_output=True, text=True).stdout
found = re.search(r"release (\d+\.\d+)", nvcc).group(1)
if found.split(".")[0] != torch.version.cuda.split(".")[0]:
    sys.exit(f"nvcc {found} and torch CUDA {torch.version.cuda} have different major versions")
if found != torch.version.cuda:
    print(f"warning: nvcc {found} != torch CUDA {torch.version.cuda}; prefer the matching toolkit", file=sys.stderr)
if not torch.cuda.is_available() and os.environ.get("FORCE_CUDA") != "1":
    sys.exit("No visible CUDA device; set BUILD_GPU to an idle GPU index")
print(f"Building with {sys.executable}, torch {torch.__version__}, nvcc {found}")
PY
tmp_root="${TMPDIR:-$repo_dir/build}"
mkdir -p "$tmp_root" "$out"
work="$(mktemp -d "$tmp_root/msda_build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
cp -r "$src"/. "$work"/
# Upstream does not implement FORCE_CUDA. Adapt only our disposable build copy,
# keeping the checked-in official sources untouched.
if [[ "${FORCE_CUDA:-0}" == "1" ]]; then
  "$py" - "$work/setup.py" <<'PY'
import pathlib, sys
setup = pathlib.Path(sys.argv[1])
text = setup.read_text()
old = "if torch.cuda.is_available() and CUDA_HOME is not None:"
new = 'if (torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1") and CUDA_HOME is not None:'
if text.count(old) != 1:
    sys.exit("Upstream CUDA build condition changed; review the temporary FORCE_CUDA adapter")
setup.write_text(text.replace(old, new))
PY
fi
( cd "$work" && "$py" setup.py build_ext --build-lib "$out" --build-temp "$work/tmp" ) > "$work/build.log" 2>&1 \
  || { tail -40 "$work/build.log" >&2; exit 1; }
"$py" - "$out" "$src" "$CUDA_HOME" <<'PY'
import json, os, pathlib, platform, re, subprocess, sys, torch
out, src, cuda_home = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(out))
import MultiScaleDeformableAttention as ext  # import check against this exact torch ABI
so = pathlib.Path(ext.__file__)
nvcc = subprocess.run([cuda_home + "/bin/nvcc", "--version"], capture_output=True, text=True).stdout
manifest = {"extension": so.name, "source": str(src),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": platform.python_version(),
            "cuda_home": cuda_home, "nvcc": re.search(r"release (\S+)", nvcc).group(1).rstrip(","),
            "arch_list": os.environ["TORCH_CUDA_ARCH_LIST"]}
(out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
