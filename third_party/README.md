# First-party model sources

`uv run --no-sync python tools/download_assets.py --only source` fetches the official source into
`third_party/dinov3` at commit `6876159a11b4df116f30f667f8c9888617df0751`.
The downloaded snapshot keeps its original license and contains no local source changes.
The commit identifier records a source version; no asset checksum step is used.

Source: https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751

The wrapper imports the official backbone, DINOv3 Adapter, deformable pixel
decoder, and masked transformer decoder. Runtime compatibility changes live in
`moonseg/models`, so they remain reviewable separately from upstream source.

The complete pinned official source is included. `dinov3/SOURCE_COMPLETE` and
`dinov3/SOURCE_MANIFEST.json` record the revision and archive URL. Its 221 source
files were checked against the official repository tree by path and byte size.
The model uses the official positional encoding and attention implementation;
the local wrapper supplies PyTorch autograd compatibility for deformable attention.

SegFormer-B5 uses the original NVlabs MiT/SegFormerHead sources at
`65fa8cfa9b52b6ee7e8897a98705abf8570f9e32` in `segformer/`. Their model
class bodies are retained; old framework imports are redirected to a small
torch-only compatibility layer. See its `SOURCE_MANIFEST.json` for the exact
original paths and import-only changes.

SAM3 uses the unmodified Meta image-model source and BPE asset at
`f6e51f59500a87c576c2df2323ce56b9fd7a12de` in `sam3/`. Lazy device-cache
construction lives in `moonseg/models/sam3.py`; the upstream files are unchanged.

The competition's lightweight baseline uses the original SMP v0.5.0 package
at `420ce84b0c2df0286fa9bb2bd1499eea625c9b33` in
`segmentation_models_pytorch/`, including its licenses. This is the implementation
named by the competition. The project adapter owns data/training/evaluation
integration and loads local ImageNet-only MobileNetV3 weights.

Each directory records upstream provenance and retains its original license.
Weights are downloaded separately to `ckpts/`, never committed to Git.
