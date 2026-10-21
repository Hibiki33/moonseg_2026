# Official DINOv3 source

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
