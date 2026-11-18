"""Reproduce the submitted DA3 narrowflip predictions on the official RGB images."""
import sys
from pathlib import Path

from moonseg.predict import main

if __name__ == "__main__":
    config = Path("configs/da3mono_vitl14_finetuned_e25.yaml")
    root = Path("data/LuSNAR_official")
    main(["--config", str(config), "--profile", "narrowflip", "--weights-key", "model",
          "--checkpoint", str(Path("ckpts") / config.with_suffix(".pt").name),
          "--data-root", str(root), "--manifest", str(root / "manifests/test.json"),
          "--official-test-allowlist", str(root / "official_test_allowlist.json"),
          "--output", "predictions", *sys.argv[1:]])
