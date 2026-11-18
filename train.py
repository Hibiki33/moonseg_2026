"""Reproduce the final six-scene training run using the organizer data."""
import sys

from moonseg.train import main

if __name__ == "__main__":
    main(["--config", "configs/da3mono_vitl14_finetuned_e25.yaml",
          "--official-fulltrain-root", "data/LuSNAR_official",
          "--backbone-weights", "pretrained/da3mono-large/model.safetensors", *sys.argv[1:]])
