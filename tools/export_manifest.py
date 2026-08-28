"""Create a deterministic graph-boundary manifest for Z-Image-Turbo export work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(model_dir: str) -> dict:
    return {
        "model": "Z-Image-Turbo",
        "device_target": "Snapdragon 8 Elite",
        "batch": 1,
        "width": 512,
        "height": 512,
        "steps": 8,
        "graphs": {
            "text_encoder": {
                "backend": "qnn_htp",
                "inputs": {"input_ids": [1, 512], "attention_mask": [1, 512]},
                "outputs": {"conditioning": [1, 512, 2560]},
            },
            "transformer": {
                "backend": "adreno_gpu",
                "precision": "fp16",
                "inputs": {
                    "latent": [1, 16, 64, 64],
                    "conditioning": [1, 512, 2560],
                    "timestep": [1],
                },
                "outputs": {"predicted_latent": [1, 16, 64, 64]},
            },
            "vae_decoder": {
                "backend": "adreno_gpu",
                "precision": "fp16",
                "inputs": {"latent": [1, 16, 64, 64]},
                "outputs": {"image": [1, 3, 512, 512]},
            },
        },
        "model_dir": str(Path(model_dir).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/z-image-turbo")
    parser.add_argument("--out", default="build/z-image-turbo-manifest.json")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(args.model_dir), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
