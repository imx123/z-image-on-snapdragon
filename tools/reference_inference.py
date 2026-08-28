"""Run Z-Image-Turbo from a local ModelScope snapshot only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import ZImagePipeline

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--prompt", default="a red paper lantern on a rainy Shanghai street at night")
    parser.add_argument("--output", default="build/reference.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    root = Path(args.model_root).resolve()
    if not (root / "model_index.json").is_file():
        raise SystemExit(f"Missing model_index.json: {root}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    pipe = ZImagePipeline.from_pretrained(str(root), torch_dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=False)
    pipe.to("cpu")
    image = pipe(prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=9, guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output.resolve())

if __name__ == "__main__":
    main()
