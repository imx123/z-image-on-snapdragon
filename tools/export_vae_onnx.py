"""Export the local ModelScope VAE decoder to ONNX without remote access."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL

class Decoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # ZImagePipeline does:
        #   latents = (latents / scaling_factor) + shift_factor
        # before VAE decode. The previous export missed shift_factor.
        scaled = latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.vae.decode(scaled, return_dict=False)[0]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", default="build/vae_decoder.onnx")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(args.model_root).resolve()
    vae = AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=torch.float32, local_files_only=True)
    vae.eval()
    wrapper = Decoder(vae).eval()
    latent = torch.zeros((1, 16, 64, 64), dtype=torch.float32)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (latent,), str(output), input_names=["latent"], output_names=["image"], opset_version=18, dynamo=False)
    print(output.resolve())

if __name__ == "__main__":
    main()
