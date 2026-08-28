"""Generate QAIRT calibration input_list for all six Qwen3 text-encoder segments.

Runs the real tokenizer + Qwen3 forward pass on a handful of prompts, chains
the hidden states segment by segment (embedding -> layers_00_05 -> ... ->
layers_30_35 -> final_norm), and writes per-segment raw input files plus an
input_list.txt in the layout QAIRT's qnn-onnx-converter expects:
  one line per calibration sample, each line a raw binary file path containing
  ALL graph inputs concatenated in input order (hidden_states, attention_mask,
  cos, sin), float32 little-endian.

Usage (run with the isolated QAIRT Python so hermes' venv doesn't leak in):
  E:/projects/zimage on phone/zimage-runtime/python312/python.exe -I tools/generate_qwen3_calibration.py \
      --model-root E:/.../snapshots/master --out build/qwen3_calibration_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Model


def make_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build the fixed [batch, 1, seq, seq] additive mask expected by Qwen3."""
    batch, sequence = attention_mask.shape
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((batch, 1, sequence, sequence), neg_inf, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    valid_keys = attention_mask[:, None, None, :].to(torch.bool)
    return mask.masked_fill(~valid_keys, neg_inf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", default="build/qwen3_calibration_v2")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    sequence = args.sequence_length
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.model_root).resolve()

    # Keep everything on CPU; the QAIRT runtime is CPU-only.
    torch.set_num_threads(max(4, torch.get_num_threads()))
    # Load in fp16: calibration only needs activation ranges for min-max
    # quantization; fp16 halves weight memory (~8 GB vs 16 GB) and avoids the
    # OOM crash seen with fp32 on this host. Raw calibration files are still
    # written as float32 (QAIRT expects fp32 raw inputs).
    model = Qwen3Model.from_pretrained(
        str(root / "text_encoder"),
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        str(root / "tokenizer"), local_files_only=True, use_fast=True
    )

    prompts = [
        "a red paper lantern on a rainy Shanghai street at night",
        "a studio portrait of a mountain climber, dramatic lighting",
        "an oil painting of a quiet village in autumn, golden leaves",
        "a futuristic city skyline at dusk, neon reflections on water",
        "a cute corgi puppy sitting in a field of sunflowers",
        "an underwater scene with colorful coral and tropical fish",
        "a cozy cabin interior with a warm fireplace in winter",
        "abstract geometric shapes in pastel colors, minimal art",
    ][: args.samples]
    if len(prompts) < args.samples:
        raise ValueError(f"need {args.samples} prompts, only {len(prompts)} defined")

    enc = tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=sequence, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention = enc["attention_mask"]
    batch = input_ids.shape[0]
    position_ids = torch.arange(sequence, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        hidden = model.embed_tokens(input_ids)          # [B, 512, 2560]
        # Build the mask in fp32 (raw calibration files are fp32); fp16 min
        # (-65504) would distort min-max activation ranges.
        mask = make_causal_mask(attention, torch.float32)
        cos, sin = model.rotary_emb(hidden, position_ids)

    # Input order per layer-group ONNX graph:
    #   hidden_states [B,512,2560], attention_mask [B,1,512,512],
    #   cos [B,512,128], sin [B,512,128]
    hidden_bytes = sequence * model.config.hidden_size
    mask_bytes = sequence * sequence
    rotary_bytes = sequence * model.config.head_dim
    per_sample = 4 * (hidden_bytes + mask_bytes + 2 * rotary_bytes)

    manifest = {"sequence_length": sequence, "samples": len(prompts),
                "per_sample_bytes": per_sample, "prompts": prompts,
                "segments": {}}

    def write_raw(path: Path, tensor: torch.Tensor) -> None:
        path.write_bytes(tensor.detach().contiguous().cpu().numpy().astype("<f4").tobytes())

    # Chain hidden states through the segments exactly like runtime inference.
    # QAIRT input_list format: one line per calibration sample, each line lists
    # the raw file paths for ALL graph inputs separated by spaces (one file per
    # input, in graph input order). So per sample we write 4 separate raw files:
    # hidden_states.raw attention_mask.raw cos.raw sin.raw
    segments = ["layers_00_05", "layers_06_11", "layers_12_17",
                "layers_18_23", "layers_24_29", "layers_30_35"]
    for seg in segments:
        seg_dir = out_dir / seg
        seg_dir.mkdir(parents=True, exist_ok=True)
        inputs_list: list[str] = []
        start, end = int(seg.split("_")[1]), int(seg.split("_")[2])
        for i in range(len(prompts)):
            names = [f"sample_{i:02d}_hidden.raw", f"sample_{i:02d}_mask.raw",
                     f"sample_{i:02d}_cos.raw", f"sample_{i:02d}_sin.raw"]
            write_raw(seg_dir / names[0], hidden[i : i + 1])
            write_raw(seg_dir / names[1], mask[i : i + 1])
            write_raw(seg_dir / names[2], cos[i : i + 1])
            write_raw(seg_dir / names[3], sin[i : i + 1])
            inputs_list.append(" ".join(str(seg_dir / n) for n in names))
        (seg_dir / "input_list.txt").write_text("\n".join(inputs_list) + "\n", encoding="utf-8")
        # Run this segment forward to produce next segment's hidden states.
        out_hidden = []
        for i in range(len(prompts)):
            h = hidden[i : i + 1]
            # Cast the fp32 mask to the model's compute dtype per-sample.
            mask_i = mask[i : i + 1].to(h.dtype)
            for layer in model.layers[start : end + 1]:
                result = layer(
                    h,
                    attention_mask=mask_i,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=(cos[i : i + 1], sin[i : i + 1]),
                )
                h = result[0]
            out_hidden.append(h)
        hidden = torch.cat(out_hidden, dim=0)
        manifest["segments"][seg] = {
            "files": [f"sample_{i:02d}_*.raw" for i in range(len(prompts))],
            "input_list": f"{seg}/input_list.txt",
        }
        print(f"{seg}: {hidden.shape} hidden after forward, "
              f"{len(inputs_list)} samples, {per_sample} bytes/sample")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Calibration root: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
