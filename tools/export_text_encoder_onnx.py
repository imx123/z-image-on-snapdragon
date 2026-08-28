"""Export local ModelScope Qwen3 text encoder to a fixed-shape ONNX graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import Qwen3Model

class TextEncoder(torch.nn.Module):
    def __init__(self, model: Qwen3Model):
        super().__init__()
        self.model = model
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=False)[0]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", default="build/text_encoder.onnx")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(args.model_root).resolve()
    model = Qwen3Model.from_pretrained(str(root / "text_encoder"), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    model.config.use_cache = False
    model.eval()
    wrapper = TextEncoder(model).eval()
    input_ids = torch.ones((1, args.sequence_length), dtype=torch.long)
    attention_mask = torch.ones((1, args.sequence_length), dtype=torch.long)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (input_ids, attention_mask), str(output), input_names=["input_ids", "attention_mask"], output_names=["conditioning"], opset_version=args.opset, dynamo=False, do_constant_folding=True, external_data=True)
    import onnx
    graph = onnx.load(str(output), load_external_data=False)
    onnx.save_model(graph, str(output), save_as_external_data=True, all_tensors_to_one_file=True, location=output.name + ".data", size_threshold=1024, convert_attribute=False)
    print(output.resolve())

if __name__ == "__main__":
    main()
