"""Export the local Qwen3 text encoder as fixed-shape layer-group ONNX graphs.

The graphs deliberately share a small interface: embedding consumes token ids,
layer groups consume hidden states plus a prepared causal mask and RoPE tensors,
and the final graph applies the model norm. Tokenization and mask preparation stay
on CPU so the same artifacts can be used by QNN or a CPU fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import Qwen3Model


class EmbeddingSegment(torch.nn.Module):
    def __init__(self, model: Qwen3Model):
        super().__init__()
        self.embedding = model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LayerGroupSegment(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = (cos, sin)
        for layer in self.layers:
            result = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = result[0]
        return hidden_states


class FinalNormSegment(torch.nn.Module):
    def __init__(self, model: Qwen3Model):
        super().__init__()
        self.norm = model.norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def make_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build the fixed [batch, 1, seq, seq] additive mask expected by Qwen3."""
    batch, sequence = attention_mask.shape
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((batch, 1, sequence, sequence), neg_inf, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    valid_keys = attention_mask[:, None, None, :].to(torch.bool)
    return mask.masked_fill(~valid_keys, neg_inf)


def export_graph(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, names: list[str], outputs: list[str], opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(output),
        input_names=names,
        output_names=outputs,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        external_data=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out-dir", default="build/text_encoder_segments")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--layers-per-segment", type=int, default=6)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(args.model_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    model = Qwen3Model.from_pretrained(
        str(root / "text_encoder"),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    model.config.use_cache = False
    if args.layers_per_segment < 1:
        raise ValueError("--layers-per-segment must be positive")
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    sequence = args.sequence_length
    input_ids = torch.ones((1, sequence), dtype=torch.long)
    attention = torch.ones((1, sequence), dtype=torch.long)
    position_ids = torch.arange(sequence, dtype=torch.long).unsqueeze(0)
    hidden = model.embed_tokens(input_ids)
    mask = make_causal_mask(attention, hidden.dtype)
    with torch.no_grad():
        rotary = model.rotary_emb(hidden, position_ids)
    cos, sin = rotary

    embedding_file = Path("embedding") / "model.onnx"
    export_graph(
        EmbeddingSegment(model), (input_ids,), out_dir / embedding_file,
        ["input_ids"], ["hidden_states"], args.opset,
    )
    layer_count = len(model.layers)
    segments = []
    for start in range(0, layer_count, args.layers_per_segment):
        end = min(start + args.layers_per_segment, layer_count)
        name = Path(f"layers_{start:02d}_{end - 1:02d}") / "model.onnx"
        export_graph(
            LayerGroupSegment(torch.nn.ModuleList(model.layers[start:end])),
            (hidden, mask, cos, sin), out_dir / name,
            ["hidden_states", "attention_mask", "cos", "sin"],
            ["hidden_states_out"], args.opset,
        )
        segments.append({"start": start, "end": end, "file": name.as_posix()})
    final_file = Path("final_norm") / "model.onnx"
    export_graph(
        FinalNormSegment(model), (hidden,), out_dir / final_file,
        ["hidden_states"], ["conditioning"], args.opset,
    )
    manifest = {
        "sequence_length": sequence,
        "hidden_size": model.config.hidden_size,
        "num_layers": layer_count,
        "layers_per_segment": args.layers_per_segment,
        "inputs": {"embedding": [1, sequence], "layer_group": [1, sequence, model.config.hidden_size]},
        "attention_mask": [1, 1, sequence, sequence],
        "rotary": [1, sequence, model.config.head_dim],
        "opset": args.opset,
        "dtype": str(hidden.dtype).replace("torch.", ""),
        "segments": segments,
        "embedding": embedding_file.as_posix(),
        "final": final_file.as_posix(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
