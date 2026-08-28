"""Prepare fixed-shape local tokenizer inputs for QNN text-encoder calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--out", default="build/text-inputs.json")
    args = parser.parse_args()
    prompts = args.prompt or ["a red paper lantern on a rainy Shanghai street at night", "a studio portrait of a mountain climber"]
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_root) / "tokenizer"), local_files_only=True, use_fast=True)
    encoded = tokenizer(prompts, padding="max_length", truncation=True, max_length=512, return_tensors="np")
    result = {"sequence_length": 512, "prompts": prompts, "input_ids": encoded["input_ids"].tolist(), "attention_mask": encoded["attention_mask"].tolist()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())

if __name__ == "__main__":
    main()
