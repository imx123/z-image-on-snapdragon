"""Inspect a local ModelScope snapshot without loading tensors into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

def inspect_file(path: Path) -> dict:
    tensors = 0
    params = 0
    dtypes: dict[str, int] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = handle.get_slice(name).get_shape()
            count = 1
            for dim in shape: count *= dim
            tensors += 1
            params += count
            dtype = str(handle.get_tensor(name).dtype).replace("torch.", "")
            dtypes[dtype] = dtypes.get(dtype, 0) + count
    return {"file": str(path), "bytes": path.stat().st_size, "tensors": tensors, "parameters": params, "dtypes": dtypes}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", default="build/model-inventory.json")
    args = parser.parse_args()
    root = Path(args.model_root).resolve()
    files = sorted(root.rglob("*.safetensors"))
    inventory = {"root": str(root), "files": [inspect_file(path) for path in files]}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2))

if __name__ == "__main__":
    main()
