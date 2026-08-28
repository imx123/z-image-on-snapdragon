"""Inspect ONNX initializers and external weight data."""
from __future__ import annotations
import argparse
from pathlib import Path
import onnx

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    path = Path(args.model).resolve()
    graph = onnx.load(str(path), load_external_data=False)
    external = sum(1 for tensor in graph.graph.initializer if tensor.external_data)
    inline = sum(len(tensor.raw_data) for tensor in graph.graph.initializer)
    print({"initializers": len(graph.graph.initializer), "nodes": len(graph.graph.node), "external_initializers": external, "inline_bytes": inline, "sidecar": str(path) + ".data", "sidecar_exists": Path(str(path) + ".data").is_file()})
