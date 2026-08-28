"""Consolidate ONNX tensor external data into one sidecar file."""
from __future__ import annotations
import argparse
from pathlib import Path
import onnx
from onnx.external_data_helper import convert_model_to_external_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    model = onnx.load(args.input, load_external_data=True)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=output.name + ".data", size_threshold=0, convert_attribute=False)
    onnx.save_model(model, str(output), save_as_external_data=True, all_tensors_to_one_file=True, location=output.name + ".data", size_threshold=0, convert_attribute=False)
    print(output)

if __name__ == "__main__":
    main()
