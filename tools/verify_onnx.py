"""Validate an exported ONNX graph and run one deterministic shape check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    path = Path(args.model).resolve()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 16, 64, 64), dtype=np.float32)})
    print({"inputs": [(x.name, x.shape, x.type) for x in session.get_inputs()], "outputs": [(x.name, x.shape, x.type) for x in session.get_outputs()], "result_shape": result[0].shape})

if __name__ == "__main__":
    main()
