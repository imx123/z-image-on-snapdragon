"""Fold constant LessOrEqual nodes for QAIRT compatibility."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    graph = onnx.load(args.input, load_external_data=False)
    constants = {node.output[0]: numpy_helper.to_array(helper.get_attribute_value(node.attribute[0])) for node in graph.graph.node if node.op_type == "Constant"}
    folded = 0
    kept = []
    for node in graph.graph.node:
        if node.op_type == "LessOrEqual" and all(name in constants for name in node.input):
            value = np.less_equal(constants[node.input[0]], constants[node.input[1]])
            replacement = helper.make_node("Constant", [], node.output, value=numpy_helper.from_array(value.astype(np.bool_), name=node.output[0] + "_value"), name=node.name + "_folded")
            kept.append(replacement)
            folded += 1
        else:
            kept.append(node)
    graph.graph.ClearField("node")
    graph.graph.node.extend(kept)
    onnx.save_model(graph, args.output, save_as_external_data=False)
    print({"folded_less_or_equal": folded, "output": str(Path(args.output).resolve())})

if __name__ == "__main__":
    main()
