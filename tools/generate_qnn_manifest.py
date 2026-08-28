"""Generate a JSON manifest of QNN graph I/O (type/name/shape/quant encoding).

Parses model_net.json emitted by qnn-onnx-converter. The resulting manifest is
the contract used by the Android JNI runtime to allocate buffers, quantize
inputs and dequantize outputs.

Data types follow QnnTypes.h:
  0x0108 UINT_8, 0x0104? ... printed symbolically where possible.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

QNN_DT = {
    0x0104: "INT_4", 0x0108: "UINT_8", 0x0110: "UINT_16", 0x0116: "INT_16",
    0x0120: "INT_32", 0x0128: "UINT_32", 0x0132: "UINT_64", 0x0140: "INT_8",
    0x0150: "INT_64", 0x0158: "UINT_8", 0x0164: "INT_4",
    0x0216: "FLOAT_16", 0x0232: "FLOAT_32", 0x0264: "FLOAT_64",
    0x0408: "UFIXED_POINT_8", 0x0410: "UFIXED_POINT_16", 0x0420: "UFIXED_POINT_32",
    0x0508: "BOOL_8",
    0x0040: "INT_64", 0x0064: "INT_64",
}

def tensor_entry(key, t):
    qp = t.get("quant_params", {})
    so = qp.get("scale_offset", {})
    return {
        "name": t.get("name") or key,
        "type": t.get("type"),
        "data_type": t.get("data_type"),
        "data_type_name": QNN_DT.get(t.get("data_type"), f"0x{t.get('data_type'):04X}"),
        "dims": t.get("dims"),
        "quant": {
            "definition": qp.get("definition"),
            "encoding": qp.get("encoding"),
            "scale": so.get("scale"),
            "offset": so.get("offset"),
            "bitwidth": so.get("bitwidth"),
            "is_symmetric": so.get("is_symmetric"),
        } if qp.get("definition", 2147483647) != 2147483647 else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qnn-dir", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = Path(args.qnn_dir) / args.segment / "model_net.json"
    d = json.loads(p.read_text())["graph"]["tensors"]
    ins = sorted([tensor_entry(k, t) for k, t in d.items() if t.get("type") == 0], key=lambda x: x["name"])
    outs = sorted([tensor_entry(k, t) for k, t in d.items() if t.get("type") == 1], key=lambda x: x["name"])
    result = {"segment": args.segment, "inputs": ins, "outputs": outs}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"{args.segment}: {len(ins)} inputs, {len(outs)} outputs -> {args.out}")

if __name__ == "__main__":
    main()
