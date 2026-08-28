"""Generate chain-consistent calibration data for the segmented Transformer.

The earlier per-segment calibration fed every 6-layer group independent
random inputs (std 0.02). Real activations at later segment boundaries are
orders of magnitude larger, so independent calibration produces mismatched
input/output encodings at every boundary. This tool runs the FP16 ONNX
segments in sequence on the CPU and saves, for each segment and sample, the
exact inputs produced by the previous segment (and the exact outputs for the
next one).

Outputs:
  E:\zimage_calib_transformer_chain\layers_00_05\input_list.txt ...
  E:\zimage_calib_transformer_chain\final\input_list.txt
"""
from __future__ import annotations
import argparse, json, glob, os, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

GROUPS = ["layers_00_05", "layers_06_11", "layers_12_17", "layers_18_23", "layers_24_29"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments-dir", default="build/transformer_segments_v2")
    ap.add_argument("--frontend-cal", default=r"E:\zimage_calib_transformer_frontend")
    ap.add_argument("--out-root", default=r"E:\zimage_calib_transformer_chain")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--samples", type=int, default=0, help="0 = all samples in frontend calibration")
    args = ap.parse_args()

    fcal = Path(args.frontend_cal)
    sample_ids = sorted({p.stem.split("_")[1] for p in fcal.glob("sample_*_latent.raw")})
    if args.samples:
        sample_ids = sample_ids[: args.samples]
    if not sample_ids:
        raise SystemExit("no frontend calibration samples found")
    print("samples:", sample_ids, flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.log_severity_level = 3

    # 1) Frontend -> per-sample unified/freqs/mask/adaln
    front_path = Path(args.segments_dir) / "frontend" / "model.onnx"
    print("load frontend", flush=True)
    front = ort.InferenceSession(str(front_path), so, providers=["CPUExecutionProvider"])
    chains = []  # dict per sample
    for sid in sample_ids:
        t = time.time()
        latent = np.fromfile(str(fcal / f"sample_{sid}_latent.raw"), dtype=np.float32).reshape(1, 16, 64, 64)
        timestep = np.fromfile(str(fcal / f"sample_{sid}_timestep.raw"), dtype=np.float32).reshape(1)
        cap_feats = np.fromfile(str(fcal / f"sample_{sid}_cap_feats.raw"), dtype=np.float32).reshape(512, 2560)
        cap_mask = np.fromfile(str(fcal / f"sample_{sid}_cap_mask.raw"), dtype=np.float32).reshape(512)
        res = front.run(["unified", "unified_freqs", "unified_mask", "adaln_input"], {
            "latent": latent.astype(np.float16),
            "timestep": timestep.astype(np.float16),
            "cap_feats": cap_feats.astype(np.float16),
            "cap_mask": cap_mask.astype(bool),
        })
        chains.append({"id": sid, "unified": res[0].astype(np.float32),
                       "freqs": res[1].astype(np.float32), "mask": res[2].astype(np.float32),
                       "adaln": res[3].astype(np.float32)})
        print(f"frontend {sid}: {time.time()-t:.2f}s unified "
              f"[{float(chains[-1]['unified'].min()):.3f},{float(chains[-1]['unified'].max()):.3f}]", flush=True)

    def write_cal(dirname: str, names, arrays):
        d = out_root / dirname
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for c in chains:
            parts = []
            for name, arr in zip(names, arrays):
                p = d / f"sample_{c['id']}_{name}.raw"
                data = c[name] if name in c else arr(c)
                data.astype(np.float32).tofile(p)
                parts.append(str(p))
            lines.append(" ".join(parts))
        (d / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        print("wrote", d / "input_list.txt")

    # 2) Layer groups in sequence
    ranges = {}
    prev = None
    for g in GROUPS:
        sess = ort.InferenceSession(str(Path(args.segments_dir) / g / "model.onnx"), so,
                                    providers=["CPUExecutionProvider"])
        d = out_root / g
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        outs = []
        for c in chains:
            unified_in = c["unified"] if prev is None else prev[c["id"]]
            t = time.time()
            out = sess.run(["unified_out"], {
                "unified_in": unified_in.astype(np.float16),
                "unified_freqs": c["freqs"],
                "unified_mask": c["mask"],
                "adaln_input": c["adaln"].astype(np.float16),
            })[0].astype(np.float32)
            parts = []
            for name, arr in [("unified", unified_in), ("freqs", c["freqs"]), ("mask", c["mask"]), ("adaln", c["adaln"])]:
                p = d / f"sample_{c['id']}_{name}.raw"
                arr.astype(np.float32).tofile(p)
                parts.append(str(p))
            lines.append(" ".join(parts))
            outs.append((c["id"], out))
            print(f"{g} {c['id']}: {time.time()-t:.2f}s out "
                  f"[{float(out.min()):.2f},{float(out.max()):.2f}]", flush=True)
        (d / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        for sid, out in outs:
            (d / f"sample_{sid}_unified_out.raw").write_bytes(out.tobytes())
        ranges[g] = {"min": float(min(o.min() for _, o in outs)),
                     "max": float(max(o.max() for _, o in outs)),
                     "std": float(np.concatenate([o.ravel() for _, o in outs]).std())}
        prev = {sid: out for sid, out in outs}

    # 3) Final segment calibration from the last group's output
    final_sess = ort.InferenceSession(str(Path(args.segments_dir) / "final" / "model.onnx"), so,
                                      providers=["CPUExecutionProvider"])
    d = out_root / "final"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    final_ranges = {}
    for c in chains:
        unified_in = prev[c["id"]]
        p_u = d / f"sample_{c['id']}_unified.raw"
        p_a = d / f"sample_{c['id']}_adaln.raw"
        unified_in.tofile(p_u)
        c["adaln"].tofile(p_a)
        lines.append(f"{p_u} {p_a}")
        out = final_sess.run(["noise_pred"], {"unified_in": unified_in.astype(np.float16),
                                              "adaln_input": c["adaln"].astype(np.float16)})[0].astype(np.float32)
        (d / f"sample_{c['id']}_noise_pred.raw").write_bytes(out.tobytes())
        final_ranges[c["id"]] = [float(out.min()), float(out.max())]
        print(f"final {c['id']}: out [{out.min():.3f},{out.max():.3f}]", flush=True)
    (d / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    ranges["final"] = {"by_sample": final_ranges}
    (out_root / "segment_ranges.json").write_text(json.dumps(ranges, indent=2))
    print("DONE", out_root)

if __name__ == "__main__":
    main()
