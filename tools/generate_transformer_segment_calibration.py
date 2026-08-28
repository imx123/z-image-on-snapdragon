"""Generate raw calibration samples for one Transformer layer-group segment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=r"E:\zimage_calib_transformer_layers")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lines = []
    for i in range(args.samples):
        unified = rng.normal(0, 0.02, size=(1, 1536, 3840)).astype(np.float32)
        freqs = rng.uniform(-1, 1, size=(1, 1536, 128)).astype(np.float32)
        mask = np.ones((1, 1536), dtype=np.float32)
        adaln = rng.normal(0, 1, size=(1, 256)).astype(np.float32)
        names = []
        for key, arr in (("unified", unified), ("freqs", freqs), ("mask", mask), ("adaln", adaln)):
            p = out / f"sample_{i:02d}_{key}.raw"
            p.write_bytes(arr.tobytes()); names.append(str(p))
        lines.append(" ".join(names))
    (out / "input_list.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(out / "input_list.txt")

if __name__ == "__main__":
    main()
