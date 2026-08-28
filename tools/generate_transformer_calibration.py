"""Generate raw calibration samples for Transformer QNN quantization.

QAIRT reads one line per sample; each line lists raw files for all graph
inputs, space separated. Keep the output outside the workspace path because
QAIRT tools have had issues with paths containing spaces.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=r"E:\zimage_calib_transformer")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cap-length", type=int, default=512)
    parser.add_argument("--include-cap-mask", action="store_true",
                        help="generate the 4-input cap_mask variant as well")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lines = []

    for i in range(args.samples):
        latent = rng.normal(0.0, 1.0, size=(1, 16, 64, 64)).astype(np.float32)
        timestep = rng.uniform(0.0, 1.0, size=(1,)).astype(np.float32)
        cap = rng.normal(0.0, 1.0, size=(args.cap_length, 2560)).astype(np.float32)
        latent_p = out / f"sample_{i:02d}_latent.raw"
        t_p = out / f"sample_{i:02d}_timestep.raw"
        cap_p = out / f"sample_{i:02d}_cap_feats.raw"
        latent_p.write_bytes(latent.tobytes())
        t_p.write_bytes(timestep.tobytes())
        cap_p.write_bytes(cap.tobytes())
        parts = [str(latent_p), str(t_p), str(cap_p)]
        if args.include_cap_mask:
            valid = rng.integers(32, args.cap_length, endpoint=True)
            mask = np.zeros((args.cap_length,), dtype=np.float32)
            mask[:valid] = 1.0
            mask_p = out / f"sample_{i:02d}_cap_mask.raw"
            mask_p.write_bytes(mask.tobytes())
            parts.append(str(mask_p))
        lines.append(" ".join(parts))

    list_p = out / ("input_list_4.txt" if args.include_cap_mask else "input_list_3.txt")
    list_p.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(list_p)


if __name__ == "__main__":
    main()
