"""Validate the manifest and fixed interfaces of exported Qwen3 segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence = manifest["sequence_length"]
    hidden = manifest["hidden_size"]
    if manifest["inputs"] != {"embedding": [1, sequence], "layer_group": [1, sequence, hidden]}:
        raise ValueError("manifest input shapes are inconsistent")
    if manifest["attention_mask"] != [1, 1, sequence, sequence]:
        raise ValueError("manifest attention mask shape is inconsistent")
    segments = manifest["segments"]
    if not segments or segments[0]["start"] != 0 or segments[-1]["end"] != manifest["num_layers"]:
        raise ValueError("layer segments do not cover the model")
    for previous, current in zip(segments, segments[1:]):
        if previous["end"] != current["start"]:
            raise ValueError("layer segments are not contiguous")
    files = [manifest["embedding"], manifest["final"], *[segment["file"] for segment in segments]]
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing segment files: " + ", ".join(missing))
    print({"segments": len(segments), "layers": manifest["num_layers"], "sequence_length": sequence, "status": "ok"})


if __name__ == "__main__":
    main()
