"""Download Z-Image-Turbo from ModelScope only."""

from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--root", default="models")
    args = parser.parse_args()
    if args.model != "Tongyi-MAI/Z-Image-Turbo":
        raise SystemExit("This downloader is restricted to the ModelScope Z-Image-Turbo model.")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the ModelScope SDK first: pip install modelscope") from exc
    target = Path(args.root).resolve() / "z-image-turbo"
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.model, revision=args.revision, local_dir=str(target))
    print(path)

if __name__ == "__main__":
    main()
