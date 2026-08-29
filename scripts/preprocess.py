#!/usr/bin/env python3
"""Convert the already split raw CommonRoad files to compact torch scenario files."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.scenario import save_scenario


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    p.add_argument("--processed-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--limit-per-split", type=int, default=0, help="0 = all; useful for smoke tests")
    args = p.parse_args()
    raw_root, processed_root = Path(args.raw_root), Path(args.processed_root)
    for split in ("train", "val", "test"):
        src, dst = raw_root / split, processed_root / split
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.xml"))
        if args.limit_per_split > 0:
            files = files[:args.limit_per_split]
        for i, path in enumerate(files, 1):
            out = dst / f"{path.stem}.scenario.pt"
            save_scenario(path, out)
            print(f"{split} [{i}/{len(files)}] {path.name} -> {out.name}")

if __name__ == "__main__":
    main()
