#!/usr/bin/env python
"""Prepare one Kaggle NuPlan/CommonRoad city at a time.

Input expected from the Zenodo cr-geo graph source archive, as mounted on Kaggle:
  /kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata
  /kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata
  /kaggle/input/datasets/abdullge26z811/singapore[/singapore_t0.2_cleaneddata]

The source files in these archives are CommonRoad XML scenarios.  This script
creates a deterministic same-city scenario-level 70/15/15 split, then converts
XML once into compact .scenario.pt files for fast training.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import SEED, TRAIN_RATIO, VAL_RATIO
from model.scenario import save_scenario

EXPECTED_PAPER_SCENARIOS = {"boston": 938, "pittsburgh": 1560, "singapore": 2372}

KAGGLE_CANDIDATES = {
    "boston": [
        Path("/kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata"),
        Path("/kaggle/input/datasets/abdullge26z811/boston"),
    ],
    "pittsburgh": [
        Path("/kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata"),
        Path("/kaggle/input/datasets/abdullge26z811/pittsburgh"),
    ],
    "singapore": [
        Path("/kaggle/input/datasets/abdullge26z811/singapore/singapore_t0.2_cleaneddata"),
        Path("/kaggle/input/datasets/abdullge26z811/singapore"),
    ],
}


def find_source_dir(city: str, explicit: str | None = None) -> Path:
    candidates = [Path(explicit)] if explicit else KAGGLE_CANDIDATES[city]
    diagnostics = []
    for candidate in candidates:
        if not candidate.exists():
            diagnostics.append(f"missing: {candidate}")
            continue
        direct = sorted(candidate.glob("*.xml"))
        if direct:
            return candidate
        # Dataset may contain one nested *_t0.2_cleaneddata folder.
        nested = sorted(candidate.rglob("*.xml"))
        if nested:
            # Use their common parent when possible; files are still returned recursively later.
            return candidate
        diagnostics.append(f"exists but contains no .xml: {candidate}")
    raise FileNotFoundError(
        f"Could not locate {city} CommonRoad XML files. Checked:\n  " + "\n  ".join(diagnostics)
    )


def list_xmls(source: Path) -> list[Path]:
    files = sorted(source.glob("*.xml"))
    if not files:
        files = sorted(source.rglob("*.xml"))
    return files


def split_files(files: list[Path], seed: int):
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # Put rounding remainder into test so every scenario appears exactly once.
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train+n_val],
        "test": shuffled[n_train+n_val:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=sorted(KAGGLE_CANDIDATES))
    ap.add_argument("--source", default=None, help="Optional explicit source folder. Normally not needed on Kaggle.")
    ap.add_argument("--output-root", default=None, help="Default: data/<city>/processed")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-scenarios", type=int, default=0, help="Smoke-test only; 0 means all scenarios.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    city = args.city.lower()
    source = find_source_dir(city, args.source)
    files = list_xmls(source)
    if args.max_scenarios > 0:
        files = files[:args.max_scenarios]
    if not files:
        raise RuntimeError(f"No XML files found under {source}")

    expected = EXPECTED_PAPER_SCENARIOS[city]
    print(f"city={city}")
    print(f"source={source}")
    print(f"found_xml={len(files)}")
    if args.max_scenarios == 0 and len(files) != expected:
        print(f"WARNING: paper reports {expected} {city.title()} scenarios, but this Kaggle mount contains {len(files)} XML files.")
        print("Training will use the files actually present. Check the dataset upload if you expected the paper count.")

    out_root = Path(args.output_root) if args.output_root else ROOT / "data" / city / "processed"
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    splits = split_files(files, args.seed)
    manifest_rows = []
    failures = []
    for split, split_files_ in splits.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{split}: {len(split_files_)} scenarios")
        for i, xml in enumerate(split_files_, 1):
            # Preserve stem but avoid collisions from nested directories.
            target = split_dir / f"{xml.stem}.scenario.pt"
            try:
                if args.overwrite or not target.exists():
                    save_scenario(xml, target)
                manifest_rows.append({"city": city, "split": split, "source": str(xml), "processed": str(target)})
                if i == 1 or i % 100 == 0 or i == len(split_files_):
                    print(f"  [{i}/{len(split_files_)}] {xml.name}")
            except Exception as exc:
                failures.append({"split": split, "source": str(xml), "error": repr(exc)})
                print(f"  FAILED {xml.name}: {exc}")

    manifest = out_root.parent / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["city", "split", "source", "processed"])
        w.writeheader(); w.writerows(manifest_rows)

    summary = {
        "city": city,
        "source": str(source),
        "seed": args.seed,
        "paper_reported_scenarios": expected,
        "found_xml": len(files),
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "processed": {s: sum(1 for r in manifest_rows if r["split"] == s) for s in ("train", "val", "test")},
        "failures": failures,
    }
    summary_path = out_root.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE")
    print(json.dumps(summary, indent=2))
    print(f"manifest={manifest}")
    print(f"summary={summary_path}")
    if failures:
        raise SystemExit(f"{len(failures)} scenario(s) failed preprocessing; inspect summary.json before training.")

if __name__ == "__main__":
    main()
