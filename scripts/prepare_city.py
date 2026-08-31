#!/usr/bin/env python
"""Prepare one NuPlan/CommonRoad city for the paper-only experiment.

The script mirrors the paper's dataset-creation concept: CommonRoad scenario ->
persistent graph-ready dataset. It performs expensive geometry once, writes
compact .scenario.pt files, and writes sample_index.pt so training starts fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.config import SEED, TRAIN_RATIO, VAL_RATIO
from model.indexing import build_sample_records
from model.scenario import parse_commonroad_xml

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
        if next(candidate.rglob("*.xml"), None) is not None:
            return candidate
        diagnostics.append(f"exists but contains no .xml: {candidate}")
    raise FileNotFoundError(
        "Could not locate CommonRoad XML files. Checked:\n  " + "\n  ".join(diagnostics)
    )


def list_xmls(source: Path) -> list[Path]:
    direct = sorted(source.glob("*.xml"))
    return direct if direct else sorted(source.rglob("*.xml"))


def split_files(files: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _process_one(job):
    xml_path, target_path = job
    scenario = parse_commonroad_xml(xml_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scenario, target)
    records = build_sample_records(scenario, target)
    return str(xml_path), str(target), records, scenario.get("benchmark_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=sorted(KAGGLE_CANDIDATES))
    parser.add_argument("--source", default=None, help="Optional explicit source directory")
    parser.add_argument("--output-root", default=None, help="Default: data/<city>/processed")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-scenarios", type=int, default=0, help="Smoke-test only; 0 = all")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="Parallel XML preprocessing workers",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    city = args.city.lower()
    source = find_source_dir(city, args.source)
    files = list_xmls(source)
    if args.max_scenarios > 0:
        files = files[: args.max_scenarios]
    if not files:
        raise RuntimeError(f"No XML files found under {source}")

    expected = EXPECTED_PAPER_SCENARIOS[city]
    print(f"city={city}")
    print(f"source={source}")
    print(f"found_xml={len(files)}")
    print(f"workers={args.workers}")
    if args.max_scenarios == 0 and len(files) != expected:
        print(
            f"WARNING: the paper reports {expected} {city.title()} scenarios, "
            f"but this Kaggle mount contains {len(files)} XML files."
        )

    output_root = Path(args.output_root) if args.output_root else ROOT / "data" / city / "processed"
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = split_files(files, args.seed)
    manifest_rows = []
    failures = []
    start_time = time.time()

    for split, split_files_ in splits.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(str(xml), str(split_dir / f"{xml.stem}.scenario.pt")) for xml in split_files_]
        sample_records = []
        print(f"\n{split}: {len(jobs)} scenarios")

        if args.workers <= 1:
            for i, job in enumerate(jobs, 1):
                try:
                    source_path, processed_path, records, _ = _process_one(job)
                    sample_records.extend(records)
                    manifest_rows.append({
                        "city": city, "split": split, "source": source_path, "processed": processed_path
                    })
                except Exception as exc:
                    failures.append({"split": split, "source": job[0], "error": repr(exc)})
                    print(f"  FAILED {Path(job[0]).name}: {exc}")
                if i == 1 or i % 25 == 0 or i == len(jobs):
                    print(
                        f"  [{i}/{len(jobs)}] elapsed={(time.time() - start_time) / 60.0:.1f} min",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_one, job): job for job in jobs}
                for i, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    try:
                        source_path, processed_path, records, _ = future.result()
                        sample_records.extend(records)
                        manifest_rows.append({
                            "city": city, "split": split, "source": source_path, "processed": processed_path
                        })
                    except Exception as exc:
                        failures.append({"split": split, "source": job[0], "error": repr(exc)})
                        print(f"  FAILED {Path(job[0]).name}: {exc}")
                    if i == 1 or i % 25 == 0 or i == len(jobs):
                        print(
                            f"  [{i}/{len(jobs)}] elapsed={(time.time() - start_time) / 60.0:.1f} min",
                            flush=True,
                        )

        # Deterministic sample order makes evaluation reproducible.
        sample_records.sort(key=lambda r: (Path(r["scenario_file"]).name, tuple(r["times"])))
        torch.save(sample_records, split_dir / "sample_index.pt")
        print(f"  samples={len(sample_records)} index={split_dir / 'sample_index.pt'}")

    manifest = output_root.parent / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["city", "split", "source", "processed"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "city": city,
        "source": str(source),
        "seed": args.seed,
        "paper_reported_scenarios": expected,
        "found_xml": len(files),
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": 1.0 - TRAIN_RATIO - VAL_RATIO,
        },
        "processed": {
            split: sum(1 for row in manifest_rows if row["split"] == split)
            for split in ("train", "val", "test")
        },
        "failures": failures,
        "minutes": (time.time() - start_time) / 60.0,
    }
    summary_path = output_root.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDONE")
    print(json.dumps(summary, indent=2))
    print(f"manifest={manifest}")
    print(f"summary={summary_path}")
    if failures:
        raise SystemExit(f"{len(failures)} scenario(s) failed preprocessing; inspect {summary_path}")


if __name__ == "__main__":
    main()
