#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from model.batching import batch_graph_samples
from model.config import BATCH_SIZE, NUM_WORKERS
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["boston", "pittsburgh", "singapore"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.city / "processed"
    dataset = CommonRoadTemporalGraphDataset(data_root / "test")
    n = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    subset = Subset(dataset, list(range(n)))
    loader_kwargs = dict(
        dataset=subset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.workers),
        collate_fn=batch_graph_samples,
        pin_memory=torch.cuda.is_available(),
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrGeoTrajectoryPredictionModel(**checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    total_ade = 0.0
    total_fde = 0.0
    total_targets = 0
    sample_counter = 0

    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch)["position"]
            target = batch["target_position"].to(device)
            mask = batch["target_mask"].to(device)
            metrics = ade_fde(prediction, target, mask)
            if metrics["count"] == 0:
                continue
            total_ade += metrics["ade"] * metrics["count"]
            total_fde += metrics["fde"] * metrics["count"]
            total_targets += metrics["count"]
            for meta in batch["meta"]:
                rows.append({"sample": sample_counter, "benchmark_id": meta["benchmark_id"]})
                sample_counter += 1

    if total_targets == 0:
        raise RuntimeError("No valid test targets")

    summary = {
        "city": args.city,
        "test_windows": n,
        "test_targets": total_targets,
        "mean_ADE_m": total_ade / total_targets,
        "mean_FDE_m": total_fde / total_targets,
    }
    print(json.dumps(summary, indent=2))

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample", "benchmark_id"])
            writer.writeheader()
            writer.writerows(rows)
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"csv={output}")


if __name__ == "__main__":
    main()
