#!/usr/bin/env python3
"""Evaluate a trained paper-aligned model using ADE and FDE."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde, trajectory_error_sums


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-csv", default="outputs/test_metrics.csv")
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset_cfg = ckpt.get("dataset", {})
    ds = CommonRoadTemporalGraphDataset(
        Path(args.data_root) / args.split,
        obs_steps=int(dataset_cfg.get("obs_steps", 15)),
        pred_steps=int(dataset_cfg.get("pred_steps", 5)),
        model_dt=float(dataset_cfg.get("model_dt", 0.2)),
        stride=int(dataset_cfg.get("stride", 1)),
    )
    model = CrGeoTrajectoryPredictionModel(**ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows: list[dict] = []
    ade_sum = fde_sum = 0.0
    ade_count = fde_count = 0
    limit = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)

    with torch.no_grad():
        for i in range(limit):
            sample = ds[i]
            output = model(sample)
            target = sample["target_position"].to(device)
            mask = sample["target_mask"].to(device)
            ade, fde = ade_fde(output["position"], target, mask)
            a_sum, f_sum, a_count, f_count = trajectory_error_sums(output["position"], target, mask)
            ade_sum += float(a_sum)
            fde_sum += float(f_sum)
            ade_count += a_count
            fde_count += f_count
            rows.append({
                "benchmark_id": sample["meta"]["benchmark_id"],
                "location_group": sample["meta"]["location_group"],
                "start_time_step": sample["meta"]["observation_time_steps"][0],
                "target_vehicles": int(mask.sum()),
                "ADE_m": float(ade),
                "FDE_m": float(fde),
            })

    summary = {
        "split": args.split,
        "samples": len(rows),
        "mean_ADE_m": ade_sum / max(ade_count, 1),
        "mean_FDE_m": fde_sum / max(fde_count, 1),
        "valid_prediction_points": ade_count,
        "valid_trajectories": fde_count,
    }

    out = Path(args.output_csv)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
