#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["boston", "pittsburgh", "singapore"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / args.city / "processed"
    dataset = CommonRoadTemporalGraphDataset(data_root / "test")
    sample = dataset[args.index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrGeoTrajectoryPredictionModel(**checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        prediction = model(sample)["position"]
        target = sample["target_position"].to(device)
        mask = sample["target_mask"].to(device)
        metrics = ade_fde(prediction, target, mask)

    ids = sample["prediction_vehicle_ids"][sample["target_mask"]].tolist()
    predicted_xy = prediction[mask].cpu().tolist()
    ground_truth_xy = target[mask].cpu().tolist()
    origin_x, origin_y = sample["meta"]["origin_xy"]

    print(
        json.dumps(
            {
                "benchmark_id": sample["meta"]["benchmark_id"],
                "ADE_m": metrics["ade"],
                "FDE_m": metrics["fde"],
                "target_vehicle_ids": ids,
                "scene_origin_xy": [origin_x, origin_y],
                "predicted_xy_relative_to_origin": predicted_xy,
                "ground_truth_xy_relative_to_origin": ground_truth_xy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
