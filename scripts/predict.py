#!/usr/bin/env python3
"""Run one temporal graph through the model and print future x/y/orientation."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dcfg = ckpt.get("dataset", {})
    ds = CommonRoadTemporalGraphDataset(
        Path(args.data_root) / args.split,
        obs_steps=int(dcfg.get("obs_steps", 15)),
        pred_steps=int(dcfg.get("pred_steps", 5)),
        model_dt=float(dcfg.get("model_dt", 0.2)),
        stride=int(dcfg.get("stride", 1)),
    )
    sample = ds[args.index]
    model = CrGeoTrajectoryPredictionModel(**ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        output = model(sample)
        ade, fde = ade_fde(
            output["position"],
            sample["target_position"].to(device),
            sample["target_mask"].to(device),
        )

    print("benchmark:", sample["meta"]["benchmark_id"])
    print(f"ADE={float(ade):.4f} m  FDE={float(fde):.4f} m")
    valid_indices = torch.where(sample["target_mask"])[0].tolist()
    for track in valid_indices[:5]:
        vehicle_id = int(sample["vehicle_ids"][track])
        print(f"\nvehicle {vehicle_id}")
        print("predicted [x, y, orientation]:")
        pred = torch.cat([
            output["position"][track].cpu(),
            output["orientation"][track].cpu().unsqueeze(-1),
        ], dim=-1)
        print(pred)
        print("ground truth [x, y, orientation]:")
        gt = torch.cat([
            sample["target_position"][track],
            sample["target_orientation"][track].unsqueeze(-1),
        ], dim=-1)
        print(gt)


if __name__ == "__main__":
    main()
