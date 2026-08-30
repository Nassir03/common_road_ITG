#!/usr/bin/env python3
"""Inspect one graph sample and verify the paper-required node/edge feature sizes."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.gnn_dataset import CommonRoadTemporalGraphDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True, help="Directory containing *.scenario.pt")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    ds = CommonRoadTemporalGraphDataset(Path(args.scenario_dir), stride=args.stride)
    s = ds[args.index]
    report = {
        "samples": len(ds),
        "benchmark_id": s["meta"]["benchmark_id"],
        "source_dt": s["meta"]["source_dt"],
        "model_dt": s["meta"]["model_dt"],
        "vehicle_nodes": int(s["vehicle_x"].size(0)),
        "lanelet_nodes": int(s["lane_x"].size(0)),
        "vehicle_feature_dim": int(s["vehicle_x"].size(1)),
        "target_vehicles": int(s["target_mask"].sum()),
        "prediction_steps": int(s["target_position"].size(1)),
        "edge_counts": {k: int(v.size(1)) for k, v in s["edge_index"].items()},
        "edge_feature_dims": {
            "V2V": int(s["edge_attr"]["v2v"].size(1)),
            "V2L/L2V": int(s["edge_attr"]["v2l"].size(1)),
            "L2L_numeric": int(s["edge_attr"]["l2l_numeric"].size(1)),
            "VTV_motion_before_Time2Vec": int(s["edge_attr"]["vtv_motion"].size(1)),
        },
        "prediction_times_s": s["prediction_times"].tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
