#!/usr/bin/env python
"""Validate real prepared data before launching long training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.batching import batch_graph_samples
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel


def check_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["boston", "pittsburgh", "singapore"])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else ROOT / "data" / args.city / "processed"
    dataset = CommonRoadTemporalGraphDataset(root / "train")
    picked = [dataset[i] for i in range(min(args.samples, len(dataset)))]
    if not picked:
        raise RuntimeError("No train samples were created")

    for sample_index, sample in enumerate(picked):
        check_tensor(f"sample[{sample_index}].vehicle_x", sample["vehicle_x"])
        check_tensor(f"sample[{sample_index}].lane_x", sample["lane_x"])
        check_tensor(f"sample[{sample_index}].target_position", sample["target_position"])
        for key, value in sample["edge_attr"].items():
            check_tensor(f"sample[{sample_index}].edge_attr.{key}", value)

    batch = batch_graph_samples(picked)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrGeoTrajectoryPredictionModel().to(device).eval()
    with torch.inference_mode():
        output = model(batch)["position"]
    check_tensor("prediction", output)

    print(f"OK city={args.city}")
    print(f"samples_checked={len(picked)}")
    print(f"device={device}")
    print(f"prediction_shape={tuple(output.shape)}")
    print(f"v2v_edges={batch['edge_index']['v2v'].size(1)}")
    print(f"vtv_edges={batch['edge_index']['vtv'].size(1)}")
    print("paper-only graph/model validation passed")


if __name__ == "__main__":
    main()
