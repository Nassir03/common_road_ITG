#!/usr/bin/env python3
"""Predict one test sample and print ground truth vs predicted positions."""
from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.gnn_dataset import DynamicITGDataset
from model.gnn_model import SimpleCommonRoadITGGNN
from model.metrics import ade_fde


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--data-root", default=str(ROOT / "data" / "processed")); p.add_argument("--index", type=int, default=0); args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ds = DynamicITGDataset(Path(args.data_root)/"test", stride=int(ckpt.get("stride", 10)), edge_mode=ckpt["edge_mode"])
    s = ds[args.index]; model = SimpleCommonRoadITGGNN().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad(): pred = model(s); ade, fde = ade_fde(pred, s["target"].to(device))
    absolute_pred = pred.cpu() + s["current_position"].unsqueeze(1)
    absolute_gt = s["target"] + s["current_position"].unsqueeze(1)
    print("benchmark:", s["meta"]["benchmark_id"], "VOI:", s["voi_id"])
    print(f"ADE={float(ade):.4f} m FDE={float(fde):.4f} m")
    print("predicted XY:\n", absolute_pred[0])
    print("ground truth XY:\n", absolute_gt[0])

if __name__ == "__main__": main()
