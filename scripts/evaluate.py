#!/usr/bin/env python3
"""Evaluate a saved model on the location-disjoint test split."""
from pathlib import Path
import argparse
import csv
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.gnn_dataset import DynamicITGDataset
from model.gnn_model import SimpleCommonRoadITGGNN
from model.metrics import ade_fde


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data" / "processed"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-csv", default="outputs/test_metrics.csv")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ds = DynamicITGDataset(Path(args.data_root) / "test", stride=int(ckpt.get("stride", 10)), edge_mode=ckpt["edge_mode"])
    model = SimpleCommonRoadITGGNN().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]; pred = model(s); ade, fde = ade_fde(pred, s["target"].to(device))
            rows.append({
                "benchmark_id": s["meta"]["benchmark_id"], "location_group": s["meta"]["location_group"],
                "start_time_step": s["meta"]["start_time_step"], "voi_id": s["voi_id"],
                "ADE_m": float(ade), "FDE_m": float(fde),
            })
    out = Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = {"test_samples": len(rows), "mean_ADE_m": sum(r["ADE_m"] for r in rows)/len(rows), "mean_FDE_m": sum(r["FDE_m"] for r in rows)/len(rows)}
    print(json.dumps(summary, indent=2)); print("saved:", out)

if __name__ == "__main__":
    main()
