#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--city",required=True,choices=["boston","pittsburgh","singapore"]); ap.add_argument("--checkpoint",required=True); ap.add_argument("--data-root",default=None); ap.add_argument("--index",type=int,default=0); args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False); mode=ckpt.get("v2v_mode","paper"); data_root=Path(args.data_root) if args.data_root else ROOT/"data"/args.city/"processed"
    ds=CommonRoadTemporalGraphDataset(data_root/"test",v2v_mode=mode); s=ds[args.index]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=CrGeoTrajectoryPredictionModel(**ckpt.get("model_kwargs",{"v2v_edge_dim":ds.v2v_edge_dim})).to(device); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad(): pred=model(s)["position"]; target=s["target_position"].to(device); mask=s["target_mask"].to(device); m=ade_fde(pred,target,mask)
    ids=s["vehicle_ids"][s["target_mask"]].tolist(); p=pred[mask].cpu().tolist(); y=target[mask].cpu().tolist()
    print(json.dumps({"benchmark_id":s["meta"]["benchmark_id"],"mode":mode,"ADE_m":m["ade"],"FDE_m":m["fde"],"target_vehicle_ids":ids,"predicted_xy":p,"ground_truth_xy":y},indent=2))
if __name__=="__main__": main()
