#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--city",required=True,choices=["boston","pittsburgh","singapore"]); ap.add_argument("--checkpoint",required=True); ap.add_argument("--data-root",default=None); ap.add_argument("--output-csv",default=None); ap.add_argument("--max-samples",type=int,default=0); args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False); mode=ckpt.get("v2v_mode","paper")
    data_root=Path(args.data_root) if args.data_root else ROOT/"data"/args.city/"processed"
    ds=CommonRoadTemporalGraphDataset(data_root/"test",v2v_mode=mode); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CrGeoTrajectoryPredictionModel(**ckpt.get("model_kwargs",{"v2v_edge_dim":ds.v2v_edge_dim})).to(device); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    rows=[]; total_ade=total_fde=0.0; total_targets=0; n=min(len(ds),args.max_samples) if args.max_samples>0 else len(ds)
    with torch.no_grad():
        for i in range(n):
            s=ds[i]; pred=model(s)["position"]; target=s["target_position"].to(device); mask=s["target_mask"].to(device); m=ade_fde(pred,target,mask)
            if m["count"]==0: continue
            total_ade+=m["ade"]*m["count"]; total_fde+=m["fde"]*m["count"]; total_targets+=m["count"]
            rows.append({"sample":i,"benchmark_id":s["meta"]["benchmark_id"],"ADE_m":m["ade"],"FDE_m":m["fde"],"targets":m["count"]})
    if total_targets==0: raise RuntimeError("No valid test targets")
    summary={"city":args.city,"v2v_mode":mode,"test_samples":len(rows),"test_targets":total_targets,"mean_ADE_m":total_ade/total_targets,"mean_FDE_m":total_fde/total_targets}
    print(json.dumps(summary,indent=2))
    if args.output_csv:
        out=Path(args.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["sample","benchmark_id","ADE_m","FDE_m","targets"]); w.writeheader(); w.writerows(rows)
        out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(f"csv={out}")
if __name__=="__main__": main()
