#!/usr/bin/env python3
"""Visualize one VOI-specific ITG from a processed scenario."""
from pathlib import Path
import argparse, sys
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from model.config import COMMUNICATION_RADIUS, ROI_RADIUS, MAX_HOPS
from model.itg import VehicleState, build_itg_for_vehicle
p=argparse.ArgumentParser(); p.add_argument("scenario"); p.add_argument("--time",type=int,default=None); p.add_argument("--voi",type=int,default=None); p.add_argument("--output",default="outputs/itg.png"); a=p.parse_args()
s=torch.load(a.scenario,weights_only=False); times=sorted({t for v in s['vehicles'].values() for t in v['states']}); t=a.time if a.time is not None else times[len(times)//2]
ids=sorted(vid for vid,v in s['vehicles'].items() if t in v['states']); voi=a.voi if a.voi is not None else ids[0]
st=[]
for vid in ids:
 x=s['vehicles'][vid]['states'][t]; st.append(VehicleState(vid,x['x'],x['y'],x['orientation'],x['velocity'],x['acceleration']))
snap=build_itg_for_vehicle(st,voi,t,COMMUNICATION_RADIUS,ROI_RADIUS,MAX_HOPS); by={v.vehicle_id:v for v in st}
fig,ax=plt.subplots(figsize=(8,7)); ax.scatter([v.x for v in st],[v.y for v in st]);
for v in st: ax.text(v.x,v.y,str(v.vehicle_id))
for e in snap.edges:
 u,v=by[e.source],by[e.target]; ax.annotate('',xy=(v.x,v.y),xytext=(u.x,u.y),arrowprops={'arrowstyle':'->'}); ax.text((u.x+v.x)/2,(u.y+v.y)/2,f"h{e.hop}")
ax.scatter([by[voi].x],[by[voi].y],s=160,facecolors='none',edgecolors='black'); ax.set_aspect('equal',adjustable='box'); ax.set_title(f"{s['benchmark_id']} t={t} VOI={voi}"); fig.tight_layout(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=150); print('saved:',a.output)
