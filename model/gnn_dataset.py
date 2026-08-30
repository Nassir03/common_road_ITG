"""Lazy CommonRoadTemporalData-style dataset.

Each sample contains 15 observed model steps and 5 future steps at 0.2 s.
The paper baseline uses a Delaunay/Voronoi V2V graph. The ITG extension replaces
that V2V topology with ROT -> ROC -> ROI -> BFS -> inward ITG edges while
retaining V2L, L2V, L2L and causal VTV edges.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import (
    MODEL_DT, OBS_STEPS, PRED_STEPS, WINDOW_STRIDE, MIN_CONTEXT_VEHICLES,
    COMMUNICATION_RADIUS, ROI_RADIUS, MAX_HOPS,
    PAPER_V2V_EDGE_DIM, ITG_V2V_EDGE_DIM,
)
from .geometry import (
    delaunay_directed_edges, lane_local_geometry, lane_static_feature_vector,
    l2l_numeric_feature_vector, v2l_feature_vector, v2v_feature_vector,
    vehicle_feature_vector,
)
from .itg import build_itg_multigraph
from .scenario import assign_lanelets

@dataclass(frozen=True)
class SampleRef:
    file_index: int
    times: tuple[int, ...]
    context_vehicle_ids: tuple[int, ...]


def _empty_edge_index():
    return torch.empty((2,0), dtype=torch.long)

def _edge_index(pairs):
    return torch.tensor(pairs,dtype=torch.long).t().contiguous() if pairs else _empty_edge_index()

def _edge_attr(rows,dim):
    return torch.tensor(rows,dtype=torch.float32) if rows else torch.empty((0,dim),dtype=torch.float32)

class CommonRoadTemporalGraphDataset(Dataset):
    def __init__(self, scenario_dir: str|Path, v2v_mode: str="paper", obs_steps: int=OBS_STEPS,
                 pred_steps: int=PRED_STEPS, model_dt: float=MODEL_DT, stride: int=WINDOW_STRIDE,
                 cache_size: int=2):
        self.files=sorted(Path(scenario_dir).glob("*.scenario.pt"))
        if not self.files:
            raise FileNotFoundError(f"No *.scenario.pt files found in {scenario_dir}. Run scripts/prepare_city.py first.")
        if v2v_mode not in {"paper","itg"}:
            raise ValueError("v2v_mode must be 'paper' or 'itg'")
        self.v2v_mode=v2v_mode; self.obs_steps=int(obs_steps); self.pred_steps=int(pred_steps)
        self.model_dt=float(model_dt); self.stride=int(stride); self.total_steps=self.obs_steps+self.pred_steps
        self.cache_size=max(1,int(cache_size)); self._cache: OrderedDict[int,dict[str,Any]]=OrderedDict()
        self.samples: list[SampleRef]=[]; self.skipped_scenarios=[]; self._file_to_samples=defaultdict(list)
        self._build_index()

    @property
    def v2v_edge_dim(self):
        return PAPER_V2V_EDGE_DIM if self.v2v_mode=="paper" else ITG_V2V_EDGE_DIM

    def _load(self,file_index:int):
        if file_index in self._cache:
            value=self._cache.pop(file_index); self._cache[file_index]=value; return value
        value=torch.load(self.files[file_index],weights_only=False)
        self._cache[file_index]=value
        while len(self._cache)>self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _gap(self,scenario):
        source_dt=float(scenario.get("dt",0.0))
        if source_dt<=0: return None
        ratio=self.model_dt/source_dt; gap=int(round(ratio))
        if gap<1 or abs(gap*source_dt-self.model_dt)>1e-6: return None
        return gap

    def _build_index(self):
        # Load one scenario at a time only; do not keep the whole city in RAM.
        for fi,path in enumerate(self.files):
            scenario=torch.load(path,weights_only=False); vehicles=scenario["vehicles"]
            if not vehicles: continue
            gap=self._gap(scenario)
            if gap is None:
                self.skipped_scenarios.append((scenario.get("benchmark_id",path.stem),scenario.get("dt"))); continue
            all_times=sorted({t for v in vehicles.values() for t in v["states"]})
            if not all_times: continue
            span=(self.total_steps-1)*gap; start_stride=max(1,self.stride)*gap
            for start in range(all_times[0],all_times[-1]-span+1,start_stride):
                times=tuple(start+k*gap for k in range(self.total_steps)); obs=times[:self.obs_steps]; future=times[self.obs_steps:]
                context=tuple(sorted(vid for vid,v in vehicles.items() if all(t in v["states"] for t in obs)))
                if len(context)<MIN_CONTEXT_VEHICLES: continue
                if not any(all(t in vehicles[vid]["states"] for t in future) for vid in context): continue
                idx=len(self.samples); self.samples.append(SampleRef(fi,times,context)); self._file_to_samples[fi].append(idx)
            del scenario

    def __len__(self): return len(self.samples)

    def epoch_indices(self, seed:int, max_samples:int=0):
        """Shuffle scenarios, not arbitrary windows, to keep disk I/O efficient."""
        rng=random.Random(seed); files=list(self._file_to_samples); rng.shuffle(files); out=[]
        for fi in files:
            ids=list(self._file_to_samples[fi]); rng.shuffle(ids); out.extend(ids)
            if max_samples>0 and len(out)>=max_samples: return out[:max_samples]
        return out

    def __getitem__(self,index):
        ref=self.samples[index]; scenario=self._load(ref.file_index); vehicles=scenario["vehicles"]
        vehicle_ids=list(ref.context_vehicle_ids); track={vid:i for i,vid in enumerate(vehicle_ids)}; n=len(vehicle_ids)
        obs=list(ref.times[:self.obs_steps]); future=list(ref.times[self.obs_steps:])

        # Lane assignments and one-hop map context.
        assignments=[]; used=set()
        for t in obs:
            a={}
            for vid in vehicle_ids:
                s=vehicles[vid]["states"][t]; lids=assign_lanelets(s["x"],s["y"],scenario["lanelets"]); a[vid]=lids; used.update(lids)
            assignments.append(a)
        for a,b,_ in scenario["l2l_edges"]:
            if a in used or b in used: used.update((a,b))
        lane_ids=sorted(l for l in used if l in scenario["lanelets"]); lane_index={l:i for i,l in enumerate(lane_ids)}
        seqs=[torch.tensor(lane_local_geometry(scenario["lanelets"][l]),dtype=torch.float32) for l in lane_ids]
        if seqs:
            lane_lengths=torch.tensor([s.size(0) for s in seqs],dtype=torch.long); lane_geometry=pad_sequence(seqs,batch_first=True)
            lane_x=torch.tensor([lane_static_feature_vector(scenario["lanelets"][l]) for l in lane_ids],dtype=torch.float32)
        else:
            lane_lengths=torch.empty((0,),dtype=torch.long); lane_geometry=torch.empty((0,1,4)); lane_x=torch.empty((0,4))
        l2l_pairs=[]; l2l_num=[]; l2l_types=[]
        for a,b,r in scenario["l2l_edges"]:
            if a in lane_index and b in lane_index:
                l2l_pairs.append([lane_index[a],lane_index[b]]); l2l_num.append(l2l_numeric_feature_vector(scenario["lanelets"][a],scenario["lanelets"][b])); l2l_types.append(int(r))

        # Vehicle nodes repeated over time (time-major).
        vrows=[]
        for t in obs:
            for vid in vehicle_ids: vrows.append(vehicle_feature_vector(vehicles[vid],vehicles[vid]["states"][t]))
        vehicle_x=torch.tensor(vrows,dtype=torch.float32)
        latest=torch.tensor([(self.obs_steps-1)*n+track[vid] for vid in vehicle_ids],dtype=torch.long)

        # V2V edges: paper baseline or ITG replacement.
        v2v_pairs=[]; v2v_attrs=[]
        for ti,t in enumerate(obs):
            state_by_id={vid:vehicles[vid]["states"][t] for vid in vehicle_ids}
            if self.v2v_mode=="paper":
                points=[(float(state_by_id[vid]["x"]),float(state_by_id[vid]["y"])) for vid in vehicle_ids]
                for st,dt in delaunay_directed_edges(points):
                    svid,dvid=vehicle_ids[st],vehicle_ids[dt]
                    v2v_pairs.append([ti*n+st,ti*n+dt]); v2v_attrs.append(v2v_feature_vector(state_by_id[svid],state_by_id[dvid]))
            else:
                for e in build_itg_multigraph(state_by_id,COMMUNICATION_RADIUS,ROI_RADIUS,MAX_HOPS):
                    if e.source not in track or e.target not in track: continue
                    base=v2v_feature_vector(state_by_id[e.source],state_by_id[e.target])
                    extra=[e.hop/max(MAX_HOPS,1), e.branch/max(e.branch_count,1), 1.0 if e.direct else 0.0, min(e.distance/max(ROI_RADIUS,1e-6),1.0)]
                    v2v_pairs.append([ti*n+track[e.source],ti*n+track[e.target]]); v2v_attrs.append(base+extra)

        # V2L and reverse L2V.
        v2l_pairs=[]; l2v_pairs=[]; v2l_attrs=[]; l2v_attrs=[]
        for ti,t in enumerate(obs):
            for vid in vehicle_ids:
                vn=ti*n+track[vid]; state=vehicles[vid]["states"][t]
                for lid in assignments[ti][vid]:
                    if lid not in lane_index: continue
                    ln=lane_index[lid]; attr=v2l_feature_vector(state,scenario["lanelets"][lid])
                    v2l_pairs.append([vn,ln]); v2l_attrs.append(attr); l2v_pairs.append([ln,vn]); l2v_attrs.append(attr)

        # Causal VTV: every earlier observation -> every later observation of same vehicle.
        vtv_pairs=[]; vtv_motion=[]; vtv_dt=[]
        for tr,vid in enumerate(vehicle_ids):
            for a in range(self.obs_steps-1):
                for b in range(a+1,self.obs_steps):
                    vtv_pairs.append([a*n+tr,b*n+tr]); vtv_motion.append(v2v_feature_vector(vehicles[vid]["states"][obs[a]],vehicles[vid]["states"][obs[b]])); vtv_dt.append((b-a)*self.model_dt)

        current_pos=[]; current_ori=[]; target_pos=[]; target_ori=[]; mask=[]; last=obs[-1]
        for vid in vehicle_ids:
            cur=vehicles[vid]["states"][last]; current_pos.append([float(cur["x"]),float(cur["y"])]); current_ori.append(float(cur.get("orientation",0.0)))
            valid=all(t in vehicles[vid]["states"] for t in future); mask.append(valid)
            if valid:
                target_pos.append([[float(vehicles[vid]["states"][t]["x"]),float(vehicles[vid]["states"][t]["y"])] for t in future]); target_ori.append([float(vehicles[vid]["states"][t].get("orientation",0.0)) for t in future])
            else:
                target_pos.append([[float(cur["x"]),float(cur["y"])] for _ in future]); target_ori.append([float(cur.get("orientation",0.0)) for _ in future])

        return {
            "vehicle_x":vehicle_x,
            "vehicle_ids":torch.tensor(vehicle_ids,dtype=torch.long),
            "latest_vehicle_node_index":latest,
            "lane_x":lane_x,"lane_geometry":lane_geometry,"lane_geometry_lengths":lane_lengths,
            "edge_index":{"v2v":_edge_index(v2v_pairs),"v2l":_edge_index(v2l_pairs),"l2v":_edge_index(l2v_pairs),"l2l":_edge_index(l2l_pairs),"vtv":_edge_index(vtv_pairs)},
            "edge_attr":{
                "v2v":_edge_attr(v2v_attrs,self.v2v_edge_dim),"v2l":_edge_attr(v2l_attrs,6),"l2v":_edge_attr(l2v_attrs,6),
                "l2l_numeric":_edge_attr(l2l_num,6),"l2l_type":torch.tensor(l2l_types,dtype=torch.long) if l2l_types else torch.empty((0,),dtype=torch.long),
                "vtv_motion":_edge_attr(vtv_motion,8),"vtv_delta_t":torch.tensor(vtv_dt,dtype=torch.float32) if vtv_dt else torch.empty((0,),dtype=torch.float32),
            },
            "current_position":torch.tensor(current_pos,dtype=torch.float32),"current_orientation":torch.tensor(current_ori,dtype=torch.float32),
            "target_position":torch.tensor(target_pos,dtype=torch.float32),"target_orientation":torch.tensor(target_ori,dtype=torch.float32),"target_mask":torch.tensor(mask,dtype=torch.bool),
            "prediction_times":torch.tensor([(k+1)*self.model_dt for k in range(self.pred_steps)],dtype=torch.float32),
            "meta":{"benchmark_id":scenario["benchmark_id"],"source_file":scenario.get("source_file"),"observation_time_steps":obs,"future_time_steps":future,"v2v_mode":self.v2v_mode},
        }
