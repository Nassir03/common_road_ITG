#!/usr/bin/env python
"""Inspect one prepared temporal graph and its paper feature dimensions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.gnn_dataset import CommonRoadTemporalGraphDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenario-dir',required=True); p.add_argument('--index',type=int,default=0); args=p.parse_args()
    ds=CommonRoadTemporalGraphDataset(Path(args.scenario_dir)); s=ds[args.index]
    report={
        'samples':len(ds),'benchmark_id':s['meta']['benchmark_id'],
        'vehicle_nodes':int(s['vehicle_x'].size(0)),'lanelet_nodes':int(s['lane_x'].size(0)),
        'prediction_vehicles':int(s['prediction_vehicle_ids'].numel()),'valid_targets':int(s['target_mask'].sum()),
        'vehicle_feature_dim':int(s['vehicle_x'].size(1)),'prediction_steps':int(s['target_position'].size(1)),
        'edge_counts':{k:int(v.size(1)) for k,v in s['edge_index'].items()},
        'edge_feature_dims':{
            'V2V':int(s['edge_attr']['v2v'].size(1)),
            'V2L/L2V':int(s['edge_attr']['v2l'].size(1)),
            'L2L_numeric':int(s['edge_attr']['l2l_numeric'].size(1)),
            'VTV_motion':int(s['edge_attr']['vtv_motion'].size(1)),
            'VTV_delta_time':1,
        },
        'prediction_times_s':s['prediction_times'].tolist(),
        'v2v_drawer':s['meta']['v2v_drawer'],'vtv_drawer':s['meta']['vtv_drawer'],
    }
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
