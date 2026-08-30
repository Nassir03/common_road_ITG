import torch
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_loss


def empty_edges():
    return torch.empty((2,0),dtype=torch.long)

def test_model_forward_minimal_paper_graph():
    # 1 vehicle across 15 time steps, no lanes; causal VTV edges.
    n=15
    pairs=[]; motion=[]; dt=[]
    for a in range(14):
        for b in range(a+1,15):
            pairs.append([a,b]); motion.append([0.0]*8); dt.append((b-a)*0.2)
    sample={
        "vehicle_x":torch.zeros((n,10)),
        "latest_vehicle_node_index":torch.tensor([14]),
        "lane_x":torch.empty((0,4)),"lane_geometry":torch.empty((0,1,4)),"lane_geometry_lengths":torch.empty((0,),dtype=torch.long),
        "edge_index":{"v2v":empty_edges(),"v2l":empty_edges(),"l2v":empty_edges(),"l2l":empty_edges(),"vtv":torch.tensor(pairs,dtype=torch.long).t()},
        "edge_attr":{"v2v":torch.empty((0,8)),"v2l":torch.empty((0,6)),"l2v":torch.empty((0,6)),"l2l_numeric":torch.empty((0,6)),"l2l_type":torch.empty((0,),dtype=torch.long),"vtv_motion":torch.tensor(motion),"vtv_delta_t":torch.tensor(dt)},
        "current_position":torch.zeros((1,2)),"current_orientation":torch.zeros((1,)),
        "target_position":torch.zeros((1,5,2)),"target_mask":torch.tensor([True]),
    }
    model=CrGeoTrajectoryPredictionModel(v2v_edge_dim=8,hidden_dim=32,heads=4,hgt_layers=2,decoder_hidden_dim=32)
    out=model(sample)
    assert out["position"].shape==(1,5,2)
    loss=ade_loss(out["position"],sample["target_position"],sample["target_mask"])
    loss.backward()
