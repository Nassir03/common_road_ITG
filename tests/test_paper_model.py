from pathlib import Path

import torch

from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_fde


def _synthetic_scenario():
    lane = {
        "id": 1,
        "left": [(0.0, 2.0), (100.0, 2.0)],
        "right": [(0.0, -2.0), (100.0, -2.0)],
        "center": [(0.0, 0.0), (100.0, 0.0)],
        "length": 100.0,
        "heading": 0.0,
    }
    vehicles = {}
    for vid, y, speed in [(1, 0.0, 5.0), (2, 1.0, 6.0)]:
        states = {}
        for t in range(0, 50):
            time = t * 0.1
            states[t] = {
                "x": 10.0 + speed * time,
                "y": y,
                "orientation": 0.0,
                "velocity": speed,
                "acceleration": 0.0,
                "yaw_rate": 0.0,
            }
        vehicles[vid] = {"id": vid, "length": 4.5, "width": 1.8, "states": states}
    return {
        "source_file": "synthetic.xml",
        "xml_member": None,
        "benchmark_id": "TEST_City-1",
        "location_group": "TEST_City",
        "dt": 0.1,
        "lanelets": {1: lane},
        "l2l_edges": [],
        "vehicles": vehicles,
    }


def test_temporal_graph_has_paper_edge_types_and_5_step_target(tmp_path: Path):
    torch.save(_synthetic_scenario(), tmp_path / "synthetic.scenario.pt")
    ds = CommonRoadTemporalGraphDataset(tmp_path, stride=5)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["vehicle_x"].shape[1] == 10
    assert set(sample["edge_index"]) == {"v2v", "v2l", "l2v", "l2l", "vtv"}
    assert sample["edge_attr"]["v2v"].shape[1] == 8
    assert sample["edge_attr"]["v2l"].shape[1] == 6
    assert sample["edge_attr"]["vtv_motion"].shape[1] == 8
    assert sample["target_position"].shape == (2, 5, 2)
    assert sample["target_orientation"].shape == (2, 5)
    assert torch.allclose(sample["prediction_times"], torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]))


def test_small_hgt_gru_model_forward_backward(tmp_path: Path):
    torch.save(_synthetic_scenario(), tmp_path / "synthetic.scenario.pt")
    sample = CommonRoadTemporalGraphDataset(tmp_path, stride=5)[0]
    model = CrGeoTrajectoryPredictionModel(
        hidden_dim=32,
        heads=4,
        hgt_layers=2,
        pred_steps=5,
        decoder_hidden_dim=64,
    )
    output = model(sample)
    assert output["position"].shape == (2, 5, 2)
    assert output["orientation"].shape == (2, 5)
    assert output["local_delta"].shape == (2, 5, 3)
    ade, fde = ade_fde(output["position"], sample["target_position"], sample["target_mask"])
    assert torch.isfinite(ade) and torch.isfinite(fde)
    ade.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
