from pathlib import Path
import torch
from model.gnn_dataset import CommonRoadTemporalGraphDataset
from model.indexing import build_sample_records
from model.config import OBS_STEPS, PRED_STEPS


def _state(t, x, y=0.0):
    return {"x": x, "y": y, "orientation": 0.0, "velocity": 1.0, "acceleration": 0.0, "yaw_rate": 0.0}


def test_variable_vehicle_temporal_dataset(tmp_path: Path):
    # Vehicle 1 spans the whole sample; vehicle 2 appears only in later observations.
    states1 = {t: _state(t, float(t)) for t in range(OBS_STEPS + PRED_STEPS)}
    states2 = {t: _state(t, float(t), 2.0) for t in range(2, OBS_STEPS)}
    scenario = {
        "benchmark_id": "SYNTH",
        "source_file": "synthetic.xml",
        "dt": 0.2,
        "lanelets": {},
        "l2l_edges": [],
        "vehicles": {
            1: {"id": 1, "length": 4.5, "width": 1.8, "states": states1},
            2: {"id": 2, "length": 4.5, "width": 1.8, "states": states2},
        },
        "lane_assignments": {1: {}, 2: {}},
        "lane_grid": {},
        "lane_geometry_cache": {},
        "l2l_numeric_cache": {},
    }
    path = tmp_path / "s.scenario.pt"
    torch.save(scenario, path)
    records = build_sample_records(scenario, path)
    assert records
    torch.save(records, tmp_path / "sample_index.pt")
    dataset = CommonRoadTemporalGraphDataset(tmp_path)
    sample = dataset[0]
    assert sample["vehicle_x"].size(0) > OBS_STEPS  # vehicle 2 is retained as partial context
    assert sample["prediction_vehicle_ids"].tolist() == [1, 2]
    assert sample["target_mask"].tolist() == [True, False]
    assert sample["edge_attr"]["v2v"].size(1) == 8
    assert sample["edge_attr"]["vtv_motion"].size(1) == 8
