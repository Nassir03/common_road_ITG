import torch
from model.gnn_model import CrGeoTrajectoryPredictionModel
from model.metrics import ade_loss
from model.config import OBS_STEPS, PRED_STEPS


def empty_edges():
    return torch.empty((2, 0), dtype=torch.long)


def test_model_forward_minimal_paper_graph():
    # One vehicle unrolled through the observed cache, with causal VTV edges.
    n = OBS_STEPS
    pairs = []
    motion = []
    delta_t = []
    for a in range(OBS_STEPS - 1):
        for b in range(a + 1, OBS_STEPS):
            pairs.append([a, b])
            motion.append([0.0] * 8)
            delta_t.append((b - a) * 0.2)

    sample = {
        "vehicle_x": torch.zeros((n, 10)),
        "latest_vehicle_node_index": torch.tensor([OBS_STEPS - 1]),
        "lane_x": torch.empty((0, 4)),
        "lane_geometry": torch.empty((0, 1, 4)),
        "lane_geometry_lengths": torch.empty((0,), dtype=torch.long),
        "edge_index": {
            "v2v": empty_edges(),
            "v2l": empty_edges(),
            "l2v": empty_edges(),
            "l2l": empty_edges(),
            "vtv": torch.tensor(pairs, dtype=torch.long).t().contiguous(),
        },
        "edge_attr": {
            "v2v": torch.empty((0, 8)),
            "v2l": torch.empty((0, 6)),
            "l2v": torch.empty((0, 6)),
            "l2l_numeric": torch.empty((0, 6)),
            "l2l_type": torch.empty((0,), dtype=torch.long),
            "vtv_motion": torch.tensor(motion, dtype=torch.float32),
            "vtv_delta_t": torch.tensor(delta_t, dtype=torch.float32),
        },
        "current_position": torch.zeros((1, 2)),
        "current_orientation": torch.zeros((1,)),
        "target_position": torch.zeros((1, PRED_STEPS, 2)),
        "target_mask": torch.tensor([True]),
    }
    model = CrGeoTrajectoryPredictionModel(
        hidden_dim=32, heads=4, hgt_layers=2, decoder_hidden_dim=32
    )
    output = model(sample)
    assert output["position"].shape == (1, PRED_STEPS, 2)
    assert torch.isfinite(output["position"]).all()
    loss = ade_loss(output["position"], sample["target_position"], sample["target_mask"])
    assert torch.isfinite(loss)
    loss.backward()
