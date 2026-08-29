"""Simple heterogeneous, temporal, hop-aware GNN for trajectory prediction.

It intentionally keeps the architecture understandable:
- lanelet polyline GRU + L2L message passing,
- both V2L and L2V message passing,
- MAX_HOPS V2V layers so a MAX_HOPS ITG path can reach the VOI,
- causal VTV edges with Time2Vec,
- temporal GRU + GRU trajectory decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    HIDDEN_DIM, TIME2VEC_DIM, ITG_EDGE_FEATURE_DIM, L2L_RELATION_COUNT,
    PRED_STEPS, MAX_HOPS,
)


class Time2Vec(nn.Module):
    def __init__(self, dim: int = TIME2VEC_DIM):
        super().__init__()
        if dim < 2:
            raise ValueError("Time2Vec dim must be >= 2")
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.zeros(1))
        self.periodic_w = nn.Parameter(torch.randn(dim - 1))
        self.periodic_b = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        return torch.cat([t * self.linear_w + self.linear_b, torch.sin(t * self.periodic_w + self.periodic_b)], dim=-1)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0 or edge_index.numel() == 0:
            return h
        src, dst = edge_index
        msg = self.message(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.numel(), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(self.update(agg, h))


class BipartiteMessageLayer(nn.Module):
    """Messages from one node type into another node type."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, source_h: torch.Tensor, target_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if source_h.numel() == 0 or target_h.numel() == 0 or edge_index.numel() == 0:
            return target_h
        src, dst = edge_index
        msg = self.message(torch.cat([source_h[src], target_h[dst]], dim=-1))
        agg = torch.zeros_like(target_h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(target_h.size(0), device=target_h.device, dtype=target_h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.numel(), device=target_h.device, dtype=target_h.dtype))
        return self.norm(target_h + agg / deg.clamp_min(1.0).unsqueeze(-1))


class SimpleCommonRoadITGGNN(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, pred_steps: int = PRED_STEPS, v2v_layers: int = MAX_HOPS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_steps = pred_steps

        self.vehicle_encoder = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.lane_gru = nn.GRU(4, hidden_dim, batch_first=True)
        self.lane_scalar_encoder = nn.Linear(3, hidden_dim)
        self.l2l_relation = nn.Embedding(L2L_RELATION_COUNT, 8)
        self.l2l_layer = EdgeMessageLayer(hidden_dim, 8)
        self.vehicle_to_lane = BipartiteMessageLayer(hidden_dim)
        self.lane_to_vehicle = BipartiteMessageLayer(hidden_dim)

        self.itg_edge_encoder = nn.Sequential(
            nn.Linear(ITG_EDGE_FEATURE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.v2v_layers = nn.ModuleList([EdgeMessageLayer(hidden_dim, hidden_dim) for _ in range(v2v_layers)])

        self.vtv_time2vec = Time2Vec(TIME2VEC_DIM)
        self.vtv_edge_encoder = nn.Sequential(
            nn.Linear(TIME2VEC_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.vtv_layer = EdgeMessageLayer(hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.future_time2vec = Time2Vec(TIME2VEC_DIM)
        self.decoder = nn.GRU(hidden_dim + TIME2VEC_DIM, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def encode_lanes(self, lane_geometry, lane_x, l2l_edge_index, l2l_type):
        if lane_geometry.size(0) == 0:
            return torch.empty((0, self.hidden_dim), device=lane_geometry.device)
        _, h = self.lane_gru(lane_geometry)
        lane_h = h[-1] + self.lane_scalar_encoder(lane_x)
        if l2l_edge_index.numel() > 0:
            lane_h = self.l2l_layer(lane_h, l2l_edge_index, self.l2l_relation(l2l_type))
        return lane_h

    def forward(self, sample: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        lane_h_base = self.encode_lanes(
            sample["lane_geometry"].to(device), sample["lane_x"].to(device),
            sample["l2l_edge_index"].to(device), sample["l2l_type"].to(device),
        )
        observation_times = sample["observation_times"].to(device)

        history = []
        for i, node_x in enumerate(sample["node_history"]):
            vehicle_h = self.vehicle_encoder(node_x.to(device))
            # V2L then L2V: road context is genuinely bidirectional.
            lane_h = self.vehicle_to_lane(vehicle_h, lane_h_base, sample["v2l_indices"][i].to(device))
            vehicle_h = self.lane_to_vehicle(lane_h, vehicle_h, sample["l2v_indices"][i].to(device))

            edge_attr = sample["edge_attrs"][i].to(device)
            encoded_edge = self.itg_edge_encoder(edge_attr) if edge_attr.numel() else torch.empty((0, self.hidden_dim), device=device)
            edge_index = sample["edge_indices"][i].to(device)
            for layer in self.v2v_layers:
                vehicle_h = layer(vehicle_h, edge_index, encoded_edge)
            history.append(vehicle_h)

        # Causal VTV: each observed vehicle at t-1 -> itself at t.
        time_major = torch.stack(history, dim=0)
        t_steps, n_vehicles, _ = time_major.shape
        flat = time_major.reshape(t_steps * n_vehicles, self.hidden_dim)
        if t_steps > 1:
            src, dst, delta = [], [], []
            for t in range(1, t_steps):
                dt = observation_times[t] - observation_times[t - 1]
                for v in range(n_vehicles):
                    src.append((t - 1) * n_vehicles + v)
                    dst.append(t * n_vehicles + v)
                    delta.append(dt)
            vtv_index = torch.tensor([src, dst], dtype=torch.long, device=device)
            delta_t = torch.stack(delta).to(device)
            vtv_attr = self.vtv_edge_encoder(self.vtv_time2vec(delta_t))
            flat = self.vtv_layer(flat, vtv_index, vtv_attr)

        sequence = flat.reshape(t_steps, n_vehicles, self.hidden_dim).permute(1, 0, 2)
        _, hidden_all = self.temporal_gru(sequence)
        voi_index = int(sample["voi_index"])
        context = hidden_all[-1, voi_index:voi_index + 1]
        hidden = hidden_all[:, voi_index:voi_index + 1]

        future_times = sample["prediction_times"].to(device)
        future_t = self.future_time2vec(future_times).unsqueeze(0)
        decoder_in = torch.cat([context.unsqueeze(1).expand(-1, self.pred_steps, -1), future_t], dim=-1)
        decoded, _ = self.decoder(decoder_in, hidden)
        return self.head(decoded)
