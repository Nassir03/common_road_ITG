"""Paper-aligned spatiotemporal trajectory-prediction network.

Architecture follows Meyer et al. (2023):
  1. GRU lanelet-geometry encoder.
  2. Learnable embedding for L2L adjacency type.
  3. Time2Vec encoding for the VTV delta-time attribute.
  4. Edge-enhanced Heterogeneous Graph Transformer (HGT) over
     V2V, V2L, L2V, L2L and causal VTV edges.
  5. GRU decoder producing local (dx, dy, dtheta) transitions.
  6. Repeated local-frame integration to obtain future global states.

The paper does not publish every hidden size / decoder input detail. Those are
kept explicit in model/config.py. The graph semantics, feature families,
prediction horizon and ADE objective match the paper.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .config import (
    DECODER_HIDDEN_DIM,
    HGT_HEADS,
    HGT_LAYERS,
    HIDDEN_DIM,
    L2L_NUMERIC_EDGE_DIM,
    L2L_RELATION_COUNT,
    L2L_RELATION_EMBED_DIM,
    LANE_GEOMETRY_DIM,
    LANE_GRU_HIDDEN_DIM,
    LANE_STATIC_DIM,
    PRED_STEPS,
    TIME2VEC_DIM,
    V2L_EDGE_DIM,
    V2V_EDGE_DIM,
    VEHICLE_FEATURE_DIM,
)


NODE_TYPES = ("vehicle", "lane")
RELATION_META = {
    "v2v": ("vehicle", "vehicle"),
    "v2l": ("vehicle", "lane"),
    "l2v": ("lane", "vehicle"),
    "l2l": ("lane", "lane"),
    "vtv": ("vehicle", "vehicle"),
}


class Time2Vec(nn.Module):
    """Learnable time representation with one linear and sinusoidal channels."""

    def __init__(self, dim: int = TIME2VEC_DIM):
        super().__init__()
        if dim < 2:
            raise ValueError("Time2Vec dimension must be >= 2")
        self.dim = int(dim)
        self.frequency = nn.Parameter(torch.empty(dim))
        self.phase = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.frequency, mean=0.0, std=1.0)
        nn.init.zeros_(self.phase)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        raw = t * self.frequency.reshape(1, -1) + self.phase.reshape(1, -1)
        return torch.cat([raw[:, :1], torch.sin(raw[:, 1:])], dim=-1)


class LaneletEncoder(nn.Module):
    """GRU encoding of variable-length lane boundary waypoint sequences."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, gru_hidden_dim: int = LANE_GRU_HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRU(LANE_GEOMETRY_DIM, gru_hidden_dim, batch_first=True)
        self.static_norm = nn.LayerNorm(LANE_STATIC_DIM)
        self.output = nn.Sequential(
            nn.Linear(gru_hidden_dim + LANE_STATIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, geometry: torch.Tensor, lengths: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        if static_x.size(0) == 0:
            return static_x.new_empty((0, self.output[-1].normalized_shape[0]))
        packed = pack_padded_sequence(
            geometry,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        lane_geometry_h = hidden[-1]
        return self.output(torch.cat([lane_geometry_h, self.static_norm(static_x)], dim=-1))


class EdgeEnhancedHGTLayer(nn.Module):
    """HGT message passing where both attention and messages use edge features.

    For each relation r=(source,target), node-type-specific Q/K/V projections are
    combined with edge-specific Q/K/V projections. Relation-specific attention
    and message matrices then transform K and V. Attention is normalized over
    incoming edges per destination node and head. Different relation outputs
    are merged by elementwise max before a learned residual update.
    """

    def __init__(self, hidden_dim: int, heads: int, edge_dims: dict[str, int]):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.q_node = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in NODE_TYPES})
        self.k_node = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in NODE_TYPES})
        self.v_node = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in NODE_TYPES})
        self.out_node = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in NODE_TYPES})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in NODE_TYPES})
        self.skip = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in NODE_TYPES})

        self.edge_q = nn.ModuleDict({r: nn.Linear(edge_dims[r], hidden_dim) for r in RELATION_META})
        self.edge_k = nn.ModuleDict({r: nn.Linear(edge_dims[r], hidden_dim) for r in RELATION_META})
        self.edge_v = nn.ModuleDict({r: nn.Linear(edge_dims[r], hidden_dim) for r in RELATION_META})

        self.relation_attention = nn.ParameterDict()
        self.relation_message = nn.ParameterDict()
        self.relation_prior = nn.ParameterDict()
        for r in RELATION_META:
            att = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
            msg = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
            nn.init.xavier_uniform_(att)
            nn.init.xavier_uniform_(msg)
            self.relation_attention[r] = att
            self.relation_message[r] = msg
            self.relation_prior[r] = nn.Parameter(torch.ones(heads))

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, n_dst: int) -> torch.Tensor:
        """Softmax over incoming edges for every destination node and head."""
        if scores.numel() == 0:
            return scores
        index = dst[:, None].expand(-1, scores.size(1))
        max_per_dst = torch.full(
            (n_dst, scores.size(1)), -torch.inf,
            dtype=scores.dtype, device=scores.device,
        )
        max_per_dst.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_dst[dst])
        denom = torch.zeros((n_dst, scores.size(1)), dtype=scores.dtype, device=scores.device)
        denom.scatter_add_(0, index, exp_scores)
        return exp_scores / denom[dst].clamp_min(1e-12)

    def forward(
        self,
        node_h: dict[str, torch.Tensor],
        edge_index: dict[str, torch.Tensor],
        edge_attr: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        relation_outputs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {t: [] for t in NODE_TYPES}

        for relation, (src_type, dst_type) in RELATION_META.items():
            ei = edge_index[relation]
            if ei.numel() == 0 or node_h[src_type].size(0) == 0 or node_h[dst_type].size(0) == 0:
                continue
            src, dst = ei[0], ei[1]
            e = edge_attr[relation]
            if e.size(0) != src.numel():
                raise ValueError(f"{relation}: edge attribute count does not match edge count")

            q = self.q_node[dst_type](node_h[dst_type][dst]).view(-1, self.heads, self.head_dim)
            k = self.k_node[src_type](node_h[src_type][src]).view(-1, self.heads, self.head_dim)
            v = self.v_node[src_type](node_h[src_type][src]).view(-1, self.heads, self.head_dim)

            q = q + self.edge_q[relation](e).view(-1, self.heads, self.head_dim)
            k = k + self.edge_k[relation](e).view(-1, self.heads, self.head_dim)
            v = v + self.edge_v[relation](e).view(-1, self.heads, self.head_dim)

            k_meta = torch.einsum("ehd,hdf->ehf", k, self.relation_attention[relation])
            v_meta = torch.einsum("ehd,hdf->ehf", v, self.relation_message[relation])
            score = (q * k_meta).sum(dim=-1)
            score = score * self.relation_prior[relation].reshape(1, -1) / math.sqrt(self.head_dim)
            alpha = self._segment_softmax(score, dst, node_h[dst_type].size(0))

            agg = node_h[dst_type].new_zeros((node_h[dst_type].size(0), self.heads, self.head_dim))
            agg.index_add_(0, dst, alpha.unsqueeze(-1) * v_meta)
            agg = agg.reshape(node_h[dst_type].size(0), self.hidden_dim)
            has_incoming = torch.zeros(node_h[dst_type].size(0), dtype=torch.bool, device=dst.device)
            has_incoming[dst] = True
            relation_outputs[dst_type].append((agg, has_incoming))

        out: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            h = node_h[node_type]
            if h.size(0) == 0 or not relation_outputs[node_type]:
                out[node_type] = h
                continue

            candidates = []
            masks = []
            for rel_agg, rel_mask in relation_outputs[node_type]:
                candidates.append(rel_agg)
                masks.append(rel_mask)
            stacked = torch.stack(candidates, dim=0)  # [R,N,H]
            mask = torch.stack(masks, dim=0)          # [R,N]
            masked = stacked.masked_fill(~mask.unsqueeze(-1), -torch.inf)
            aggregated = masked.max(dim=0).values
            any_incoming = mask.any(dim=0)
            aggregated = torch.where(any_incoming.unsqueeze(-1), aggregated, torch.zeros_like(aggregated))

            transformed = torch.relu(self.out_node[node_type](aggregated))
            beta = torch.sigmoid(self.skip[node_type])
            updated = beta * transformed + (1.0 - beta) * h
            # Nodes with no incoming messages remain unchanged.
            updated = torch.where(any_incoming.unsqueeze(-1), updated, h)
            out[node_type] = self.norm[node_type](updated)
        return out


class LocalTrajectoryGRUDecoder(nn.Module):
    """Generate and integrate local position/orientation deltas autoregressively."""

    def __init__(self, context_dim: int, hidden_dim: int = DECODER_HIDDEN_DIM, pred_steps: int = PRED_STEPS):
        super().__init__()
        self.pred_steps = int(pred_steps)
        self.init_hidden = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.cell = nn.GRUCell(3, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 3)
        # Small initial transitions make the first optimization steps stable.
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def _wrap_tensor(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(
        self,
        context: torch.Tensor,
        current_position: torch.Tensor,
        current_orientation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n = context.size(0)
        hidden = self.init_hidden(context)
        previous_delta = context.new_zeros((n, 3))
        position = current_position
        orientation = current_orientation
        positions = []
        orientations = []
        local_deltas = []

        for _ in range(self.pred_steps):
            hidden = self.cell(previous_delta, hidden)
            delta = self.delta_head(hidden)
            dx_local, dy_local, dtheta = delta[:, 0], delta[:, 1], delta[:, 2]

            c = torch.cos(orientation)
            s = torch.sin(orientation)
            dx_world = c * dx_local - s * dy_local
            dy_world = s * dx_local + c * dy_local
            position = position + torch.stack([dx_world, dy_world], dim=-1)
            orientation = self._wrap_tensor(orientation + dtheta)

            positions.append(position)
            orientations.append(orientation)
            local_deltas.append(delta)
            previous_delta = delta

        return {
            "position": torch.stack(positions, dim=1),
            "orientation": torch.stack(orientations, dim=1),
            "local_delta": torch.stack(local_deltas, dim=1),
        }


class CrGeoTrajectoryPredictionModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        heads: int = HGT_HEADS,
        hgt_layers: int = HGT_LAYERS,
        pred_steps: int = PRED_STEPS,
        decoder_hidden_dim: int = DECODER_HIDDEN_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_steps = pred_steps

        self.vehicle_encoder = nn.Sequential(
            nn.LayerNorm(VEHICLE_FEATURE_DIM),
            nn.Linear(VEHICLE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lane_encoder = LaneletEncoder(hidden_dim=hidden_dim)
        self.l2l_relation_embedding = nn.Embedding(L2L_RELATION_COUNT, L2L_RELATION_EMBED_DIM)
        self.vtv_time2vec = Time2Vec(TIME2VEC_DIM)

        edge_dims = {
            "v2v": V2V_EDGE_DIM,
            "v2l": V2L_EDGE_DIM,
            "l2v": V2L_EDGE_DIM,
            "l2l": L2L_NUMERIC_EDGE_DIM + L2L_RELATION_EMBED_DIM,
            "vtv": V2V_EDGE_DIM + TIME2VEC_DIM,
        }
        self.hgt = nn.ModuleList([
            EdgeEnhancedHGTLayer(hidden_dim, heads, edge_dims) for _ in range(hgt_layers)
        ])
        self.decoder = LocalTrajectoryGRUDecoder(hidden_dim, decoder_hidden_dim, pred_steps)

    def _move_edge_index(self, sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in sample["edge_index"].items()}

    def _prepare_edge_attr(self, sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
        raw = sample["edge_attr"]
        l2l_numeric = raw["l2l_numeric"].to(device)
        l2l_type = raw["l2l_type"].to(device)
        if l2l_numeric.size(0):
            l2l = torch.cat([l2l_numeric, self.l2l_relation_embedding(l2l_type)], dim=-1)
        else:
            l2l = l2l_numeric.new_empty((0, L2L_NUMERIC_EDGE_DIM + L2L_RELATION_EMBED_DIM))

        vtv_motion = raw["vtv_motion"].to(device)
        vtv_dt = raw["vtv_delta_t"].to(device)
        if vtv_motion.size(0):
            vtv = torch.cat([vtv_motion, self.vtv_time2vec(vtv_dt)], dim=-1)
        else:
            vtv = vtv_motion.new_empty((0, V2V_EDGE_DIM + TIME2VEC_DIM))

        return {
            "v2v": raw["v2v"].to(device),
            "v2l": raw["v2l"].to(device),
            "l2v": raw["l2v"].to(device),
            "l2l": l2l,
            "vtv": vtv,
        }

    def forward(self, sample: dict) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        vehicle_h = self.vehicle_encoder(sample["vehicle_x"].to(device))
        lane_h = self.lane_encoder(
            sample["lane_geometry"].to(device),
            sample["lane_geometry_lengths"].to(device),
            sample["lane_x"].to(device),
        )
        node_h = {"vehicle": vehicle_h, "lane": lane_h}
        edge_index = self._move_edge_index(sample, device)
        edge_attr = self._prepare_edge_attr(sample, device)

        for layer in self.hgt:
            node_h = layer(node_h, edge_index, edge_attr)

        latest = sample["latest_vehicle_node_index"].to(device)
        context = node_h["vehicle"][latest]
        return self.decoder(
            context,
            sample["current_position"].to(device),
            sample["current_orientation"].to(device),
        )


# Backward-compatible alias for old notebooks. It now points to the paper model.
SimpleCommonRoadITGGNN = CrGeoTrajectoryPredictionModel
