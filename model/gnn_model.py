"""Edge-enhanced HGT encoder and GRU decoder described by the cr-geo paper."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .config import (
    HIDDEN_DIM,
    LANE_GRU_HIDDEN_DIM,
    L2L_RELATION_EMBED_DIM,
    TIME2VEC_DIM,
    HGT_LAYERS,
    HGT_HEADS,
    DECODER_HIDDEN_DIM,
    PRED_STEPS,
    VEHICLE_FEATURE_DIM,
    V2V_EDGE_DIM,
    V2L_EDGE_DIM,
    L2L_NUMERIC_EDGE_DIM,
    L2L_RELATION_COUNT,
    LANE_GEOMETRY_DIM,
    LANE_STATIC_DIM,
    DROPOUT,
)

NODE_TYPES = ("vehicle", "lane")
REL_META = {
    "v2v": ("vehicle", "vehicle"),
    "v2l": ("vehicle", "lane"),
    "l2v": ("lane", "vehicle"),
    "l2l": ("lane", "lane"),
    "vtv": ("vehicle", "vehicle"),
}


class Time2Vec(nn.Module):
    """Learnable time representation used for the VTV delta-time attribute."""

    def __init__(self, dim: int = TIME2VEC_DIM):
        super().__init__()
        if dim < 1:
            raise ValueError("Time2Vec dimension must be positive")
        self.frequency = nn.Parameter(torch.empty(dim))
        self.phase = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.frequency, mean=0.0, std=0.2)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        t = delta_t.reshape(-1, 1)
        raw = t * self.frequency.reshape(1, -1) + self.phase.reshape(1, -1)
        if raw.size(1) == 1:
            return raw
        return torch.cat([raw[:, :1], torch.sin(raw[:, 1:])], dim=-1)


class LaneletEncoder(nn.Module):
    """GRU encoder for variable-length left/right lanelet waypoint sequences."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRU(LANE_GEOMETRY_DIM, LANE_GRU_HIDDEN_DIM, batch_first=True)
        self.static_norm = nn.LayerNorm(LANE_STATIC_DIM)
        self.output = nn.Sequential(
            nn.Linear(LANE_GRU_HIDDEN_DIM + LANE_STATIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, geometry: torch.Tensor, lengths: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        if static_x.size(0) == 0:
            return static_x.new_empty((0, self.hidden_dim))
        packed = pack_padded_sequence(
            geometry,
            lengths.cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.output(torch.cat([hidden[-1], self.static_norm(static_x)], dim=-1))


class EdgeEnhancedHGTLayer(nn.Module):
    """Compact HGT-style relation-aware attention using node and edge features.

    The paper states that both attention weights and messages depend on node and
    edge features. This layer implements exactly that property while keeping the
    code small enough for Kaggle.
    """

    def __init__(self, hidden_dim: int, heads: int, edge_dims: dict[str, int], dropout: float = DROPOUT):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by number of attention heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.query = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in NODE_TYPES})
        self.key = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in NODE_TYPES})
        self.value = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in NODE_TYPES})
        self.output = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in NODE_TYPES})
        self.norm = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES})
        self.skip = nn.ParameterDict({node_type: nn.Parameter(torch.zeros(())) for node_type in NODE_TYPES})

        self.edge_norm = nn.ModuleDict({relation: nn.LayerNorm(edge_dims[relation]) for relation in REL_META})
        self.edge_query = nn.ModuleDict({relation: nn.Linear(edge_dims[relation], hidden_dim) for relation in REL_META})
        self.edge_key = nn.ModuleDict({relation: nn.Linear(edge_dims[relation], hidden_dim) for relation in REL_META})
        self.edge_value = nn.ModuleDict({relation: nn.Linear(edge_dims[relation], hidden_dim) for relation in REL_META})

        self.relation_attention = nn.ParameterDict()
        self.relation_message = nn.ParameterDict()
        self.relation_prior = nn.ParameterDict()
        for relation in REL_META:
            attention = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
            message = nn.Parameter(torch.empty(heads, self.head_dim, self.head_dim))
            nn.init.xavier_uniform_(attention)
            nn.init.xavier_uniform_(message)
            self.relation_attention[relation] = attention
            self.relation_message[relation] = message
            self.relation_prior[relation] = nn.Parameter(torch.ones(heads))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, destination: torch.Tensor, n_destination: int) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        index = destination[:, None].expand(-1, scores.size(1))
        maxima = torch.full(
            (n_destination, scores.size(1)),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        stabilized = (scores - maxima[destination]).clamp(min=-30.0, max=30.0)
        exponent = torch.exp(stabilized)
        denominator = torch.zeros(
            (n_destination, scores.size(1)), dtype=scores.dtype, device=scores.device
        )
        denominator.scatter_add_(0, index, exponent)
        return exponent / denominator[destination].clamp_min(1e-12)

    def forward(
        self,
        node_h: dict[str, torch.Tensor],
        edge_index: dict[str, torch.Tensor],
        edge_attr: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        accumulated = {
            node_type: node_h[node_type].new_zeros((node_h[node_type].size(0), self.hidden_dim))
            for node_type in NODE_TYPES
        }
        incoming_relation_count = {
            node_type: node_h[node_type].new_zeros((node_h[node_type].size(0), 1))
            for node_type in NODE_TYPES
        }

        for relation, (source_type, target_type) in REL_META.items():
            edges = edge_index[relation]
            if edges.numel() == 0 or node_h[source_type].size(0) == 0 or node_h[target_type].size(0) == 0:
                continue

            source, destination = edges[0], edges[1]
            edge = self.edge_norm[relation](edge_attr[relation])

            q = (
                self.query[target_type](node_h[target_type][destination])
                + self.edge_query[relation](edge)
            ).view(-1, self.heads, self.head_dim)
            k = (
                self.key[source_type](node_h[source_type][source])
                + self.edge_key[relation](edge)
            ).view(-1, self.heads, self.head_dim)
            v = (
                self.value[source_type](node_h[source_type][source])
                + self.edge_value[relation](edge)
            ).view(-1, self.heads, self.head_dim)

            transformed_key = torch.einsum(
                "ehd,hdf->ehf", k, self.relation_attention[relation]
            )
            transformed_value = torch.einsum(
                "ehd,hdf->ehf", v, self.relation_message[relation]
            )
            scores = (
                (q * transformed_key).sum(dim=-1)
                * self.relation_prior[relation].reshape(1, -1)
                / math.sqrt(self.head_dim)
            )
            scores = torch.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            attention = self._segment_softmax(scores, destination, node_h[target_type].size(0))

            relation_aggregate = node_h[target_type].new_zeros(
                (node_h[target_type].size(0), self.heads, self.head_dim)
            )
            relation_aggregate.index_add_(
                0, destination, attention.unsqueeze(-1) * transformed_value
            )
            relation_aggregate = relation_aggregate.reshape(
                node_h[target_type].size(0), self.hidden_dim
            )

            has_incoming = torch.zeros(
                node_h[target_type].size(0), dtype=torch.bool, device=destination.device
            )
            has_incoming[destination] = True
            accumulated[target_type] += relation_aggregate
            incoming_relation_count[target_type] += has_incoming.unsqueeze(-1).to(accumulated[target_type].dtype)

        output: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            h = node_h[node_type]
            if h.size(0) == 0:
                output[node_type] = h
                continue
            count = incoming_relation_count[node_type]
            has_incoming = count.squeeze(-1) > 0
            aggregate = accumulated[node_type] / count.clamp_min(1.0)
            transformed = self.dropout(torch.relu(self.output[node_type](aggregate)))
            beta = torch.sigmoid(self.skip[node_type])
            updated = beta * transformed + (1.0 - beta) * h
            updated = torch.where(has_incoming.unsqueeze(-1), updated, h)
            output[node_type] = self.norm[node_type](updated)
        return output


class LocalTrajectoryGRUDecoder(nn.Module):
    """Predict local dx, dy, dtheta and integrate them recursively."""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = DECODER_HIDDEN_DIM,
        pred_steps: int = PRED_STEPS,
    ):
        super().__init__()
        self.pred_steps = pred_steps
        self.init_hidden = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.cell = nn.GRUCell(3, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

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
            delta = self.head(hidden)
            dx, dy, dtheta = delta[:, 0], delta[:, 1], delta[:, 2]
            c, s = torch.cos(orientation), torch.sin(orientation)
            position = position + torch.stack([c * dx - s * dy, s * dx + c * dy], dim=-1)
            orientation = torch.atan2(
                torch.sin(orientation + dtheta), torch.cos(orientation + dtheta)
            )
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
        self.vehicle_encoder = nn.Sequential(
            nn.LayerNorm(VEHICLE_FEATURE_DIM),
            nn.Linear(VEHICLE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lane_encoder = LaneletEncoder(hidden_dim)
        self.l2l_embedding = nn.Embedding(L2L_RELATION_COUNT, L2L_RELATION_EMBED_DIM)
        self.time2vec = Time2Vec(TIME2VEC_DIM)

        edge_dims = {
            "v2v": V2V_EDGE_DIM,
            "v2l": V2L_EDGE_DIM,
            "l2v": V2L_EDGE_DIM,
            "l2l": L2L_NUMERIC_EDGE_DIM + L2L_RELATION_EMBED_DIM,
            "vtv": V2V_EDGE_DIM + TIME2VEC_DIM,
        }
        self.hgt = nn.ModuleList(
            [EdgeEnhancedHGTLayer(hidden_dim, heads, edge_dims) for _ in range(hgt_layers)]
        )
        self.decoder = LocalTrajectoryGRUDecoder(
            hidden_dim, decoder_hidden_dim, pred_steps
        )

    def _encoded_edge_attributes(self, sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
        raw = sample["edge_attr"]

        l2l_numeric = raw["l2l_numeric"].to(device)
        l2l_type = raw["l2l_type"].to(device)
        if l2l_numeric.size(0):
            l2l = torch.cat([l2l_numeric, self.l2l_embedding(l2l_type)], dim=-1)
        else:
            l2l = l2l_numeric.new_empty(
                (0, L2L_NUMERIC_EDGE_DIM + L2L_RELATION_EMBED_DIM)
            )

        vtv_motion = raw["vtv_motion"].to(device)
        vtv_delta_t = raw["vtv_delta_t"].to(device)
        if vtv_motion.size(0):
            vtv = torch.cat([vtv_motion, self.time2vec(vtv_delta_t)], dim=-1)
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
        nodes = {"vehicle": vehicle_h, "lane": lane_h}
        edges = {key: value.to(device) for key, value in sample["edge_index"].items()}
        attrs = self._encoded_edge_attributes(sample, device)

        for layer in self.hgt:
            nodes = layer(nodes, edges, attrs)

        context = nodes["vehicle"][sample["latest_vehicle_node_index"].to(device)]
        output = self.decoder(
            context,
            sample["current_position"].to(device),
            sample["current_orientation"].to(device),
        )
        if not torch.isfinite(output["position"]).all():
            raise FloatingPointError("Model produced a non-finite trajectory prediction")
        return output
