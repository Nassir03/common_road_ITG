"""VOI-specific dynamic heterogeneous graph dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import (
    COMMUNICATION_RADIUS, ROI_RADIUS, MAX_HOPS, OBS_STEPS, PRED_STEPS,
    WINDOW_STRIDE, MIN_CONTEXT_VEHICLES, MAX_TARGETS_PER_WINDOW, LANE_POINTS,
)
from .itg import VehicleState, build_itg_for_vehicle, build_radius_baseline_edges
from .scenario import assign_lanelets


@dataclass(frozen=True)
class SampleRef:
    file_index: int
    times: tuple[int, ...]
    context_vehicle_ids: tuple[int, ...]
    target_vehicle_id: int


def _sample_polyline(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    if not points:
        return [(0.0, 0.0)] * n
    if len(points) == 1:
        return [points[0]] * n
    return [points[round(i * (len(points) - 1) / (n - 1))] for i in range(n)]


def _lane_geometry(lane: dict[str, Any]) -> torch.Tensor:
    left = _sample_polyline(lane["left"], LANE_POINTS)
    right = _sample_polyline(lane["right"], LANE_POINTS)
    origin = lane["center"][0] if lane["center"] else (0.0, 0.0)
    rows = [[
        (l[0] - origin[0]) / 50.0,
        (l[1] - origin[1]) / 50.0,
        (r[0] - origin[0]) / 50.0,
        (r[1] - origin[1]) / 50.0,
    ] for l, r in zip(left, right)]
    return torch.tensor(rows, dtype=torch.float32)


def _vehicle_state(vehicle_id: int, state: dict[str, float]) -> VehicleState:
    return VehicleState(
        vehicle_id, state["x"], state["y"], state["orientation"],
        state["velocity"], state["acceleration"],
    )


def _vehicle_features(vehicle: dict[str, Any], state: dict[str, float], origin: tuple[float, float]) -> list[float]:
    h, speed, acc = state["orientation"], state["velocity"], state["acceleration"]
    return [
        (state["x"] - origin[0]) / 100.0,
        (state["y"] - origin[1]) / 100.0,
        math.cos(h), math.sin(h),
        speed * math.cos(h) / 30.0,
        speed * math.sin(h) / 30.0,
        acc * math.cos(h) / 5.0,
        acc * math.sin(h) / 5.0,
        float(vehicle["length"]) / 5.0,
        float(vehicle["width"]) / 2.0,
    ]


def _edge_tensors(edges, id_to_index: dict[int, int], vehicle_count: int):
    pairs, attrs = [], []
    for edge in edges:
        if edge.source not in id_to_index or edge.target not in id_to_index:
            continue
        pairs.append([id_to_index[edge.source], id_to_index[edge.target]])
        attrs.append([
            edge.dx_local / max(1.0, ROI_RADIUS),
            edge.dy_local / max(1.0, ROI_RADIUS),
            edge.distance / max(1.0, ROI_RADIUS),
            edge.rel_vx_local / 30.0,
            edge.rel_vy_local / 30.0,
            edge.hop / max(1, MAX_HOPS),
            edge.branch / max(1, vehicle_count),
            float(edge.direct),
            1.0 - min(edge.distance / max(1.0, ROI_RADIUS), 1.0),
        ])
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.empty((0, 9), dtype=torch.float32)
    return edge_index, edge_attr


def _select_targets(targets: list[int], maximum: int) -> list[int]:
    if len(targets) <= maximum:
        return targets
    if maximum == 1:
        return [targets[len(targets) // 2]]
    idx = sorted({round(i * (len(targets) - 1) / (maximum - 1)) for i in range(maximum)})
    return [targets[i] for i in idx]


class DynamicITGDataset(Dataset):
    """One sample predicts one vehicle of interest (VOI).

    Context vehicles only need to exist during the observation history. The VOI
    also needs future ground truth. This preserves vehicles that influence the
    VOI during observation even if they leave before the prediction horizon.
    """
    def __init__(
        self,
        scenario_dir: str | Path,
        obs_steps: int = OBS_STEPS,
        pred_steps: int = PRED_STEPS,
        stride: int = WINDOW_STRIDE,
        edge_mode: str = "itg",
        max_targets_per_window: int = MAX_TARGETS_PER_WINDOW,
    ):
        self.files = sorted(Path(scenario_dir).glob("*.scenario.pt"))
        if not self.files:
            raise FileNotFoundError(f"No *.scenario.pt files found in {scenario_dir}")
        if edge_mode not in {"itg", "radius"}:
            raise ValueError("edge_mode must be 'itg' or 'radius'")
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_steps = obs_steps + pred_steps
        self.stride = stride
        self.edge_mode = edge_mode
        self.max_targets_per_window = max_targets_per_window
        self.scenarios = [torch.load(p, weights_only=False) for p in self.files]
        self.samples: list[SampleRef] = []
        self._build_index()

    def _build_index(self) -> None:
        for file_index, scenario in enumerate(self.scenarios):
            vehicles = scenario["vehicles"]
            all_times = sorted({t for v in vehicles.values() for t in v["states"]})
            for start in range(0, max(0, len(all_times) - self.total_steps + 1), self.stride):
                times = tuple(all_times[start:start + self.total_steps])
                if len(times) != self.total_steps or any(b != a + 1 for a, b in zip(times, times[1:])):
                    continue
                obs_times = times[:self.obs_steps]
                context = sorted(
                    vid for vid, vehicle in vehicles.items()
                    if all(t in vehicle["states"] for t in obs_times)
                )
                if len(context) < MIN_CONTEXT_VEHICLES:
                    continue
                targets = sorted(
                    vid for vid in context
                    if all(t in vehicles[vid]["states"] for t in times)
                )
                for target in _select_targets(targets, self.max_targets_per_window):
                    self.samples.append(SampleRef(file_index, times, tuple(context), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        scenario = self.scenarios[ref.file_index]
        vehicle_ids = list(ref.context_vehicle_ids)
        id_to_index = {vid: i for i, vid in enumerate(vehicle_ids)}
        voi = ref.target_vehicle_id
        voi_index = id_to_index[voi]
        obs_times = list(ref.times[:self.obs_steps])
        future_times = list(ref.times[self.obs_steps:])
        dt = float(scenario["dt"])

        last_obs = obs_times[-1]
        xs = [scenario["vehicles"][vid]["states"][last_obs]["x"] for vid in vehicle_ids]
        ys = [scenario["vehicles"][vid]["states"][last_obs]["y"] for vid in vehicle_ids]
        origin = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Dynamic multi-lanelet center assignment.
        used_lane_ids: set[int] = set()
        per_time_assignments: list[dict[int, list[int]]] = []
        for t in obs_times:
            assignment: dict[int, list[int]] = {}
            for vid in vehicle_ids:
                state = scenario["vehicles"][vid]["states"][t]
                lane_ids = assign_lanelets(state["x"], state["y"], scenario["lanelets"])
                assignment[vid] = lane_ids
                used_lane_ids.update(lane_ids)
            per_time_assignments.append(assignment)
        # Add one-hop lane neighbours so map topology can send context to used lanes.
        for a, b, _ in scenario["l2l_edges"]:
            if a in used_lane_ids or b in used_lane_ids:
                used_lane_ids.update([a, b])
        lane_ids = sorted(lid for lid in used_lane_ids if lid in scenario["lanelets"])
        lane_to_index = {lid: i for i, lid in enumerate(lane_ids)}

        lane_geometry = torch.stack([_lane_geometry(scenario["lanelets"][lid]) for lid in lane_ids]) if lane_ids else torch.empty((0, LANE_POINTS, 4))
        lane_x = torch.tensor([[
            scenario["lanelets"][lid]["length"] / 100.0,
            math.cos(scenario["lanelets"][lid]["heading"]),
            math.sin(scenario["lanelets"][lid]["heading"]),
        ] for lid in lane_ids], dtype=torch.float32) if lane_ids else torch.empty((0, 3), dtype=torch.float32)

        l2l_pairs, l2l_types = [], []
        for a, b, relation in scenario["l2l_edges"]:
            if a in lane_to_index and b in lane_to_index:
                l2l_pairs.append([lane_to_index[a], lane_to_index[b]])
                l2l_types.append(relation)
        l2l_edge_index = torch.tensor(l2l_pairs, dtype=torch.long).t().contiguous() if l2l_pairs else torch.empty((2, 0), dtype=torch.long)
        l2l_type = torch.tensor(l2l_types, dtype=torch.long) if l2l_types else torch.empty((0,), dtype=torch.long)

        node_history, edge_indices, edge_attrs = [], [], []
        v2l_indices, l2v_indices, snapshots = [], [], []
        for obs_i, t in enumerate(obs_times):
            states, rows, v2l_pairs = [], [], []
            for vid in vehicle_ids:
                vehicle = scenario["vehicles"][vid]
                state = vehicle["states"][t]
                states.append(_vehicle_state(vid, state))
                rows.append(_vehicle_features(vehicle, state, origin))
                for lane_id in per_time_assignments[obs_i][vid]:
                    if lane_id in lane_to_index:
                        v2l_pairs.append([id_to_index[vid], lane_to_index[lane_id]])
            node_history.append(torch.tensor(rows, dtype=torch.float32))
            v2l = torch.tensor(v2l_pairs, dtype=torch.long).t().contiguous() if v2l_pairs else torch.empty((2, 0), dtype=torch.long)
            v2l_indices.append(v2l)
            l2v_indices.append(v2l.flip(0))

            if self.edge_mode == "itg":
                snapshot = build_itg_for_vehicle(states, voi, t, COMMUNICATION_RADIUS, ROI_RADIUS, MAX_HOPS)
                edges = snapshot.edges
                snapshots.append(snapshot.to_dict())
            else:
                edges = build_radius_baseline_edges(states, COMMUNICATION_RADIUS)
                snapshots.append({"time_step": t, "vehicle_of_interest": voi, "edges": [e.__dict__ for e in edges]})
            ei, ea = _edge_tensors(edges, id_to_index, len(vehicle_ids))
            edge_indices.append(ei)
            edge_attrs.append(ea)

        current = scenario["vehicles"][voi]["states"][last_obs]
        current_position = torch.tensor([[current["x"], current["y"]]], dtype=torch.float32)
        target = torch.tensor([[
            [scenario["vehicles"][voi]["states"][t]["x"] - current["x"],
             scenario["vehicles"][voi]["states"][t]["y"] - current["y"]]
            for t in future_times
        ]], dtype=torch.float32)

        observation_times = torch.tensor([(i - (self.obs_steps - 1)) * dt for i in range(self.obs_steps)], dtype=torch.float32)
        prediction_times = torch.tensor([(i + 1) * dt for i in range(self.pred_steps)], dtype=torch.float32)

        return {
            "node_history": node_history,
            "edge_indices": edge_indices,
            "edge_attrs": edge_attrs,
            "v2l_indices": v2l_indices,
            "l2v_indices": l2v_indices,
            "lane_geometry": lane_geometry,
            "lane_x": lane_x,
            "l2l_edge_index": l2l_edge_index,
            "l2l_type": l2l_type,
            "observation_times": observation_times,
            "prediction_times": prediction_times,
            "target": target,
            "current_position": current_position,
            "vehicle_ids": vehicle_ids,
            "voi_id": voi,
            "voi_index": voi_index,
            "meta": {
                "benchmark_id": scenario["benchmark_id"],
                "location_group": scenario.get("location_group"),
                "start_time_step": ref.times[0],
                "edge_mode": self.edge_mode,
                "snapshots": snapshots,
            },
        }
