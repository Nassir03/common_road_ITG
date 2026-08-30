"""Paper-aligned CommonRoadTemporalData-style graph dataset.

One sample is a complete temporal heterogeneous traffic graph over the observed
history. Vehicle nodes are repeated over time; lanelet nodes are static. The
edge types are V2V, V2L, L2V, L2L and causal VTV, exactly matching Sec. III-A
of Meyer et al. (2023). A sample predicts every observed vehicle that has a
complete 1-second future trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import (
    MODEL_DT, OBS_STEPS, PRED_STEPS, WINDOW_STRIDE, MIN_CONTEXT_VEHICLES,
    VTV_MAX_FUTURE_STEPS,
)
from .geometry import (
    delaunay_directed_edges,
    lane_local_geometry,
    lane_static_feature_vector,
    l2l_numeric_feature_vector,
    v2l_feature_vector,
    v2v_feature_vector,
    vehicle_feature_vector,
)
from .scenario import assign_lanelets


@dataclass(frozen=True)
class SampleRef:
    file_index: int
    times: tuple[int, ...]
    context_vehicle_ids: tuple[int, ...]


def _empty_edge_index() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def _edge_index(pairs: list[list[int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else _empty_edge_index()


def _edge_attr(rows: list[list[float]], dim: int) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, dim), dtype=torch.float32)


class CommonRoadTemporalGraphDataset(Dataset):
    def __init__(
        self,
        scenario_dir: str | Path,
        obs_steps: int = OBS_STEPS,
        pred_steps: int = PRED_STEPS,
        model_dt: float = MODEL_DT,
        stride: int = WINDOW_STRIDE,
    ):
        self.files = sorted(Path(scenario_dir).glob("*.scenario.pt"))
        if not self.files:
            raise FileNotFoundError(f"No *.scenario.pt files found in {scenario_dir}")
        self.obs_steps = int(obs_steps)
        self.pred_steps = int(pred_steps)
        self.model_dt = float(model_dt)
        self.stride = int(stride)
        self.total_steps = self.obs_steps + self.pred_steps
        self.scenarios = [torch.load(p, weights_only=False) for p in self.files]
        self.samples: list[SampleRef] = []
        self.skipped_scenarios: list[tuple[str, str]] = []
        self._build_index()

    def _scenario_step_gap(self, scenario: dict[str, Any]) -> int | None:
        source_dt = float(scenario.get("dt", 0.0))
        if source_dt <= 0:
            return None
        ratio = self.model_dt / source_dt
        gap = int(round(ratio))
        if gap < 1 or abs(gap * source_dt - self.model_dt) > 1e-6:
            return None
        return gap

    def _build_index(self) -> None:
        for file_index, scenario in enumerate(self.scenarios):
            vehicles = scenario["vehicles"]
            if not vehicles:
                continue
            gap = self._scenario_step_gap(scenario)
            if gap is None:
                self.skipped_scenarios.append((scenario["benchmark_id"], f"source dt={scenario.get('dt')} cannot sample exactly at {self.model_dt}s"))
                continue

            all_times = sorted({t for v in vehicles.values() for t in v["states"]})
            if not all_times:
                continue
            min_t, max_t = all_times[0], all_times[-1]
            span = (self.total_steps - 1) * gap
            start_step = max(1, self.stride) * gap
            for start in range(min_t, max_t - span + 1, start_step):
                times = tuple(start + k * gap for k in range(self.total_steps))
                obs_times = times[: self.obs_steps]
                future_times = times[self.obs_steps :]

                # Dense observation histories make the most-recent vehicle node
                # well-defined for every track while preserving all graph types.
                context = sorted(
                    vid for vid, vehicle in vehicles.items()
                    if all(t in vehicle["states"] for t in obs_times)
                )
                if len(context) < MIN_CONTEXT_VEHICLES:
                    continue
                if not any(all(t in vehicles[vid]["states"] for t in future_times) for vid in context):
                    continue
                self.samples.append(SampleRef(file_index, times, tuple(context)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        scenario = self.scenarios[ref.file_index]
        vehicles = scenario["vehicles"]
        vehicle_ids = list(ref.context_vehicle_ids)
        track_to_index = {vid: i for i, vid in enumerate(vehicle_ids)}
        n_tracks = len(vehicle_ids)
        obs_times = list(ref.times[: self.obs_steps])
        future_times = list(ref.times[self.obs_steps :])

        # ----------------------------- lanes -----------------------------
        per_time_assignments: list[dict[int, list[int]]] = []
        used_lane_ids: set[int] = set()
        for t in obs_times:
            assignment: dict[int, list[int]] = {}
            for vid in vehicle_ids:
                state = vehicles[vid]["states"][t]
                lane_ids = assign_lanelets(state["x"], state["y"], scenario["lanelets"])
                assignment[vid] = lane_ids
                used_lane_ids.update(lane_ids)
            per_time_assignments.append(assignment)

        # Include one-hop lane neighbors so L2L map context can reach lanes used
        # by vehicles without loading the full map for every sample.
        for a, b, _ in scenario["l2l_edges"]:
            if a in used_lane_ids or b in used_lane_ids:
                used_lane_ids.update((a, b))
        lane_ids = sorted(lid for lid in used_lane_ids if lid in scenario["lanelets"])
        lane_to_index = {lid: i for i, lid in enumerate(lane_ids)}

        lane_sequences = [torch.tensor(lane_local_geometry(scenario["lanelets"][lid]), dtype=torch.float32) for lid in lane_ids]
        if lane_sequences:
            lane_lengths = torch.tensor([seq.size(0) for seq in lane_sequences], dtype=torch.long)
            lane_geometry = pad_sequence(lane_sequences, batch_first=True)
            lane_x = torch.tensor([lane_static_feature_vector(scenario["lanelets"][lid]) for lid in lane_ids], dtype=torch.float32)
        else:
            lane_lengths = torch.empty((0,), dtype=torch.long)
            lane_geometry = torch.empty((0, 1, 4), dtype=torch.float32)
            lane_x = torch.empty((0, 4), dtype=torch.float32)

        l2l_pairs: list[list[int]] = []
        l2l_numeric: list[list[float]] = []
        l2l_types: list[int] = []
        for a, b, relation in scenario["l2l_edges"]:
            if a in lane_to_index and b in lane_to_index:
                l2l_pairs.append([lane_to_index[a], lane_to_index[b]])
                l2l_numeric.append(l2l_numeric_feature_vector(scenario["lanelets"][a], scenario["lanelets"][b]))
                l2l_types.append(int(relation))

        # -------------------------- vehicle nodes -------------------------
        # Node ordering is time-major: node(t, track) = t_idx*n_tracks + track.
        vehicle_rows: list[list[float]] = []
        vehicle_time_index: list[int] = []
        vehicle_track_index: list[int] = []
        for ti, t in enumerate(obs_times):
            for vid in vehicle_ids:
                vehicle_rows.append(vehicle_feature_vector(vehicles[vid], vehicles[vid]["states"][t]))
                vehicle_time_index.append(ti)
                vehicle_track_index.append(track_to_index[vid])

        vehicle_x = torch.tensor(vehicle_rows, dtype=torch.float32)
        vehicle_time_index_t = torch.tensor(vehicle_time_index, dtype=torch.long)
        vehicle_track_index_t = torch.tensor(vehicle_track_index, dtype=torch.long)
        latest_vehicle_node_index = torch.tensor(
            [(self.obs_steps - 1) * n_tracks + track_to_index[vid] for vid in vehicle_ids],
            dtype=torch.long,
        )

        # ----------------------------- V2V -------------------------------
        v2v_pairs: list[list[int]] = []
        v2v_attrs: list[list[float]] = []
        for ti, t in enumerate(obs_times):
            states = [vehicles[vid]["states"][t] for vid in vehicle_ids]
            points = [(float(s["x"]), float(s["y"])) for s in states]
            for src_track, dst_track in delaunay_directed_edges(points):
                src_node = ti * n_tracks + src_track
                dst_node = ti * n_tracks + dst_track
                v2v_pairs.append([src_node, dst_node])
                v2v_attrs.append(v2v_feature_vector(states[src_track], states[dst_track]))

        # ------------------------- V2L and L2V ---------------------------
        v2l_pairs: list[list[int]] = []
        l2v_pairs: list[list[int]] = []
        v2l_attrs: list[list[float]] = []
        l2v_attrs: list[list[float]] = []
        for ti, t in enumerate(obs_times):
            assignment = per_time_assignments[ti]
            for vid in vehicle_ids:
                track = track_to_index[vid]
                vehicle_node = ti * n_tracks + track
                state = vehicles[vid]["states"][t]
                for lane_id in assignment[vid]:
                    if lane_id not in lane_to_index:
                        continue
                    lane_node = lane_to_index[lane_id]
                    attr = v2l_feature_vector(state, scenario["lanelets"][lane_id])
                    v2l_pairs.append([vehicle_node, lane_node])
                    v2l_attrs.append(attr)
                    # The paper includes L2V as the reverse heterogeneous type;
                    # Table II lists V2L features, so the reverse edge carries
                    # the same geometric relation values.
                    l2v_pairs.append([lane_node, vehicle_node])
                    l2v_attrs.append(attr)

        # ----------------------------- VTV -------------------------------
        vtv_pairs: list[list[int]] = []
        vtv_motion: list[list[float]] = []
        vtv_delta_t: list[float] = []
        for track, vid in enumerate(vehicle_ids):
            for src_ti in range(self.obs_steps - 1):
                max_dst = self.obs_steps
                if VTV_MAX_FUTURE_STEPS is not None:
                    max_dst = min(max_dst, src_ti + 1 + int(VTV_MAX_FUTURE_STEPS))
                for dst_ti in range(src_ti + 1, max_dst):
                    src_state = vehicles[vid]["states"][obs_times[src_ti]]
                    dst_state = vehicles[vid]["states"][obs_times[dst_ti]]
                    vtv_pairs.append([src_ti * n_tracks + track, dst_ti * n_tracks + track])
                    vtv_motion.append(v2v_feature_vector(src_state, dst_state))
                    vtv_delta_t.append((dst_ti - src_ti) * self.model_dt)

        # ----------------------------- targets ---------------------------
        current_positions = []
        current_orientations = []
        target_positions = []
        target_orientations = []
        target_mask = []
        last_obs_t = obs_times[-1]
        for vid in vehicle_ids:
            current = vehicles[vid]["states"][last_obs_t]
            current_positions.append([float(current["x"]), float(current["y"])])
            current_orientations.append(float(current.get("orientation", 0.0)))
            valid = all(t in vehicles[vid]["states"] for t in future_times)
            target_mask.append(valid)
            if valid:
                target_positions.append([[float(vehicles[vid]["states"][t]["x"]), float(vehicles[vid]["states"][t]["y"])] for t in future_times])
                target_orientations.append([float(vehicles[vid]["states"][t].get("orientation", 0.0)) for t in future_times])
            else:
                # Placeholder values are ignored by target_mask.
                target_positions.append([[float(current["x"]), float(current["y"])] for _ in future_times])
                target_orientations.append([float(current.get("orientation", 0.0)) for _ in future_times])

        return {
            "vehicle_x": vehicle_x,
            "vehicle_time_index": vehicle_time_index_t,
            "vehicle_track_index": vehicle_track_index_t,
            "vehicle_ids": torch.tensor(vehicle_ids, dtype=torch.long),
            "latest_vehicle_node_index": latest_vehicle_node_index,
            "lane_x": lane_x,
            "lane_geometry": lane_geometry,
            "lane_geometry_lengths": lane_lengths,
            "lane_ids": torch.tensor(lane_ids, dtype=torch.long),
            "edge_index": {
                "v2v": _edge_index(v2v_pairs),
                "v2l": _edge_index(v2l_pairs),
                "l2v": _edge_index(l2v_pairs),
                "l2l": _edge_index(l2l_pairs),
                "vtv": _edge_index(vtv_pairs),
            },
            "edge_attr": {
                "v2v": _edge_attr(v2v_attrs, 8),
                "v2l": _edge_attr(v2l_attrs, 6),
                "l2v": _edge_attr(l2v_attrs, 6),
                "l2l_numeric": _edge_attr(l2l_numeric, 6),
                "l2l_type": torch.tensor(l2l_types, dtype=torch.long) if l2l_types else torch.empty((0,), dtype=torch.long),
                "vtv_motion": _edge_attr(vtv_motion, 8),
                "vtv_delta_t": torch.tensor(vtv_delta_t, dtype=torch.float32) if vtv_delta_t else torch.empty((0,), dtype=torch.float32),
            },
            "current_position": torch.tensor(current_positions, dtype=torch.float32),
            "current_orientation": torch.tensor(current_orientations, dtype=torch.float32),
            "target_position": torch.tensor(target_positions, dtype=torch.float32),
            "target_orientation": torch.tensor(target_orientations, dtype=torch.float32),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "prediction_times": torch.tensor([(k + 1) * self.model_dt for k in range(self.pred_steps)], dtype=torch.float32),
            "meta": {
                "benchmark_id": scenario["benchmark_id"],
                "location_group": scenario.get("location_group"),
                "source_dt": float(scenario["dt"]),
                "model_dt": self.model_dt,
                "observation_time_steps": obs_times,
                "future_time_steps": future_times,
                "source_file": scenario.get("source_file"),
            },
        }


# Backward-compatible name so existing notebooks fail less abruptly. The old
# ITG edge-mode argument is intentionally gone because ITG is not the paper model.
DynamicITGDataset = CommonRoadTemporalGraphDataset
