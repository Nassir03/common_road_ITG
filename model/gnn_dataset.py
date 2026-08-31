"""Paper-only CommonRoadTemporalData-style trajectory dataset.

The graph contains exactly the relation families described in the attached
CommonRoad-Geometric paper: V2V, V2L, L2V, L2L, and causal VTV. V2V uses the
paper's default Voronoi/Delaunay edge construction.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import (
    MODEL_DT,
    OBS_STEPS,
    PRED_STEPS,
    WINDOW_STRIDE,
    VTV_MAX_FUTURE_STEPS,
    V2V_EDGE_DIM,
    V2L_EDGE_DIM,
    L2L_NUMERIC_EDGE_DIM,
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
from .indexing import build_sample_records
from .scenario import assign_lanelets


@dataclass(frozen=True)
class SampleRef:
    file_index: int
    times: tuple[int, ...]


def _empty_edge_index() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def _edge_index(pairs: list[list[int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else _empty_edge_index()


def _edge_attr(rows: list[list[float]], dim: int) -> torch.Tensor:
    tensor = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, dim), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Non-finite graph edge feature encountered during dataset construction")
    return tensor


def _float_tensor(rows) -> torch.Tensor:
    tensor = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Non-finite graph/node/target feature encountered during dataset construction")
    return tensor


class CommonRoadTemporalGraphDataset(Dataset):
    def __init__(
        self,
        scenario_dir: str | Path,
        obs_steps: int = OBS_STEPS,
        pred_steps: int = PRED_STEPS,
        model_dt: float = MODEL_DT,
        stride: int = WINDOW_STRIDE,
        cache_size: int = 4,
    ):
        self.scenario_dir = Path(scenario_dir)
        self.files = sorted(self.scenario_dir.glob("*.scenario.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No *.scenario.pt files found in {scenario_dir}. Run scripts/prepare_city.py first."
            )

        self.obs_steps = int(obs_steps)
        self.pred_steps = int(pred_steps)
        self.model_dt = float(model_dt)
        self.stride = int(stride)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.samples: list[SampleRef] = []
        self._file_to_samples: dict[int, list[int]] = defaultdict(list)

        self._path_to_index = {str(path.resolve()): i for i, path in enumerate(self.files)}
        self._path_to_index.update({path.name: i for i, path in enumerate(self.files)})

        index_path = self.scenario_dir / "sample_index.pt"
        if index_path.exists():
            self._load_index(index_path)
        else:
            self._build_index_fallback()

    @property
    def v2v_edge_dim(self) -> int:
        return V2V_EDGE_DIM

    def _load_index(self, index_path: Path) -> None:
        records = torch.load(index_path, weights_only=False)
        for record in records:
            raw_path = str(record["scenario_file"])
            path = Path(raw_path)
            file_index = self._path_to_index.get(str(path.resolve())) if path.is_absolute() else None
            if file_index is None:
                file_index = self._path_to_index.get(path.name)
            if file_index is None:
                continue
            sample_index = len(self.samples)
            self.samples.append(SampleRef(file_index=file_index, times=tuple(record["times"])))
            self._file_to_samples[file_index].append(sample_index)

    def _build_index_fallback(self) -> None:
        for file_index, path in enumerate(self.files):
            scenario = torch.load(path, weights_only=False)
            for record in build_sample_records(
                scenario,
                path,
                self.obs_steps,
                self.pred_steps,
                self.model_dt,
                self.stride,
            ):
                sample_index = len(self.samples)
                self.samples.append(SampleRef(file_index=file_index, times=tuple(record["times"])))
                self._file_to_samples[file_index].append(sample_index)

    def _load(self, file_index: int) -> dict[str, Any]:
        if file_index in self._cache:
            value = self._cache.pop(file_index)
            self._cache[file_index] = value
            return value
        value = torch.load(self.files[file_index], weights_only=False)
        self._cache[file_index] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self.samples)

    def epoch_indices(self, seed: int, max_samples: int = 0) -> list[int]:
        """Scenario-grouped shuffle to reduce disk churn while remaining random."""
        rng = random.Random(seed)
        file_indices = list(self._file_to_samples)
        rng.shuffle(file_indices)
        output: list[int] = []
        for file_index in file_indices:
            ids = list(self._file_to_samples[file_index])
            rng.shuffle(ids)
            output.extend(ids)
            if max_samples > 0 and len(output) >= max_samples:
                return output[:max_samples]
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        scenario = self._load(ref.file_index)
        vehicles = scenario["vehicles"]
        observed = list(ref.times[: self.obs_steps])
        future = list(ref.times[self.obs_steps :])
        latest_time = observed[-1]

        active_by_time: list[list[int]] = [
            sorted(vid for vid, vehicle in vehicles.items() if t in vehicle["states"])
            for t in observed
        ]
        latest_vehicle_ids = active_by_time[-1]
        if not latest_vehicle_ids:
            raise RuntimeError(f"Indexed sample unexpectedly has no vehicles at t={latest_time}")

        # Pure translation for float32 precision; it does not change distances or ADE/FDE.
        origin_x = sum(float(vehicles[vid]["states"][latest_time]["x"]) for vid in latest_vehicle_ids) / len(latest_vehicle_ids)
        origin_y = sum(float(vehicles[vid]["states"][latest_time]["y"]) for vid in latest_vehicle_ids) / len(latest_vehicle_ids)
        origin = (origin_x, origin_y)

        # Vehicle nodes are time-unrolled exactly as CommonRoadTemporalData describes.
        node_index: dict[tuple[int, int], int] = {}
        vehicle_rows: list[list[float]] = []
        vehicle_node_ids: list[int] = []
        for time_index, t in enumerate(observed):
            for vid in active_by_time[time_index]:
                node_index[(time_index, vid)] = len(vehicle_rows)
                vehicle_rows.append(vehicle_feature_vector(vehicles[vid], vehicles[vid]["states"][t], origin))
                vehicle_node_ids.append(vid)
        vehicle_x = _float_tensor(vehicle_rows)

        latest_vehicle_node_index = torch.tensor(
            [node_index[(self.obs_steps - 1, vid)] for vid in latest_vehicle_ids], dtype=torch.long
        )

        # V2L center assignment for every vehicle state, plus one-hop lane context.
        assignments: list[dict[int, list[int]]] = []
        used_lanelets: set[int] = set()
        cached_assignments = scenario.get("lane_assignments", {})
        lane_grid = scenario.get("lane_grid")
        for time_index, t in enumerate(observed):
            assignment_at_t: dict[int, list[int]] = {}
            for vid in active_by_time[time_index]:
                cached = cached_assignments.get(vid, {}).get(t)
                if cached is None:
                    state = vehicles[vid]["states"][t]
                    lane_ids = assign_lanelets(state["x"], state["y"], scenario["lanelets"], lane_grid)
                else:
                    lane_ids = list(cached)
                assignment_at_t[vid] = lane_ids
                used_lanelets.update(lane_ids)
            assignments.append(assignment_at_t)

        # Include directly related lanelets so map message passing has local topology.
        for source_lane, target_lane, _ in scenario["l2l_edges"]:
            if source_lane in used_lanelets or target_lane in used_lanelets:
                used_lanelets.update((source_lane, target_lane))

        lane_ids = sorted(lid for lid in used_lanelets if lid in scenario["lanelets"])
        lane_index = {lid: i for i, lid in enumerate(lane_ids)}
        geometry_cache = scenario.get("lane_geometry_cache", {})
        lane_sequences = [
            torch.tensor(geometry_cache.get(lid) or lane_local_geometry(scenario["lanelets"][lid]), dtype=torch.float32)
            for lid in lane_ids
        ]
        if lane_sequences:
            lane_geometry_lengths = torch.tensor([seq.size(0) for seq in lane_sequences], dtype=torch.long)
            lane_geometry = pad_sequence(lane_sequences, batch_first=True)
            lane_x = _float_tensor(
                [lane_static_feature_vector(scenario["lanelets"][lid], origin) for lid in lane_ids]
            )
        else:
            lane_geometry_lengths = torch.empty((0,), dtype=torch.long)
            lane_geometry = torch.empty((0, 1, 4), dtype=torch.float32)
            lane_x = torch.empty((0, 4), dtype=torch.float32)

        # L2L edges and Table-II features.
        l2l_pairs: list[list[int]] = []
        l2l_numeric: list[list[float]] = []
        l2l_types: list[int] = []
        l2l_cache = scenario.get("l2l_numeric_cache", {})
        for source_lane, target_lane, relation_type in scenario["l2l_edges"]:
            if source_lane not in lane_index or target_lane not in lane_index:
                continue
            l2l_pairs.append([lane_index[source_lane], lane_index[target_lane]])
            l2l_numeric.append(
                l2l_cache.get((source_lane, target_lane))
                or l2l_numeric_feature_vector(scenario["lanelets"][source_lane], scenario["lanelets"][target_lane])
            )
            l2l_types.append(int(relation_type))

        # Paper default V2V: Voronoi neighbours, implemented through Delaunay triangulation.
        v2v_pairs: list[list[int]] = []
        v2v_attributes: list[list[float]] = []
        for time_index, t in enumerate(observed):
            active_ids = active_by_time[time_index]
            points = [
                (float(vehicles[vid]["states"][t]["x"]), float(vehicles[vid]["states"][t]["y"]))
                for vid in active_ids
            ]
            for source_local, target_local in delaunay_directed_edges(points):
                source_vid = active_ids[source_local]
                target_vid = active_ids[target_local]
                v2v_pairs.append([
                    node_index[(time_index, source_vid)],
                    node_index[(time_index, target_vid)],
                ])
                v2v_attributes.append(
                    v2v_feature_vector(
                        vehicles[source_vid]["states"][t],
                        vehicles[target_vid]["states"][t],
                    )
                )

        # V2L and the reverse L2V relation contained in the paper graph type set.
        v2l_pairs: list[list[int]] = []
        l2v_pairs: list[list[int]] = []
        v2l_attributes: list[list[float]] = []
        l2v_attributes: list[list[float]] = []
        for time_index, t in enumerate(observed):
            for vid in active_by_time[time_index]:
                vehicle_node = node_index[(time_index, vid)]
                state = vehicles[vid]["states"][t]
                for lid in assignments[time_index][vid]:
                    if lid not in lane_index:
                        continue
                    lane_node = lane_index[lid]
                    attr = v2l_feature_vector(state, scenario["lanelets"][lid])
                    v2l_pairs.append([vehicle_node, lane_node])
                    v2l_attributes.append(attr)
                    l2v_pairs.append([lane_node, vehicle_node])
                    l2v_attributes.append(attr)

        # Default causal VTV: historical realization -> future realization of the same vehicle.
        vtv_pairs: list[list[int]] = []
        vtv_motion: list[list[float]] = []
        vtv_delta_t: list[float] = []
        all_observed_vehicle_ids = sorted(set().union(*[set(ids) for ids in active_by_time]))
        for vid in all_observed_vehicle_ids:
            present = [time_index for time_index in range(self.obs_steps) if (time_index, vid) in node_index]
            for i, source_time_index in enumerate(present):
                for target_time_index in present[i + 1 :]:
                    if target_time_index - source_time_index > VTV_MAX_FUTURE_STEPS:
                        break
                    source_t = observed[source_time_index]
                    target_t = observed[target_time_index]
                    vtv_pairs.append([
                        node_index[(source_time_index, vid)],
                        node_index[(target_time_index, vid)],
                    ])
                    vtv_motion.append(
                        v2v_feature_vector(
                            vehicles[vid]["states"][source_t],
                            vehicles[vid]["states"][target_t],
                        )
                    )
                    vtv_delta_t.append((target_time_index - source_time_index) * self.model_dt)

        # Targets correspond to vehicles present at the latest observation.
        current_position: list[list[float]] = []
        current_orientation: list[float] = []
        target_position: list[list[list[float]]] = []
        target_orientation: list[list[float]] = []
        target_mask: list[bool] = []
        for vid in latest_vehicle_ids:
            current = vehicles[vid]["states"][latest_time]
            current_position.append([float(current["x"]) - origin_x, float(current["y"]) - origin_y])
            current_orientation.append(float(current.get("orientation", 0.0)))
            valid = all(t in vehicles[vid]["states"] for t in future)
            target_mask.append(valid)
            if valid:
                target_position.append([
                    [float(vehicles[vid]["states"][t]["x"]) - origin_x, float(vehicles[vid]["states"][t]["y"]) - origin_y]
                    for t in future
                ])
                target_orientation.append([
                    float(vehicles[vid]["states"][t].get("orientation", 0.0)) for t in future
                ])
            else:
                target_position.append([
                    [float(current["x"]) - origin_x, float(current["y"]) - origin_y] for _ in future
                ])
                target_orientation.append([float(current.get("orientation", 0.0)) for _ in future])

        return {
            "vehicle_x": vehicle_x,
            "vehicle_node_ids": torch.tensor(vehicle_node_ids, dtype=torch.long),
            "prediction_vehicle_ids": torch.tensor(latest_vehicle_ids, dtype=torch.long),
            "latest_vehicle_node_index": latest_vehicle_node_index,
            "lane_x": lane_x,
            "lane_geometry": lane_geometry,
            "lane_geometry_lengths": lane_geometry_lengths,
            "edge_index": {
                "v2v": _edge_index(v2v_pairs),
                "v2l": _edge_index(v2l_pairs),
                "l2v": _edge_index(l2v_pairs),
                "l2l": _edge_index(l2l_pairs),
                "vtv": _edge_index(vtv_pairs),
            },
            "edge_attr": {
                "v2v": _edge_attr(v2v_attributes, V2V_EDGE_DIM),
                "v2l": _edge_attr(v2l_attributes, V2L_EDGE_DIM),
                "l2v": _edge_attr(l2v_attributes, V2L_EDGE_DIM),
                "l2l_numeric": _edge_attr(l2l_numeric, L2L_NUMERIC_EDGE_DIM),
                "l2l_type": torch.tensor(l2l_types, dtype=torch.long) if l2l_types else torch.empty((0,), dtype=torch.long),
                "vtv_motion": _edge_attr(vtv_motion, V2V_EDGE_DIM),
                "vtv_delta_t": _float_tensor(vtv_delta_t) if vtv_delta_t else torch.empty((0,), dtype=torch.float32),
            },
            "current_position": _float_tensor(current_position),
            "current_orientation": _float_tensor(current_orientation),
            "target_position": _float_tensor(target_position),
            "target_orientation": _float_tensor(target_orientation),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "prediction_times": torch.tensor(
                [(k + 1) * self.model_dt for k in range(self.pred_steps)], dtype=torch.float32
            ),
            "meta": {
                "benchmark_id": scenario["benchmark_id"],
                "source_file": scenario.get("source_file"),
                "origin_xy": origin,
                "observation_time_steps": observed,
                "future_time_steps": future,
                "v2v_drawer": "VoronoiEdgeDrawer/Delaunay",
                "vtv_drawer": "CausalEdgeDrawer",
            },
        }
