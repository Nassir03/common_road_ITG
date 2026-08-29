"""Dynamic TraInX-style ROT -> ROC -> ROI -> BFS -> ITG construction."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import math
from typing import Iterable


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    x: float
    y: float
    heading: float
    speed: float
    acceleration: float = 0.0


@dataclass(frozen=True)
class ITGEdge:
    vehicle_of_interest: int
    source: int
    target: int
    hop: int
    branch: int
    direct: bool
    dx_local: float
    dy_local: float
    distance: float
    rel_vx_local: float
    rel_vy_local: float


@dataclass(frozen=True)
class ITGSnapshot:
    time_step: int
    rot: dict
    roc: dict[int, list[int]]
    roi: dict[int, list[int]]
    bfs_hops: dict[int, dict[int, int]]
    edges: list[ITGEdge]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_rot(states: Iterable[VehicleState]) -> dict:
    states = list(states)
    if not states:
        return {"center": None, "radius": 0.0, "vehicles": []}
    cx = sum(v.x for v in states) / len(states)
    cy = sum(v.y for v in states) / len(states)
    distances = {v.vehicle_id: math.hypot(v.x - cx, v.y - cy) for v in states}
    return {"center": [cx, cy], "radius": max(distances.values(), default=0.0), "vehicles": sorted(distances)}


def communication_graph(states: Iterable[VehicleState], radius: float) -> dict[int, list[int]]:
    states = sorted(states, key=lambda s: s.vehicle_id)
    graph = {s.vehicle_id: [] for s in states}
    for i, u in enumerate(states):
        for v in states[i + 1:]:
            if math.hypot(v.x - u.x, v.y - u.y) <= radius:
                graph[u.vehicle_id].append(v.vehicle_id)
                graph[v.vehicle_id].append(u.vehicle_id)
    return {k: sorted(v) for k, v in graph.items()}


def roi_members(states: Iterable[VehicleState], radius: float) -> dict[int, list[int]]:
    states = sorted(states, key=lambda s: s.vehicle_id)
    return {
        center.vehicle_id: sorted(
            other.vehicle_id for other in states
            if other.vehicle_id != center.vehicle_id
            and math.hypot(other.x - center.x, other.y - center.y) <= radius
        )
        for center in states
    }


def bfs_tree(graph: dict[int, list[int]], source: int, allowed: set[int], max_hops: int):
    """FIFO BFS returning visit order, shortest hop count, and parent map."""
    allowed = set(allowed) | {source}
    queue = deque([source])
    visited = {source}
    hops = {source: 0}
    parent: dict[int, int] = {}
    order = [source]
    while queue:
        u = queue.popleft()
        if hops[u] >= max_hops:
            continue
        for v in graph.get(u, []):
            if v not in allowed or v in visited:
                continue
            visited.add(v)
            parent[v] = u
            hops[v] = hops[u] + 1
            order.append(v)
            queue.append(v)
    return order, hops, parent


def _velocity_xy(state: VehicleState) -> tuple[float, float]:
    return state.speed * math.cos(state.heading), state.speed * math.sin(state.heading)


def _edge_features(source: VehicleState, target: VehicleState):
    """Relative pose and velocity in the source vehicle's local frame."""
    dx = target.x - source.x
    dy = target.y - source.y
    c, s = math.cos(source.heading), math.sin(source.heading)
    dx_local = c * dx + s * dy
    dy_local = -s * dx + c * dy
    svx, svy = _velocity_xy(source)
    tvx, tvy = _velocity_xy(target)
    dvx, dvy = tvx - svx, tvy - svy
    rel_vx_local = c * dvx + s * dvy
    rel_vy_local = -s * dvx + c * dvy
    return dx_local, dy_local, math.hypot(dx, dy), rel_vx_local, rel_vy_local


def _edges_for_voi(by_id, roc, roi, voi: int, max_hops: int) -> tuple[list[ITGEdge], dict[int, int]]:
    order, hops, parent = bfs_tree(roc, source=voi, allowed=set(roi[voi]), max_hops=max_hops)
    branch_of: dict[int, int] = {voi: 0}
    next_branch = 1
    edges: list[ITGEdge] = []
    for node in order[1:]:
        p = parent[node]
        if p == voi:
            branch_of[node] = next_branch
            next_branch += 1
        else:
            branch_of[node] = branch_of[p]
        # BFS expands outward; influence is explicitly reversed inward to VOI.
        source, target = by_id[node], by_id[p]
        dx, dy, dist, rvx, rvy = _edge_features(source, target)
        edges.append(ITGEdge(
            vehicle_of_interest=voi,
            source=source.vehicle_id,
            target=target.vehicle_id,
            hop=hops[node],
            branch=branch_of[node],
            direct=hops[node] == 1,
            dx_local=dx,
            dy_local=dy,
            distance=dist,
            rel_vx_local=rvx,
            rel_vy_local=rvy,
        ))
    return edges, hops


def build_itg_for_vehicle(
    states: Iterable[VehicleState],
    vehicle_of_interest: int,
    time_step: int,
    communication_radius: float,
    roi_radius: float,
    max_hops: int,
) -> ITGSnapshot:
    """Build one ego/VOI-specific ITG.

    This avoids ambiguous duplicate hop/branch labels: hop and branch are always
    defined relative to exactly one vehicle of interest in a training sample.
    """
    if roi_radius < communication_radius:
        raise ValueError("ROI radius must be >= ROC/communication radius")
    states = sorted(states, key=lambda s: s.vehicle_id)
    by_id = {s.vehicle_id: s for s in states}
    if vehicle_of_interest not in by_id:
        raise KeyError(f"VOI {vehicle_of_interest} is not present at time step {time_step}")
    roc = communication_graph(states, communication_radius)
    roi = roi_members(states, roi_radius)
    edges, hops = _edges_for_voi(by_id, roc, roi, vehicle_of_interest, max_hops)
    return ITGSnapshot(
        time_step=time_step,
        rot=compute_rot(states),
        roc=roc,
        roi=roi,
        bfs_hops={vehicle_of_interest: hops},
        edges=edges,
    )


def build_itg_snapshot(
    states: Iterable[VehicleState],
    time_step: int,
    communication_radius: float,
    roi_radius: float,
    max_hops: int,
) -> ITGSnapshot:
    """Build the union of per-vehicle ITGs, useful for visualization/cr-geo."""
    if roi_radius < communication_radius:
        raise ValueError("ROI radius must be >= ROC/communication radius")
    states = sorted(states, key=lambda s: s.vehicle_id)
    by_id = {s.vehicle_id: s for s in states}
    roc = communication_graph(states, communication_radius)
    roi = roi_members(states, roi_radius)
    all_edges: list[ITGEdge] = []
    all_hops: dict[int, dict[int, int]] = {}
    for voi in sorted(by_id):
        edges, hops = _edges_for_voi(by_id, roc, roi, voi, max_hops)
        all_edges.extend(edges)
        all_hops[voi] = hops
    return ITGSnapshot(time_step, compute_rot(states), roc, roi, all_hops, all_edges)


def build_radius_baseline_edges(states: Iterable[VehicleState], communication_radius: float) -> list[ITGEdge]:
    """Proximity-only baseline: bidirectional ROC edges with no multi-hop labels."""
    states = sorted(states, key=lambda s: s.vehicle_id)
    by_id = {s.vehicle_id: s for s in states}
    graph = communication_graph(states, communication_radius)
    edges: list[ITGEdge] = []
    for source_id, neighbours in graph.items():
        for target_id in neighbours:
            source, target = by_id[source_id], by_id[target_id]
            dx, dy, dist, rvx, rvy = _edge_features(source, target)
            edges.append(ITGEdge(
                vehicle_of_interest=target_id,
                source=source_id,
                target=target_id,
                hop=1,
                branch=0,
                direct=True,
                dx_local=dx,
                dy_local=dy,
                distance=dist,
                rel_vx_local=rvx,
                rel_vy_local=rvy,
            ))
    return edges
