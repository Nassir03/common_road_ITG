"""Robust CommonRoad XML reader and paper graph preprocessing.

This module extracts only entities/relations described in the attached
CommonRoad-Geometric paper. Expensive relations are accelerated with spatial
indexing but are not omitted.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import torch

from .config import LANE_GRID_CELL_SIZE

RELATION_TO_ID = {
    "predecessor": 0,
    "successor": 1,
    "adjacent_left": 2,
    "adjacent_right": 3,
    "merging": 4,
    "diverging": 5,
    "conflicting": 6,
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _read_root(path: str | Path) -> tuple[ET.Element, str | None]:
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".cr.xml")]
            if not members:
                members = [n for n in archive.namelist() if n.endswith(".xml")]
            if not members:
                raise ValueError(f"No CommonRoad XML found inside {path}")
            member = members[0]
            return ET.fromstring(archive.read(member)), member
    return ET.parse(path).getroot(), None


def _exact(parent: ET.Element | None, path: str, default: float = 0.0) -> float:
    if parent is None:
        return float(default)
    node = parent.find(path)
    return _finite(node.text if node is not None else None, default)


def _point(parent: ET.Element | None, path: str = "position/point") -> tuple[float, float]:
    if parent is None:
        return 0.0, 0.0
    node = parent.find(path)
    return _exact(node, "x"), _exact(node, "y")


def _polyline(lanelet: ET.Element, tag: str) -> list[tuple[float, float]]:
    return [(_exact(p, "x"), _exact(p, "y")) for p in lanelet.findall(f"{tag}/point")]


def _centerline(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> list[tuple[float, float]]:
    n = min(len(left), len(right))
    return [((left[i][0] + right[i][0]) / 2.0, (left[i][1] + right[i][1]) / 2.0) for i in range(n)]


def _length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _heading(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])


def _bbox(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = left + right
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_state(state: ET.Element) -> tuple[int, dict[str, float]]:
    time_step = int(round(_exact(state, "time/exact")))
    x, y = _point(state)
    return time_step, {
        "x": x,
        "y": y,
        "orientation": _exact(state, "orientation/exact"),
        "velocity": _exact(state, "velocity/exact"),
        "acceleration": _exact(state, "acceleration/exact"),
        "yaw_rate": _exact(state, "yawRate/exact"),
    }


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_segment_cross(a, b, c, d, eps: float = 1e-9) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _centerlines_conflict(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    aminx, amaxx = min(x for x, _ in a), max(x for x, _ in a)
    aminy, amaxy = min(y for _, y in a), max(y for _, y in a)
    bminx, bmaxx = min(x for x, _ in b), max(x for x, _ in b)
    bminy, bmaxy = min(y for _, y in b), max(y for _, y in b)
    if amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy:
        return False
    return any(
        _proper_segment_cross(a0, a1, b0, b1)
        for a0, a1 in zip(a, a[1:])
        for b0, b1 in zip(b, b[1:])
    )


def build_lane_grid(lanelets: dict[int, dict[str, Any]], cell_size: float = LANE_GRID_CELL_SIZE) -> dict[tuple[int, int], tuple[int, ...]]:
    """Spatial index used only to reduce candidate checks."""
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cs = float(cell_size)
    for lid, lane in lanelets.items():
        minx, miny, maxx, maxy = lane["bbox"]
        for gx in range(math.floor(minx / cs), math.floor(maxx / cs) + 1):
            for gy in range(math.floor(miny / cs), math.floor(maxy / cs) + 1):
                grid[(gx, gy)].append(lid)
    return {cell: tuple(sorted(set(ids))) for cell, ids in grid.items()}


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def assign_lanelets(
    x: float,
    y: float,
    lanelets: dict[int, dict[str, Any]],
    lane_grid: dict[tuple[int, int], tuple[int, ...]] | None = None,
    cell_size: float = LANE_GRID_CELL_SIZE,
) -> list[int]:
    """Paper's Center V2L strategy: connect to every containing lanelet."""
    x, y = _finite(x), _finite(y)
    if lane_grid:
        candidates = lane_grid.get((math.floor(x / cell_size), math.floor(y / cell_size)), ())
    else:
        candidates = lanelets.keys()
    matches: list[int] = []
    for lid in candidates:
        lane = lanelets.get(lid)
        if lane is None:
            continue
        minx, miny, maxx, maxy = lane["bbox"]
        if x < minx or x > maxx or y < miny or y > maxy:
            continue
        polygon = lane["left"] + list(reversed(lane["right"]))
        if point_in_polygon(x, y, polygon):
            matches.append(lid)
    return sorted(matches)


def _add_pairwise_from_groups(groups: dict[int, list[int]], relation_id: int, edges: set[tuple[int, int, int]]) -> None:
    for members in groups.values():
        vals = sorted(set(members))
        for i, a in enumerate(vals):
            for b in vals[i + 1 :]:
                edges.add((a, b, relation_id))
                edges.add((b, a, relation_id))


def _conflict_candidate_pairs(lanelets: dict[int, dict[str, Any]], cell_size: float = LANE_GRID_CELL_SIZE) -> set[tuple[int, int]]:
    """Return only lanelet pairs whose bounding boxes share a spatial cell."""
    grid = build_lane_grid(lanelets, cell_size)
    pairs: set[tuple[int, int]] = set()
    for ids in grid.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if a != b:
                    pairs.add((min(a, b), max(a, b)))
    return pairs


def _precompute_training_cache(scenario: dict[str, Any]) -> None:
    """Cache exact paper features/assignments once so epochs do not recompute them."""
    from .geometry import lane_local_geometry, l2l_numeric_feature_vector

    lanelets = scenario["lanelets"]
    grid = build_lane_grid(lanelets)
    scenario["lane_grid"] = grid
    scenario["lane_geometry_cache"] = {lid: lane_local_geometry(lane) for lid, lane in lanelets.items()}
    scenario["l2l_numeric_cache"] = {
        (a, b): l2l_numeric_feature_vector(lanelets[a], lanelets[b])
        for a, b, _ in scenario["l2l_edges"]
        if a in lanelets and b in lanelets
    }
    assignments: dict[int, dict[int, tuple[int, ...]]] = {}
    for vid, vehicle in scenario["vehicles"].items():
        assignments[vid] = {}
        for t, state in vehicle["states"].items():
            assignments[vid][t] = tuple(assign_lanelets(state["x"], state["y"], lanelets, grid))
    scenario["lane_assignments"] = assignments


def parse_commonroad_xml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    root, xml_member = _read_root(path)
    dt = _finite(root.attrib.get("timeStepSize", 0.1), 0.1)
    benchmark_id = root.attrib.get("benchmarkID", path.stem)

    lanelets: dict[int, dict[str, Any]] = {}
    l2l_edges: set[tuple[int, int, int]] = set()
    predecessors: dict[int, set[int]] = {}
    successors: dict[int, set[int]] = {}
    by_successor: dict[int, list[int]] = defaultdict(list)
    by_predecessor: dict[int, list[int]] = defaultdict(list)

    for node in root.findall("lanelet"):
        lid = int(node.attrib["id"])
        left = _polyline(node, "leftBound")
        right = _polyline(node, "rightBound")
        center = _centerline(left, right)
        lanelets[lid] = {
            "id": lid,
            "left": left,
            "right": right,
            "center": center,
            "length": _length(center),
            "heading": _heading(center),
            "bbox": _bbox(left, right),
        }

        pred = {int(x.attrib["ref"]) for x in node.findall("predecessor")}
        succ = {int(x.attrib["ref"]) for x in node.findall("successor")}
        predecessors[lid], successors[lid] = pred, succ
        for other in pred:
            l2l_edges.add((lid, other, RELATION_TO_ID["predecessor"]))
            by_predecessor[other].append(lid)
        for other in succ:
            l2l_edges.add((lid, other, RELATION_TO_ID["successor"]))
            by_successor[other].append(lid)

        adjacent_left = node.find("adjacentLeft")
        adjacent_right = node.find("adjacentRight")
        # Table I describes the source lanelet relative to the target lanelet.
        if adjacent_left is not None:
            l2l_edges.add((lid, int(adjacent_left.attrib["ref"]), RELATION_TO_ID["adjacent_right"]))
        if adjacent_right is not None:
            l2l_edges.add((lid, int(adjacent_right.attrib["ref"]), RELATION_TO_ID["adjacent_left"]))

    # Merging/diverging are exact Table-I relations and can be derived cheaply.
    _add_pairwise_from_groups(by_successor, RELATION_TO_ID["merging"], l2l_edges)
    _add_pairwise_from_groups(by_predecessor, RELATION_TO_ID["diverging"], l2l_edges)

    # Conflicting lanelets are also in Table I. Spatial indexing avoids an
    # expensive all-pairs scan while preserving the relation.
    for a, b in _conflict_candidate_pairs(lanelets):
        if _centerlines_conflict(lanelets[a]["center"], lanelets[b]["center"]):
            l2l_edges.add((a, b, RELATION_TO_ID["conflicting"]))
            l2l_edges.add((b, a, RELATION_TO_ID["conflicting"]))

    vehicles: dict[int, dict[str, Any]] = {}
    for obstacle in root.findall("dynamicObstacle"):
        vid = int(obstacle.attrib["id"])
        rectangle = obstacle.find("shape/rectangle")
        states: dict[int, dict[str, float]] = {}
        initial = obstacle.find("initialState")
        if initial is not None:
            t, state = _parse_state(initial)
            states[t] = state
        for state_node in obstacle.findall("trajectory/state"):
            t, state = _parse_state(state_node)
            states[t] = state
        if states:
            vehicles[vid] = {
                "id": vid,
                "length": _exact(rectangle, "length", 4.5),
                "width": _exact(rectangle, "width", 1.8),
                "states": states,
            }

    scenario: dict[str, Any] = {
        "format_version": 3,
        "source_file": str(path),
        "xml_member": xml_member,
        "benchmark_id": benchmark_id,
        "location_group": benchmark_id.split("-")[0],
        "dt": dt,
        "lanelets": lanelets,
        "l2l_edges": sorted(l2l_edges),
        "vehicles": vehicles,
    }
    _precompute_training_cache(scenario)
    return scenario


def save_scenario(xml_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parse_commonroad_xml(xml_path), output_path)
