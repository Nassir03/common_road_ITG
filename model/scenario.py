"""Small CommonRoad reader for the supplied data.

Important detail: some files in the supplied archive have a ``.xml`` filename
but are actually ZIP containers holding a ``*.cr.xml`` CommonRoad scenario.
This reader supports both normal XML files and those ZIP containers.
"""
from __future__ import annotations

import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

import torch

RELATION_TO_ID = {
    "predecessor": 0,
    "successor": 1,
    "adjacent_left": 2,
    "adjacent_right": 3,
    "merging": 4,
    "diverging": 5,
    "conflicting": 6,
}


def _read_root(path: str | Path) -> tuple[ET.Element, str | None]:
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".cr.xml")]
            if not members:
                members = [n for n in archive.namelist() if n.endswith(".xml")]
            if not members:
                raise ValueError(f"No CommonRoad XML member found inside {path}")
            member = members[0]
            return ET.fromstring(archive.read(member)), member
    return ET.parse(path).getroot(), None


def _exact(parent: ET.Element | None, path: str, default: float = 0.0) -> float:
    if parent is None:
        return default
    node = parent.find(path)
    if node is None or node.text is None:
        return default
    try:
        return float(node.text)
    except ValueError:
        return default


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
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    return math.atan2(dy, dx)


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


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_segment_cross(a, b, c, d, eps: float = 1e-9) -> bool:
    """True only for an interior crossing, not a shared endpoint/touch."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _centerlines_conflict(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    # Cheap bounding-box rejection avoids most pairwise segment checks.
    aminx, amaxx = min(x for x, _ in a), max(x for x, _ in a)
    aminy, amaxy = min(y for _, y in a), max(y for _, y in a)
    bminx, bmaxx = min(x for x, _ in b), max(x for x, _ in b)
    bminy, bmaxy = min(y for _, y in b), max(y for _, y in b)
    if amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy:
        return False
    for a0, a1 in zip(a, a[1:]):
        for b0, b1 in zip(b, b[1:]):
            if _proper_segment_cross(a0, a1, b0, b1):
                return True
    return False


def parse_commonroad_xml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    root, xml_member = _read_root(path)
    dt = float(root.attrib.get("timeStepSize", 0.1))
    benchmark_id = root.attrib.get("benchmarkID", path.stem)

    lanelets: dict[int, dict[str, Any]] = {}
    l2l_edges: set[tuple[int, int, int]] = set()
    predecessor_map: dict[int, set[int]] = {}
    successor_map: dict[int, set[int]] = {}

    for node in root.findall("lanelet"):
        lane_id = int(node.attrib["id"])
        left = _polyline(node, "leftBound")
        right = _polyline(node, "rightBound")
        center = _centerline(left, right)
        lanelets[lane_id] = {
            "id": lane_id,
            "left": left,
            "right": right,
            "center": center,
            "length": _length(center),
            "heading": _heading(center),
        }
        predecessors = {int(x.attrib["ref"]) for x in node.findall("predecessor")}
        successors = {int(x.attrib["ref"]) for x in node.findall("successor")}
        predecessor_map[lane_id] = predecessors
        successor_map[lane_id] = successors
        for other in predecessors:
            l2l_edges.add((lane_id, other, RELATION_TO_ID["predecessor"]))
        for other in successors:
            l2l_edges.add((lane_id, other, RELATION_TO_ID["successor"]))
        left_adj = node.find("adjacentLeft")
        right_adj = node.find("adjacentRight")
        # CommonRoad XML names the neighbor relative to the current lanelet:
        # adjacentLeft => target is left of source. Table I in the paper names
        # the relation by the SOURCE relative to TARGET for edge L -> L'.
        # Therefore the paper relation label is intentionally swapped here.
        if left_adj is not None:
            l2l_edges.add((lane_id, int(left_adj.attrib["ref"]), RELATION_TO_ID["adjacent_right"]))
        if right_adj is not None:
            l2l_edges.add((lane_id, int(right_adj.attrib["ref"]), RELATION_TO_ID["adjacent_left"]))

    # Derived L2L relations from topology and geometry.
    lane_ids = sorted(lanelets)
    for i, a in enumerate(lane_ids):
        for b in lane_ids[i + 1:]:
            if successor_map.get(a, set()) & successor_map.get(b, set()):
                l2l_edges.add((a, b, RELATION_TO_ID["merging"]))
                l2l_edges.add((b, a, RELATION_TO_ID["merging"]))
            if predecessor_map.get(a, set()) & predecessor_map.get(b, set()):
                l2l_edges.add((a, b, RELATION_TO_ID["diverging"]))
                l2l_edges.add((b, a, RELATION_TO_ID["diverging"]))
            if _centerlines_conflict(lanelets[a]["center"], lanelets[b]["center"]):
                l2l_edges.add((a, b, RELATION_TO_ID["conflicting"]))
                l2l_edges.add((b, a, RELATION_TO_ID["conflicting"]))

    vehicles: dict[int, dict[str, Any]] = {}
    for obstacle in root.findall("dynamicObstacle"):
        vehicle_id = int(obstacle.attrib["id"])
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
            vehicles[vehicle_id] = {
                "id": vehicle_id,
                "length": _exact(rectangle, "length", 4.5),
                "width": _exact(rectangle, "width", 1.8),
                "states": states,
            }

    return {
        "source_file": str(path),
        "xml_member": xml_member,
        "benchmark_id": benchmark_id,
        "location_group": benchmark_id.split("-")[0],
        "dt": dt,
        "lanelets": lanelets,
        "l2l_edges": sorted(l2l_edges),
        "vehicles": vehicles,
    }


def save_scenario(xml_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parse_commonroad_xml(xml_path), output_path)


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def assign_lanelets(x: float, y: float, lanelets: dict[int, dict[str, Any]]) -> list[int]:
    """CommonRoad-Geometric-style center assignment to all containing lanelets.

    No nearest-lane fallback is used: if the center lies in no lanelet, the
    vehicle simply has no V2L/L2V edge for that time step.
    """
    matches: list[int] = []
    for lane_id, lane in lanelets.items():
        polygon = lane["left"] + list(reversed(lane["right"]))
        if point_in_polygon(x, y, polygon):
            matches.append(lane_id)
    return sorted(matches)
