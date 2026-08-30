"""Small CommonRoad XML reader used by the Kaggle experiment."""
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
                raise ValueError(f"No CommonRoad XML found inside {path}")
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

def _centerline(left, right):
    n = min(len(left), len(right))
    return [((left[i][0] + right[i][0]) / 2.0, (left[i][1] + right[i][1]) / 2.0) for i in range(n)]

def _length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

def _heading(points):
    if len(points) < 2:
        return 0.0
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])

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

def _orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _proper_segment_cross(a, b, c, d, eps=1e-9):
    o1, o2 = _orientation(a,b,c), _orientation(a,b,d)
    o3, o4 = _orientation(c,d,a), _orientation(c,d,b)
    return (o1*o2 < -eps) and (o3*o4 < -eps)

def _centerlines_conflict(a, b):
    if len(a) < 2 or len(b) < 2:
        return False
    aminx, amaxx = min(x for x,_ in a), max(x for x,_ in a)
    aminy, amaxy = min(y for _,y in a), max(y for _,y in a)
    bminx, bmaxx = min(x for x,_ in b), max(x for x,_ in b)
    bminy, bmaxy = min(y for _,y in b), max(y for _,y in b)
    if amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy:
        return False
    return any(_proper_segment_cross(a0,a1,b0,b1) for a0,a1 in zip(a,a[1:]) for b0,b1 in zip(b,b[1:]))

def parse_commonroad_xml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    root, xml_member = _read_root(path)
    dt = float(root.attrib.get("timeStepSize", 0.1))
    benchmark_id = root.attrib.get("benchmarkID", path.stem)

    lanelets: dict[int, dict[str, Any]] = {}
    l2l_edges: set[tuple[int,int,int]] = set()
    predecessors: dict[int,set[int]] = {}
    successors: dict[int,set[int]] = {}
    for node in root.findall("lanelet"):
        lid = int(node.attrib["id"])
        left, right = _polyline(node,"leftBound"), _polyline(node,"rightBound")
        center = _centerline(left,right)
        lanelets[lid] = {"id":lid,"left":left,"right":right,"center":center,"length":_length(center),"heading":_heading(center)}
        pred = {int(x.attrib["ref"]) for x in node.findall("predecessor")}
        succ = {int(x.attrib["ref"]) for x in node.findall("successor")}
        predecessors[lid], successors[lid] = pred, succ
        for other in pred:
            l2l_edges.add((lid,other,RELATION_TO_ID["predecessor"]))
        for other in succ:
            l2l_edges.add((lid,other,RELATION_TO_ID["successor"]))
        left_adj, right_adj = node.find("adjacentLeft"), node.find("adjacentRight")
        # Table I names source relative to target, hence the swapped labels.
        if left_adj is not None:
            l2l_edges.add((lid,int(left_adj.attrib["ref"]),RELATION_TO_ID["adjacent_right"]))
        if right_adj is not None:
            l2l_edges.add((lid,int(right_adj.attrib["ref"]),RELATION_TO_ID["adjacent_left"]))

    lids = sorted(lanelets)
    for i,a in enumerate(lids):
        for b in lids[i+1:]:
            if successors.get(a,set()) & successors.get(b,set()):
                l2l_edges.add((a,b,RELATION_TO_ID["merging"])); l2l_edges.add((b,a,RELATION_TO_ID["merging"]))
            if predecessors.get(a,set()) & predecessors.get(b,set()):
                l2l_edges.add((a,b,RELATION_TO_ID["diverging"])); l2l_edges.add((b,a,RELATION_TO_ID["diverging"]))
            if _centerlines_conflict(lanelets[a]["center"], lanelets[b]["center"]):
                l2l_edges.add((a,b,RELATION_TO_ID["conflicting"])); l2l_edges.add((b,a,RELATION_TO_ID["conflicting"]))

    vehicles: dict[int,dict[str,Any]] = {}
    for obstacle in root.findall("dynamicObstacle"):
        vid = int(obstacle.attrib["id"])
        rect = obstacle.find("shape/rectangle")
        states: dict[int,dict[str,float]] = {}
        initial = obstacle.find("initialState")
        if initial is not None:
            t,s = _parse_state(initial); states[t] = s
        for sn in obstacle.findall("trajectory/state"):
            t,s = _parse_state(sn); states[t] = s
        if states:
            vehicles[vid] = {
                "id":vid,
                "length":_exact(rect,"length",4.5),
                "width":_exact(rect,"width",1.8),
                "states":states,
            }
    return {
        "source_file":str(path),
        "xml_member":xml_member,
        "benchmark_id":benchmark_id,
        "location_group":benchmark_id.split("-")[0],
        "dt":dt,
        "lanelets":lanelets,
        "l2l_edges":sorted(l2l_edges),
        "vehicles":vehicles,
    }

def save_scenario(xml_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parse_commonroad_xml(xml_path), output_path)

def point_in_polygon(x: float, y: float, polygon: list[tuple[float,float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside, j = False, len(polygon)-1
    for i in range(len(polygon)):
        xi,yi = polygon[i]; xj,yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/((yj-yi) or 1e-12)+xi):
            inside = not inside
        j = i
    return inside

def assign_lanelets(x: float, y: float, lanelets: dict[int,dict[str,Any]]) -> list[int]:
    matches=[]
    for lid,lane in lanelets.items():
        polygon = lane["left"] + list(reversed(lane["right"]))
        if point_in_polygon(x,y,polygon):
            matches.append(lid)
    return sorted(matches)
