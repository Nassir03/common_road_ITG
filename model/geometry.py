"""Geometry and feature extraction matching Table II of the cr-geo paper."""
from __future__ import annotations

import math
import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotate_to_local(dx: float, dy: float, heading: float) -> tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return c * dx + s * dy, -s * dx + c * dy


def state_velocity_xy(state: dict) -> tuple[float, float]:
    speed = float(state.get("velocity", 0.0))
    heading = float(state.get("orientation", 0.0))
    return speed * math.cos(heading), speed * math.sin(heading)


def state_acceleration_xy(state: dict) -> tuple[float, float]:
    acceleration = float(state.get("acceleration", 0.0))
    heading = float(state.get("orientation", 0.0))
    return acceleration * math.cos(heading), acceleration * math.sin(heading)


def vehicle_feature_vector(vehicle: dict, state: dict, origin: tuple[float, float] = (0.0, 0.0)) -> list[float]:
    """Table-II vehicle features.

    A common translation origin is subtracted from pV only for float32 numerical
    precision. Distances, relative geometry, and ADE/FDE are unchanged.
    """
    vx, vy = state_velocity_xy(state)
    ax, ay = state_acceleration_xy(state)
    return [
        float(state["x"]) - origin[0],
        float(state["y"]) - origin[1],
        float(state.get("orientation", 0.0)),
        float(state.get("yaw_rate", 0.0)),
        vx,
        vy,
        ax,
        ay,
        float(vehicle.get("width", 0.0)),
        float(vehicle.get("length", 0.0)),
    ]


def v2v_feature_vector(source: dict, target: dict) -> list[float]:
    """Table-II V2V features in the source-vehicle local coordinate frame."""
    sx, sy = float(source["x"]), float(source["y"])
    tx, ty = float(target["x"]), float(target["y"])
    sh = float(source.get("orientation", 0.0))
    th = float(target.get("orientation", 0.0))
    dx, dy = rotate_to_local(tx - sx, ty - sy, sh)
    distance = math.hypot(tx - sx, ty - sy)

    svx, svy = state_velocity_xy(source)
    tvx, tvy = state_velocity_xy(target)
    rvx, rvy = rotate_to_local(tvx - svx, tvy - svy, sh)

    sax, say = state_acceleration_xy(source)
    tax, tay = state_acceleration_xy(target)
    rax, ray = rotate_to_local(tax - sax, tay - say, sh)

    return [distance, dx, dy, wrap_angle(th - sh), rvx, rvy, rax, ray]


def _project_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denominator = vx * vx + vy * vy
    if denominator <= 1e-12:
        return (ax, ay), 0.0, math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denominator))
    q = (ax + u * vx, ay + u * vy)
    return q, u, math.hypot(px - q[0], py - q[1])


def project_to_polyline(point, points):
    if not points:
        return point, 0.0, 0.0, 0.0
    if len(points) == 1:
        return points[0], 0.0, 0.0, math.dist(point, points[0])
    best = None
    prefix = 0.0
    for a, b in zip(points, points[1:]):
        segment_length = math.dist(a, b)
        q, u, distance = _project_segment(point, a, b)
        heading = math.atan2(b[1] - a[1], b[0] - a[0]) if segment_length > 1e-12 else 0.0
        candidate = (q, prefix + u * segment_length, heading, distance)
        if best is None or distance < best[3]:
            best = candidate
        prefix += segment_length
    return best


def v2l_feature_vector(state: dict, lane: dict) -> list[float]:
    """Table-II V2L Center-assignment features."""
    point = (float(state["x"]), float(state["y"]))
    _, _, _, left_distance = project_to_polyline(point, lane.get("left", []))
    _, _, _, right_distance = project_to_polyline(point, lane.get("right", []))
    _, arclength, lane_heading, _ = project_to_polyline(point, lane.get("center", []))
    lateral_offset = (left_distance - right_distance) / 2.0
    lane_length = max(float(lane.get("length", 0.0)), 1e-12)
    return [
        left_distance,
        right_distance,
        lateral_offset,
        wrap_angle(lane_heading - float(state.get("orientation", 0.0))),
        arclength,
        arclength / lane_length,
    ]


def lane_local_geometry(lane: dict) -> list[list[float]]:
    """Lane-local left/right waypoint sequences for the lanelet GRU."""
    left = lane.get("left", [])
    right = lane.get("right", [])
    n = min(len(left), len(right))
    if n == 0:
        return [[0.0, 0.0, 0.0, 0.0]]
    center = lane.get("center", [])
    origin = center[0] if center else ((left[0][0] + right[0][0]) / 2.0, (left[0][1] + right[0][1]) / 2.0)
    heading = float(lane.get("heading", 0.0))
    rows: list[list[float]] = []
    for left_point, right_point in zip(left[:n], right[:n]):
        llx, lly = rotate_to_local(left_point[0] - origin[0], left_point[1] - origin[1], heading)
        rlx, rly = rotate_to_local(right_point[0] - origin[0], right_point[1] - origin[1], heading)
        rows.append([llx, lly, rlx, rly])
    return rows


def lane_static_feature_vector(lane: dict, origin: tuple[float, float] = (0.0, 0.0)) -> list[float]:
    center = lane.get("center", [])
    lane_origin = center[0] if center else (0.0, 0.0)
    return [
        float(lane_origin[0]) - origin[0],
        float(lane_origin[1]) - origin[1],
        float(lane.get("length", 0.0)),
        float(lane.get("heading", 0.0)),
    ]


def _segment_intersection(a, b, c, d, eps: float = 1e-9):
    r = np.asarray([b[0] - a[0], b[1] - a[1]], dtype=float)
    s = np.asarray([d[0] - c[0], d[1] - c[1]], dtype=float)
    qp = np.asarray([c[0] - a[0], c[1] - a[1]], dtype=float)
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= eps:
        return None
    t = float((qp[0] * s[1] - qp[1] * s[0]) / cross)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / cross)
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return max(0.0, min(1.0, t)), max(0.0, min(1.0, u))
    return None


def intersection_arclengths(a_points, b_points) -> tuple[float, float]:
    a_prefix = 0.0
    for a0, a1 in zip(a_points, a_points[1:]):
        a_length = math.dist(a0, a1)
        b_prefix = 0.0
        for b0, b1 in zip(b_points, b_points[1:]):
            b_length = math.dist(b0, b1)
            hit = _segment_intersection(a0, a1, b0, b1)
            if hit is not None:
                ta, tb = hit
                return a_prefix + ta * a_length, b_prefix + tb * b_length
            b_prefix += b_length
        a_prefix += a_length
    return 0.0, 0.0


def l2l_numeric_feature_vector(source: dict, target: dict) -> list[float]:
    """Table-II L2L numeric features; adjacency type is encoded separately."""
    source_center = source.get("center", [])
    target_center = target.get("center", [])
    source_origin = source_center[0] if source_center else (0.0, 0.0)
    target_origin = target_center[0] if target_center else (0.0, 0.0)
    source_heading = float(source.get("heading", 0.0))
    target_heading = float(target.get("heading", 0.0))
    dx, dy = rotate_to_local(
        target_origin[0] - source_origin[0],
        target_origin[1] - source_origin[1],
        source_heading,
    )
    source_s, target_s = intersection_arclengths(source_center, target_center)
    return [
        math.dist(source_origin, target_origin),
        dx,
        dy,
        wrap_angle(target_heading - source_heading),
        source_s,
        target_s,
    ]


def _knn_directed_edges(points: list[tuple[float, float]], k: int = 3) -> list[tuple[int, int]]:
    """Paper-supported KNearestEdgeDrawer fallback for degenerate Delaunay input."""
    n = len(points)
    if n < 2:
        return []
    array = np.asarray(points, dtype=float)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        distances = np.linalg.norm(array - array[i], axis=1)
        order = np.argsort(distances)
        for j in order[1 : min(n, k + 1)]:
            edges.add((i, int(j)))
    return sorted(edges)


def delaunay_directed_edges(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """VoronoiEdgeDrawer semantics: Delaunay-neighbour V2V edges.

    Qhull's joggling option handles most collinear/duplicate snapshots. If it
    still cannot triangulate a degenerate snapshot, the paper's K-nearest edge
    drawer is used as a documented fallback instead of constructing O(N^2)
    complete graphs.
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1), (1, 0)]
    try:
        from scipy.spatial import Delaunay

        triangulation = Delaunay(np.asarray(points, dtype=float), qhull_options="QJ")
        undirected: set[tuple[int, int]] = set()
        for simplex in triangulation.simplices:
            values = [int(v) for v in simplex]
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a, b = sorted((values[i], values[j]))
                    if a != b:
                        undirected.add((a, b))
        return sorted([(a, b) for a, b in undirected] + [(b, a) for a, b in undirected])
    except Exception:
        return _knn_directed_edges(points, k=3)
