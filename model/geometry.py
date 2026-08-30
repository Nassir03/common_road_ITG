"""Geometry used to construct the heterogeneous traffic graph in the paper."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_to_local(dx: float, dy: float, source_heading: float) -> tuple[float, float]:
    """Rotate a world-frame vector into the source entity's local frame."""
    c = math.cos(source_heading)
    s = math.sin(source_heading)
    # R(-theta) @ [dx, dy]
    return c * dx + s * dy, -s * dx + c * dy


def rotate_to_world(dx_local: float, dy_local: float, heading: float) -> tuple[float, float]:
    """Rotate a local-frame vector into the world frame."""
    c = math.cos(heading)
    s = math.sin(heading)
    return c * dx_local - s * dy_local, s * dx_local + c * dy_local


def state_velocity_xy(state: dict[str, float]) -> tuple[float, float]:
    speed = float(state.get("velocity", 0.0))
    heading = float(state.get("orientation", 0.0))
    return speed * math.cos(heading), speed * math.sin(heading)


def state_acceleration_xy(state: dict[str, float]) -> tuple[float, float]:
    acceleration = float(state.get("acceleration", 0.0))
    heading = float(state.get("orientation", 0.0))
    return acceleration * math.cos(heading), acceleration * math.sin(heading)


def vehicle_feature_vector(vehicle: dict, state: dict[str, float]) -> list[float]:
    """Table-II vehicle features: p, theta, yaw-rate, v, a, width, length."""
    vx, vy = state_velocity_xy(state)
    ax, ay = state_acceleration_xy(state)
    return [
        float(state["x"]),
        float(state["y"]),
        float(state.get("orientation", 0.0)),
        float(state.get("yaw_rate", 0.0)),
        vx,
        vy,
        ax,
        ay,
        float(vehicle.get("width", 0.0)),
        float(vehicle.get("length", 0.0)),
    ]


def v2v_feature_vector(source: dict[str, float], target: dict[str, float]) -> list[float]:
    """Table-II V2V features, expressed in the source vehicle frame."""
    sx, sy = float(source["x"]), float(source["y"])
    tx, ty = float(target["x"]), float(target["y"])
    sh = float(source.get("orientation", 0.0))
    th = float(target.get("orientation", 0.0))

    dx_local, dy_local = rotate_to_local(tx - sx, ty - sy, sh)
    distance = math.hypot(tx - sx, ty - sy)

    svx, svy = state_velocity_xy(source)
    tvx, tvy = state_velocity_xy(target)
    rvx, rvy = rotate_to_local(tvx - svx, tvy - svy, sh)

    sax, say = state_acceleration_xy(source)
    tax, tay = state_acceleration_xy(target)
    rax, ray = rotate_to_local(tax - sax, tay - say, sh)

    return [
        distance,
        dx_local,
        dy_local,
        wrap_angle(th - sh),
        rvx,
        rvy,
        rax,
        ray,
    ]


def _project_point_to_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[tuple[float, float], float, float]:
    """Return projection, segment fraction u in [0,1], and distance."""
    px, py = point
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        q = (ax, ay)
        return q, 0.0, math.hypot(px - ax, py - ay)
    u = ((px - ax) * vx + (py - ay) * vy) / denom
    u = max(0.0, min(1.0, u))
    qx, qy = ax + u * vx, ay + u * vy
    return (qx, qy), u, math.hypot(px - qx, py - qy)


def project_to_polyline(
    point: tuple[float, float],
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], float, float, float]:
    """Nearest polyline projection.

    Returns (projected_point, arclength, tangent_heading, euclidean_distance).
    """
    if not points:
        return point, 0.0, 0.0, 0.0
    if len(points) == 1:
        return points[0], 0.0, 0.0, math.dist(point, points[0])

    best = None
    prefix = 0.0
    for a, b in zip(points, points[1:]):
        seg_len = math.dist(a, b)
        q, u, dist = _project_point_to_segment(point, a, b)
        heading = math.atan2(b[1] - a[1], b[0] - a[0]) if seg_len > 1e-12 else 0.0
        arclength = prefix + u * seg_len
        candidate = (q, arclength, heading, dist)
        if best is None or dist < best[3]:
            best = candidate
        prefix += seg_len
    assert best is not None
    return best


def v2l_feature_vector(state: dict[str, float], lane: dict) -> list[float]:
    """Paper Eq. (2)-(4) and Table-II V2L features."""
    point = (float(state["x"]), float(state["y"]))
    _, _, _, d_left = project_to_polyline(point, lane.get("left", []))
    _, _, _, d_right = project_to_polyline(point, lane.get("right", []))
    _, arclength, lane_heading, _ = project_to_polyline(point, lane.get("center", []))
    lateral_offset = (d_left - d_right) / 2.0
    heading_error = wrap_angle(lane_heading - float(state.get("orientation", 0.0)))
    lane_length = max(float(lane.get("length", 0.0)), 1e-12)
    return [
        d_left,
        d_right,
        lateral_offset,
        heading_error,
        arclength,
        arclength / lane_length,
    ]


def lane_local_geometry(lane: dict) -> list[list[float]]:
    """Lane-local [left_x,left_y,right_x,right_y] waypoint sequence."""
    left = lane.get("left", [])
    right = lane.get("right", [])
    n = min(len(left), len(right))
    if n == 0:
        return [[0.0, 0.0, 0.0, 0.0]]
    center = lane.get("center", [])
    origin = center[0] if center else ((left[0][0] + right[0][0]) / 2.0, (left[0][1] + right[0][1]) / 2.0)
    heading = float(lane.get("heading", 0.0))
    rows: list[list[float]] = []
    for l, r in zip(left[:n], right[:n]):
        llx, lly = rotate_to_local(l[0] - origin[0], l[1] - origin[1], heading)
        rlx, rly = rotate_to_local(r[0] - origin[0], r[1] - origin[1], heading)
        rows.append([llx, lly, rlx, rly])
    return rows


def lane_static_feature_vector(lane: dict) -> list[float]:
    center = lane.get("center", [])
    origin = center[0] if center else (0.0, 0.0)
    return [float(origin[0]), float(origin[1]), float(lane.get("length", 0.0)), float(lane.get("heading", 0.0))]


def _segment_intersection(
    a: tuple[float, float], b: tuple[float, float],
    c: tuple[float, float], d: tuple[float, float],
    eps: float = 1e-9,
) -> tuple[float, float, tuple[float, float]] | None:
    """Return segment parameters (t,u) and point, including endpoint touches."""
    r = np.asarray([b[0] - a[0], b[1] - a[1]], dtype=float)
    s = np.asarray([d[0] - c[0], d[1] - c[1]], dtype=float)
    qp = np.asarray([c[0] - a[0], c[1] - a[1]], dtype=float)
    cross_rs = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross_rs) <= eps:
        return None
    t = float((qp[0] * s[1] - qp[1] * s[0]) / cross_rs)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / cross_rs)
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        x = a[0] + t * r[0]
        y = a[1] + t * r[1]
        return max(0.0, min(1.0, t)), max(0.0, min(1.0, u)), (float(x), float(y))
    return None


def polyline_intersection_arclengths(
    a_points: list[tuple[float, float]],
    b_points: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return first centerline intersection arclengths; (0,0) if none."""
    a_prefix = 0.0
    for a0, a1 in zip(a_points, a_points[1:]):
        a_len = math.dist(a0, a1)
        b_prefix = 0.0
        for b0, b1 in zip(b_points, b_points[1:]):
            b_len = math.dist(b0, b1)
            hit = _segment_intersection(a0, a1, b0, b1)
            if hit is not None:
                ta, tb, _ = hit
                return a_prefix + ta * a_len, b_prefix + tb * b_len
            b_prefix += b_len
        a_prefix += a_len
    return 0.0, 0.0


def l2l_numeric_feature_vector(source_lane: dict, target_lane: dict) -> list[float]:
    """Table-II L2L numeric features (relation type is embedded separately)."""
    sc = source_lane.get("center", [])
    tc = target_lane.get("center", [])
    sp = sc[0] if sc else (0.0, 0.0)
    tp = tc[0] if tc else (0.0, 0.0)
    sh = float(source_lane.get("heading", 0.0))
    th = float(target_lane.get("heading", 0.0))
    dx_local, dy_local = rotate_to_local(tp[0] - sp[0], tp[1] - sp[1], sh)
    distance = math.dist(sp, tp)
    s_source, s_target = polyline_intersection_arclengths(sc, tc)
    return [distance, dx_local, dy_local, wrap_angle(th - sh), s_source, s_target]


def delaunay_directed_edges(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """VoronoiEdgeDrawer equivalent: directed neighbors from Delaunay triangulation.

    CommonRoad-Geometric's paper states VoronoiEdgeDrawer as the default V2V
    drawer and describes it through its dual Delaunay triangulation (Fig. 5a).
    Degenerate scenes fall back to a complete directed graph so graph creation
    remains defined for 1-D/duplicate point arrangements.
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1), (1, 0)]
    try:
        from scipy.spatial import Delaunay, QhullError
        xy = np.asarray(points, dtype=float)
        tri = Delaunay(xy)
        undirected: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            simplex = [int(v) for v in simplex]
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = sorted((simplex[i], simplex[j]))
                    if a != b:
                        undirected.add((a, b))
        return sorted([(a, b) for a, b in undirected] + [(b, a) for a, b in undirected])
    except Exception:
        # Qhull can reject collinear or duplicate positions. This fallback is
        # only for such degenerate snapshots; ordinary snapshots use Delaunay.
        return [(i, j) for i in range(n) for j in range(n) if i != j]
