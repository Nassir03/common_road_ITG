"""Optional CommonRoad-Geometric adapter.

The edge drawer uses actual cr-geo vehicle IDs, positions, orientations,
velocities, accelerations, and the simulation's current time step. It exposes
aligned ITG features and an optional postprocessor attaches them to ``data.v2v``.
"""
from __future__ import annotations

try:
    import torch
    from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer
except ImportError:
    torch = None
    BaseEdgeDrawer = object

from .config import COMMUNICATION_RADIUS, ROI_RADIUS, MAX_HOPS
from .itg import VehicleState, build_itg_snapshot


class InfluenceTransferEdgeDrawer(BaseEdgeDrawer):
    def __init__(self, communication_radius=COMMUNICATION_RADIUS, roi_radius=ROI_RADIUS, max_hops=MAX_HOPS):
        if torch is None:
            raise ImportError("Install commonroad-geometric to use InfluenceTransferEdgeDrawer")
        super().__init__(dist_threshold=None)
        self.communication_radius = communication_radius
        self.roi_radius = roi_radius
        self.max_hops = max_hops
        self.last_snapshot = None
        self.last_edge_features = None

    def __call__(self, options):
        # Override BaseEdgeDrawer.__call__ because its n_vehicles==2 special case
        # bypasses custom topology. ITG must always be constructed dynamically.
        pos = options.pos.float()
        dist_matrix = torch.cdist(pos, pos)
        edge_index = self._draw(options)
        return edge_index, dist_matrix

    def _draw(self, options):
        simulation = options.simulation
        vehicle_ids = [int(x) for x in options.v_data["id"].view(-1).detach().cpu().tolist()]
        obstacle_by_id = {int(o.obstacle_id): o for o in simulation.current_obstacles}
        states = []
        for row, vehicle_id in enumerate(vehicle_ids):
            obstacle = obstacle_by_id[vehicle_id]
            state = simulation.get_current_obstacle_state(obstacle)
            p = options.pos[row].detach().cpu()
            states.append(VehicleState(
                vehicle_id=vehicle_id,
                x=float(p[0]), y=float(p[1]),
                heading=float(getattr(state, "orientation", 0.0)),
                speed=float(getattr(state, "velocity", 0.0)),
                acceleration=float(getattr(state, "acceleration", 0.0)),
            ))
        snapshot = build_itg_snapshot(
            states, int(simulation.current_time_step),
            self.communication_radius, self.roi_radius, self.max_hops,
        )
        self.last_snapshot = snapshot
        id_to_row = {vid: i for i, vid in enumerate(vehicle_ids)}
        pairs, feats = [], []
        for e in snapshot.edges:
            if e.source not in id_to_row or e.target not in id_to_row:
                continue
            pairs.append([id_to_row[e.source], id_to_row[e.target]])
            feats.append([
                e.hop / max(1, self.max_hops),
                e.branch / max(1, len(vehicle_ids)),
                float(e.direct),
                e.dx_local / max(1.0, self.roi_radius),
                e.dy_local / max(1.0, self.roi_radius),
                e.distance / max(1.0, self.roi_radius),
                e.rel_vx_local / 30.0,
                e.rel_vy_local / 30.0,
            ])
        device = options.pos.device
        self.last_edge_features = torch.tensor(feats, dtype=torch.float32, device=device) if feats else torch.empty((0, 8), dtype=torch.float32, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long, device=device)


class ITGEdgeFeaturePostprocessor:
    """Attach the edge-drawer's hop-aware ITG metadata to cr-geo CommonRoadData."""
    def __init__(self, edge_drawer: InfluenceTransferEdgeDrawer):
        self.edge_drawer = edge_drawer

    def __call__(self, samples, simulation=None, ego_vehicle=None):
        features = self.edge_drawer.last_edge_features
        if features is None:
            return samples
        for data in samples:
            if data.v2v is not None and data.v2v.edge_index.size(1) == features.size(0):
                data.v2v.itg_edge_attr = features.to(data.v2v.edge_index.device)
                data.v2v.itg_hop = data.v2v.itg_edge_attr[:, 0:1]
                data.v2v.itg_branch = data.v2v.itg_edge_attr[:, 1:2]
                data.v2v.itg_direct = data.v2v.itg_edge_attr[:, 2:3]
        return samples
