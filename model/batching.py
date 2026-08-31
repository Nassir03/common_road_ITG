"""Block-diagonal batching for the paper-only heterogeneous graph."""
from __future__ import annotations

import torch
import torch.nn.functional as F

REL_META = {
    "v2v": ("vehicle", "vehicle"),
    "v2l": ("vehicle", "lane"),
    "l2v": ("lane", "vehicle"),
    "l2l": ("lane", "lane"),
    "vtv": ("vehicle", "vehicle"),
}


def batch_graph_samples(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot batch an empty sample list")
    if len(samples) == 1:
        sample = samples[0].copy()
        sample["meta"] = [samples[0]["meta"]]
        return sample

    vehicle_offset = 0
    lane_offset = 0
    vehicle_x_parts = []
    vehicle_node_id_parts = []
    prediction_vehicle_id_parts = []
    latest_parts = []
    lane_x_parts = []
    lane_geometry_parts = []
    lane_length_parts = []
    edge_parts = {relation: [] for relation in REL_META}
    attr_parts = {
        key: []
        for key in [
            "v2v",
            "v2l",
            "l2v",
            "l2l_numeric",
            "l2l_type",
            "vtv_motion",
            "vtv_delta_t",
        ]
    }
    current_position_parts = []
    current_orientation_parts = []
    target_position_parts = []
    target_orientation_parts = []
    target_mask_parts = []
    metadata = []

    max_lane_sequence = max((sample["lane_geometry"].size(1) for sample in samples), default=1)

    for sample in samples:
        n_vehicle_nodes = sample["vehicle_x"].size(0)
        n_lanes = sample["lane_x"].size(0)

        vehicle_x_parts.append(sample["vehicle_x"])
        vehicle_node_id_parts.append(sample["vehicle_node_ids"])
        prediction_vehicle_id_parts.append(sample["prediction_vehicle_ids"])
        latest_parts.append(sample["latest_vehicle_node_index"] + vehicle_offset)
        lane_x_parts.append(sample["lane_x"])

        lane_geometry = sample["lane_geometry"]
        if lane_geometry.size(1) < max_lane_sequence:
            lane_geometry = F.pad(lane_geometry, (0, 0, 0, max_lane_sequence - lane_geometry.size(1)))
        lane_geometry_parts.append(lane_geometry)
        lane_length_parts.append(sample["lane_geometry_lengths"])

        for relation, (source_type, target_type) in REL_META.items():
            edge_index = sample["edge_index"][relation]
            if edge_index.numel() == 0:
                continue
            shifted = edge_index.clone()
            shifted[0] += vehicle_offset if source_type == "vehicle" else lane_offset
            shifted[1] += vehicle_offset if target_type == "vehicle" else lane_offset
            edge_parts[relation].append(shifted)

        for key in attr_parts:
            attr_parts[key].append(sample["edge_attr"][key])

        current_position_parts.append(sample["current_position"])
        current_orientation_parts.append(sample["current_orientation"])
        target_position_parts.append(sample["target_position"])
        target_orientation_parts.append(sample["target_orientation"])
        target_mask_parts.append(sample["target_mask"])
        metadata.append(sample["meta"])

        vehicle_offset += n_vehicle_nodes
        lane_offset += n_lanes

    def cat(parts, dim=0):
        return torch.cat(parts, dim=dim) if parts else torch.empty((0,))

    edge_index = {
        relation: torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.long)
        for relation, parts in edge_parts.items()
    }
    edge_attr = {key: cat(parts) for key, parts in attr_parts.items()}

    return {
        "vehicle_x": cat(vehicle_x_parts),
        "vehicle_node_ids": cat(vehicle_node_id_parts),
        "prediction_vehicle_ids": cat(prediction_vehicle_id_parts),
        "latest_vehicle_node_index": cat(latest_parts),
        "lane_x": cat(lane_x_parts),
        "lane_geometry": cat(lane_geometry_parts),
        "lane_geometry_lengths": cat(lane_length_parts),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "current_position": cat(current_position_parts),
        "current_orientation": cat(current_orientation_parts),
        "target_position": cat(target_position_parts),
        "target_orientation": cat(target_orientation_parts),
        "target_mask": cat(target_mask_parts),
        "prediction_times": samples[0]["prediction_times"],
        "meta": metadata,
    }
