"""Trajectory-prediction loss and evaluation metrics from the paper."""
from __future__ import annotations

import torch


def _valid_distances(
    prediction_position: torch.Tensor,
    target_position: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction_position.shape != target_position.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction_position.shape} vs {target_position.shape}")
    distances = torch.linalg.vector_norm(prediction_position - target_position, dim=-1)
    if target_mask is None:
        target_mask = torch.ones(distances.size(0), dtype=torch.bool, device=distances.device)
    target_mask = target_mask.to(distances.device, dtype=torch.bool)
    if not bool(target_mask.any()):
        raise ValueError("No valid target vehicles in this sample")
    return distances[target_mask], target_mask


def ade_fde(
    prediction_position: torch.Tensor,
    target_position: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ADE and FDE in metres, averaged over valid vehicle trajectories."""
    distances, _ = _valid_distances(prediction_position, target_position, target_mask)
    ade = distances.mean()
    fde = distances[:, -1].mean()
    return ade, fde


def trajectory_ade_loss(
    prediction_position: torch.Tensor,
    target_position: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paper training objective: Average Displacement Error (ADE)."""
    ade, _ = ade_fde(prediction_position, target_position, target_mask)
    return ade


def trajectory_error_sums(
    prediction_position: torch.Tensor,
    target_position: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return sums/counts for dataset-level ADE/FDE aggregation."""
    distances, _ = _valid_distances(prediction_position, target_position, target_mask)
    return distances.sum(), distances[:, -1].sum(), int(distances.numel()), int(distances.size(0))
