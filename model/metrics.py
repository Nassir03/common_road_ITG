"""Average displacement error (ADE) and final displacement error (FDE)."""
from __future__ import annotations
import torch


def ade_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean displacement over all future steps and valid targets."""
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    valid = mask[:, None].expand_as(dist)
    if not bool(valid.any()):
        return dist.sum() * 0.0
    return dist[valid].mean()


def ade_fde(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    """Per-target averaged ADE/FDE in metres."""
    if not bool(mask.any()):
        return {"ade": float("nan"), "fde": float("nan"), "count": 0}
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    valid_dist = dist[mask]
    ade_per_target = valid_dist.mean(dim=1)
    fde_per_target = valid_dist[:, -1]
    return {
        "ade": float(ade_per_target.mean().detach().cpu()),
        "fde": float(fde_per_target.mean().detach().cpu()),
        "count": int(mask.sum().detach().cpu()),
    }

# Backward-compatible alias used by earlier repository versions.
trajectory_ade_loss = ade_loss
