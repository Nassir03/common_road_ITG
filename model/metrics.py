"""Trajectory-prediction metrics."""
from __future__ import annotations
import torch


def ade_fde(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(prediction - target, dim=-1)
    return distances.mean(), distances[..., -1].mean()
