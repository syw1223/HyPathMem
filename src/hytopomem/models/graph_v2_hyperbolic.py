from __future__ import annotations

import torch


EPS = 1e-7


class GraphV2HyperbolicMapper(torch.nn.Module):
    def __init__(self, input_dim: int = 128, tangent_dim: int = 31, scale: float = 0.9):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, tangent_dim)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tangent = self.scale * torch.tanh(self.linear(x))
        return expmap0(tangent)


def expmap0(tangent: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(EPS)
    time_coord = torch.cosh(norm)
    spatial = torch.sinh(norm) * tangent / norm
    return torch.cat([time_coord, spatial], dim=-1)


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prod = (-lorentz_inner(x, y)).clamp_min(1.0 + EPS)
    return torch.acosh(prod).squeeze(-1)


def lorentz_radius(x: torch.Tensor) -> torch.Tensor:
    return torch.acosh(x[..., 0].clamp_min(1.0 + EPS))
