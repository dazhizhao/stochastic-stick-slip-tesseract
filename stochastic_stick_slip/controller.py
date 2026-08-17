"""Small PyTorch controller for the five-dimensional Fourier interface."""

import torch
from torch import nn


def build_controller() -> nn.Sequential:
    """Build the reproducible 6-16-16-5 controller at the fixed baseline."""
    torch.manual_seed(0)
    controller = nn.Sequential(
        nn.Linear(6, 16, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(16, 16, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(16, 5, dtype=torch.float64),
    )
    final_layer = controller[-1]
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    return controller


def parameter_gradient_norm(parameters) -> float:
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm = squared_norm + torch.sum(parameter.grad**2)
    return float(torch.sqrt(squared_norm))
