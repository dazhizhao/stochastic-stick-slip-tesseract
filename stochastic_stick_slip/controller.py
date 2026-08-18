"""Small PyTorch controller for the five-dimensional Fourier interface."""

import torch
from torch import nn


CONTROLLER_PARAMETER_LAYOUT = (
    ("0.weight", (16, 6)),
    ("0.bias", (16,)),
    ("2.weight", (16, 16)),
    ("2.bias", (16,)),
    ("4.weight", (5, 16)),
    ("4.bias", (5,)),
)
NUM_CONTROLLER_PARAMETERS = sum(
    torch.Size(shape).numel() for _, shape in CONTROLLER_PARAMETER_LAYOUT
)
assert NUM_CONTROLLER_PARAMETERS == 469


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


def flatten_controller_parameters(controller: nn.Module) -> torch.Tensor:
    """Flatten the fixed controller parameters in their declared order."""
    parameters = dict(controller.named_parameters())
    expected_names = tuple(name for name, _ in CONTROLLER_PARAMETER_LAYOUT)
    if tuple(parameters) != expected_names:
        raise ValueError("controller parameters do not match the fixed layout")
    return torch.cat([parameters[name].reshape(-1) for name in expected_names])


def controller_parameter_dict(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map the flat 469-vector to the fixed functional parameter dictionary."""
    if theta.ndim != 1 or theta.numel() != NUM_CONTROLLER_PARAMETERS:
        raise ValueError(
            f"theta must have shape ({NUM_CONTROLLER_PARAMETERS},)"
        )
    parameters = {}
    offset = 0
    for name, shape in CONTROLLER_PARAMETER_LAYOUT:
        size = torch.Size(shape).numel()
        parameters[name] = theta[offset : offset + size].reshape(shape)
        offset += size
    return parameters


def functional_controller(
    controller: nn.Module,
    theta: torch.Tensor,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the fixed MLP from a differentiable flat parameter vector."""
    return torch.func.functional_call(
        controller,
        controller_parameter_dict(theta),
        (descriptors,),
        strict=True,
    )


def parameter_gradient_norm(parameters) -> float:
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm = squared_norm + torch.sum(parameter.grad**2)
    return float(torch.sqrt(squared_norm))
