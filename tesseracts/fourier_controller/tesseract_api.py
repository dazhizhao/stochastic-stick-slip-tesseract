"""Tesseract API for the fixed PyTorch Fourier controller."""

import numpy as np
from pydantic import BaseModel
import torch
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

from stochastic_stick_slip.controller import (
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    functional_controller,
)


_CONTROLLER = build_controller()


class InputSchema(BaseModel):
    theta: Differentiable[Array[(NUM_CONTROLLER_PARAMETERS,), Float64]]
    descriptors: Array[(8, 6), Float64]


class OutputSchema(BaseModel):
    coeffs: Differentiable[Array[(8, 5), Float64]]


def _forward(theta, descriptors):
    return functional_controller(
        _CONTROLLER,
        theta,
        descriptors,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    theta = torch.tensor(np.asarray(inputs.theta), dtype=torch.float64)
    descriptors = torch.tensor(
        np.asarray(inputs.descriptors), dtype=torch.float64
    )
    with torch.no_grad():
        coefficients = _forward(theta, descriptors)
    return OutputSchema(coeffs=coefficients.detach().cpu().numpy())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    if set(vjp_inputs) != {"theta"} or set(vjp_outputs) != {"coeffs"}:
        raise ValueError(
            "fourier_controller differentiates coeffs with respect to theta"
        )
    with torch.enable_grad():
        theta = torch.tensor(
            np.asarray(inputs.theta), dtype=torch.float64, requires_grad=True
        )
        descriptors = torch.tensor(
            np.asarray(inputs.descriptors), dtype=torch.float64
        )
        cotangent = torch.tensor(
            np.asarray(cotangent_vector["coeffs"]), dtype=torch.float64
        )
        coefficients = _forward(theta, descriptors)
        gradient = torch.autograd.grad(
            coefficients,
            theta,
            grad_outputs=cotangent,
        )[0]
    return {"theta": gradient.detach().cpu().numpy()}


def abstract_eval(abstract_inputs):
    del abstract_inputs
    return {"coeffs": ShapeDType(shape=(8, 5), dtype="float64")}
