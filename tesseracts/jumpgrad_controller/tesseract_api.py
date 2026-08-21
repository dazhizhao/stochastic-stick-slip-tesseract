"""Tesseract API for the condition-aware PyTorch JumpGrad controller."""

import numpy as np
from pydantic import BaseModel
import torch
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

from stochastic_stick_slip.jumpgrad import (
    NUM_CONTROLLER_PARAMETERS,
    build_jumpgrad_controller,
    functional_jumpgrad_controller,
)


_CONTROLLER = build_jumpgrad_controller()


class InputSchema(BaseModel):
    theta: Differentiable[
        Array[(NUM_CONTROLLER_PARAMETERS,), Float64]
    ]
    descriptors: Array[(None, 2), Float64]


class OutputSchema(BaseModel):
    q: Differentiable[Array[(None, 2), Float64]]


def _forward(theta, descriptors):
    return functional_jumpgrad_controller(_CONTROLLER, theta, descriptors)


def apply(inputs: InputSchema) -> OutputSchema:
    theta = torch.tensor(np.asarray(inputs.theta), dtype=torch.float64)
    descriptors = torch.tensor(
        np.asarray(inputs.descriptors), dtype=torch.float64
    )
    with torch.no_grad():
        q = _forward(theta, descriptors)
    return OutputSchema(q=q.detach().cpu().numpy())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    if set(vjp_inputs) != {"theta"} or set(vjp_outputs) != {"q"}:
        raise ValueError("JumpGrad controller differentiates q with respect to theta")
    with torch.enable_grad():
        theta = torch.tensor(
            np.asarray(inputs.theta), dtype=torch.float64, requires_grad=True
        )
        descriptors = torch.tensor(
            np.asarray(inputs.descriptors), dtype=torch.float64
        )
        cotangent = torch.tensor(
            np.asarray(cotangent_vector["q"]), dtype=torch.float64
        )
        q = _forward(theta, descriptors)
        gradient = torch.autograd.grad(q, theta, grad_outputs=cotangent)[0]
    return {"theta": gradient.detach().cpu().numpy()}


def abstract_eval(abstract_inputs):
    batch_size = abstract_inputs.descriptors.shape[0]
    return {"q": ShapeDType(shape=(batch_size, 2), dtype="float64")}
