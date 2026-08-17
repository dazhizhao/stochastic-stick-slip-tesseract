"""Tesseract API for the fixed-seed stochastic mean objective."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    seed_losses: Differentiable[Array[(8,), Float64]]


class OutputSchema(BaseModel):
    objective: Differentiable[Float64]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(objective=np.mean(inputs.seed_losses))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    del inputs
    if jvp_inputs != {"seed_losses"} or jvp_outputs != {"objective"}:
        raise ValueError(
            "stochastic_objective differentiates objective with respect to seed_losses"
        )
    return {"objective": np.mean(tangent_vector["seed_losses"])}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    del inputs
    if vjp_inputs != {"seed_losses"} or vjp_outputs != {"objective"}:
        raise ValueError(
            "stochastic_objective differentiates objective with respect to seed_losses"
        )
    cotangent = np.asarray(cotangent_vector["objective"])
    return {"seed_losses": np.full(8, cotangent / 8.0)}


def abstract_eval(abstract_inputs):
    del abstract_inputs
    return {"objective": ShapeDType(shape=(), dtype="float64")}
