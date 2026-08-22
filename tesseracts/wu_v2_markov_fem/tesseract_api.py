"""Tesseract API for condition-wise Wu-V2 hard-Markov JAX-FEM physics."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int64,
    ShapeDType,
)
from tesseract_core.runtime.experimental import finite_difference_vjp

from stochastic_stick_slip.jumpgrad import evaluate_jumpgrad_bank
from stochastic_stick_slip.wu_v2_markov import FD_EPSILON, NUM_STEPS


class InputSchema(BaseModel):
    q: Differentiable[Array[(None, 2), Float64]]
    conditions: Array[(None, 2), Float64]
    markov_tapes: Array[(None, None, NUM_STEPS + 1, 2), Float64]


class OutputSchema(BaseModel):
    objectives: Differentiable[Array[(None,), Float64]]
    transition_counts: Array[(None, None, 2), Int64]
    high_fraction: Array[(None, None, 2), Float64]


def apply(inputs: InputSchema) -> OutputSchema:
    result = evaluate_jumpgrad_bank(
        inputs.q, inputs.conditions, inputs.markov_tapes
    )
    return OutputSchema(
        objectives=result["objectives"],
        transition_counts=np.asarray(result["transition_counts"], dtype=np.int64),
        high_fraction=result["high_fraction"],
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    if set(vjp_inputs) != {"q"} or set(vjp_outputs) != {"objectives"}:
        raise ValueError("Wu-V2 physics differentiates objectives with respect to q")
    cotangent = np.asarray(cotangent_vector["objectives"], dtype=np.float64)
    if cotangent.shape != inputs.q.shape[:1]:
        raise ValueError("objective cotangent must have shape (condition,)")
    return finite_difference_vjp(
        apply,
        inputs,
        set(vjp_inputs),
        set(vjp_outputs),
        {"objectives": cotangent},
        algorithm="central",
        eps=FD_EPSILON,
        independent_batch_axis=0,
    )


def abstract_eval(abstract_inputs):
    batch_size = abstract_inputs.q.shape[0]
    num_realizations = abstract_inputs.markov_tapes.shape[1]
    return {
        "objectives": ShapeDType(shape=(batch_size,), dtype="float64"),
        "transition_counts": ShapeDType(
            shape=(batch_size, num_realizations, 2), dtype="int64"
        ),
        "high_fraction": ShapeDType(
            shape=(batch_size, num_realizations, 2), dtype="float64"
        ),
    }
