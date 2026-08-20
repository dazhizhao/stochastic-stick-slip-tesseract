"""Tesseract API for the hard Markov-jump Jenkins FEM response."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int64,
    ShapeDType,
)

from stochastic_stick_slip.engineering_markov import (
    FD_EPSILON,
    evaluate_markov_bank,
)
from stochastic_stick_slip.engineering_showcase import forcing_batch
from stochastic_stick_slip.model import NUM_STEPS


class InputSchema(BaseModel):
    coeffs: Differentiable[Array[(8, 5), Float64]]
    forcing_seeds: Array[(8,), Int64]
    markov_uniforms: Array[(8, 4, NUM_STEPS + 1, 2), Float64]


class OutputSchema(BaseModel):
    seed_losses: Differentiable[Array[(8,), Float64]]
    transition_counts: Array[(8, 4, 2), Int64]
    high_mode_fraction: Array[(8, 4, 2), Float64]


def _evaluate(coefficients, forcing_seeds, markov_uniforms):
    forcing = forcing_batch(np.asarray(forcing_seeds, dtype=np.int64))
    return evaluate_markov_bank(coefficients, forcing, markov_uniforms)


def apply(inputs: InputSchema) -> OutputSchema:
    result = _evaluate(
        inputs.coeffs,
        inputs.forcing_seeds,
        inputs.markov_uniforms,
    )
    return OutputSchema(
        seed_losses=np.mean(np.asarray(result.losses), axis=1),
        transition_counts=np.asarray(result.transition_counts, dtype=np.int64),
        high_mode_fraction=np.asarray(result.high_mode_fraction),
    )


def _fd_coefficients(inputs: InputSchema) -> np.ndarray:
    coefficients = np.asarray(inputs.coeffs, dtype=np.float64)
    forcing_seeds = np.asarray(inputs.forcing_seeds, dtype=np.int64)
    uniforms = np.asarray(inputs.markov_uniforms, dtype=np.float64)
    columns = []
    for column in range(coefficients.shape[1]):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[:, column] += FD_EPSILON
        minus[:, column] -= FD_EPSILON
        plus_losses = np.mean(
            np.asarray(_evaluate(plus, forcing_seeds, uniforms).losses), axis=1
        )
        minus_losses = np.mean(
            np.asarray(_evaluate(minus, forcing_seeds, uniforms).losses), axis=1
        )
        columns.append((plus_losses - minus_losses) / (2.0 * FD_EPSILON))
    return np.stack(columns, axis=1)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    if set(vjp_inputs) != {"coeffs"} or set(vjp_outputs) != {"seed_losses"}:
        raise ValueError(
            "markov_jump_fem differentiates seed_losses with respect to coeffs"
        )
    cotangent = np.asarray(cotangent_vector["seed_losses"], dtype=np.float64)
    return {"coeffs": cotangent[:, None] * _fd_coefficients(inputs)}


def abstract_eval(abstract_inputs):
    del abstract_inputs
    return {
        "seed_losses": ShapeDType(shape=(8,), dtype="float64"),
        "transition_counts": ShapeDType(shape=(8, 4, 2), dtype="int64"),
        "high_mode_fraction": ShapeDType(shape=(8, 4, 2), dtype="float64"),
    }
