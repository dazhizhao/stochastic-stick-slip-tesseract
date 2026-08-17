"""Tesseract API for the stochastic hard stick-slip FEM response."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float64,
    Int64,
    ShapeDType,
)

from stochastic_stick_slip.model import (
    NUM_STEPS,
    crn_fd_coefficient_jacobian,
    crn_fd_controlled_q_jacobian,
    evaluate_controlled_batch,
)


class InputSchema(BaseModel):
    q: Differentiable[Array[(2,), Float64]]
    coeffs: Differentiable[Array[(8, 5), Float64]]
    seeds: Array[(8,), Int64]


class OutputSchema(BaseModel):
    seed_losses: Differentiable[Array[(8,), Float64]]
    displacement_min: Array[(8,), Float64]
    displacement_max: Array[(8,), Float64]
    velocity_min: Array[(8,), Float64]
    velocity_max: Array[(8,), Float64]
    stick_to_slip: Array[(8, 2), Int64]
    slip_to_stick: Array[(8, 2), Int64]
    representative_displacement: Array[(NUM_STEPS,), Float64]
    representative_velocity: Array[(NUM_STEPS,), Float64]
    representative_slip: Array[(NUM_STEPS, 2), Int64]


def apply(inputs: InputSchema) -> OutputSchema:
    result = evaluate_controlled_batch(
        inputs.q, inputs.coeffs, np.asarray(inputs.seeds)
    )
    displacement = np.asarray(result.displacement)
    velocity = np.asarray(result.velocity)
    slip = np.asarray(result.slip, dtype=np.int64)
    return OutputSchema(
        seed_losses=np.asarray(result.losses),
        displacement_min=displacement.min(axis=1),
        displacement_max=displacement.max(axis=1),
        velocity_min=velocity.min(axis=1),
        velocity_max=velocity.max(axis=1),
        stick_to_slip=np.asarray(result.stick_to_slip, dtype=np.int64),
        slip_to_stick=np.asarray(result.slip_to_stick, dtype=np.int64),
        representative_displacement=displacement[0],
        representative_velocity=velocity[0],
        representative_slip=slip[0],
    )


def _fd_q(inputs: InputSchema) -> np.ndarray:
    return crn_fd_controlled_q_jacobian(
        inputs.q, inputs.coeffs, np.asarray(inputs.seeds)
    )


def _fd_coeffs(inputs: InputSchema) -> np.ndarray:
    return crn_fd_coefficient_jacobian(
        inputs.q, inputs.coeffs, np.asarray(inputs.seeds)
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    requested_inputs = set(jvp_inputs)
    if not requested_inputs or not requested_inputs <= {"q", "coeffs"}:
        raise ValueError("stick_slip_fem differentiates with respect to q or coeffs")
    if set(jvp_outputs) != {"seed_losses"}:
        raise ValueError("stick_slip_fem only differentiates seed_losses")
    result = np.zeros(8, dtype=np.float64)
    if "q" in requested_inputs:
        result += _fd_q(inputs) @ np.asarray(tangent_vector["q"])
    if "coeffs" in requested_inputs:
        result += np.sum(
            _fd_coeffs(inputs) * np.asarray(tangent_vector["coeffs"]), axis=1
        )
    return {"seed_losses": result}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    requested_inputs = set(vjp_inputs)
    if not requested_inputs or not requested_inputs <= {"q", "coeffs"}:
        raise ValueError("stick_slip_fem differentiates with respect to q or coeffs")
    if set(vjp_outputs) != {"seed_losses"}:
        raise ValueError("stick_slip_fem only differentiates seed_losses")
    cotangent = np.asarray(cotangent_vector["seed_losses"])
    result = {}
    if "q" in requested_inputs:
        result["q"] = cotangent @ _fd_q(inputs)
    if "coeffs" in requested_inputs:
        result["coeffs"] = cotangent[:, None] * _fd_coeffs(inputs)
    return result


def abstract_eval(abstract_inputs):
    del abstract_inputs
    return {
        "seed_losses": ShapeDType(shape=(8,), dtype="float64"),
        "displacement_min": ShapeDType(shape=(8,), dtype="float64"),
        "displacement_max": ShapeDType(shape=(8,), dtype="float64"),
        "velocity_min": ShapeDType(shape=(8,), dtype="float64"),
        "velocity_max": ShapeDType(shape=(8,), dtype="float64"),
        "stick_to_slip": ShapeDType(shape=(8, 2), dtype="int64"),
        "slip_to_stick": ShapeDType(shape=(8, 2), dtype="int64"),
        "representative_displacement": ShapeDType(
            shape=(NUM_STEPS,), dtype="float64"
        ),
        "representative_velocity": ShapeDType(
            shape=(NUM_STEPS,), dtype="float64"
        ),
        "representative_slip": ShapeDType(
            shape=(NUM_STEPS, 2), dtype="int64"
        ),
    }
