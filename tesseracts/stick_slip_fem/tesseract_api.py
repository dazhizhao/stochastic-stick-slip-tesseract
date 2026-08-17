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
    crn_fd_jacobian,
    evaluate_batch,
)


class InputSchema(BaseModel):
    q: Differentiable[Array[(2,), Float64]]
    seeds: Array[(8,), Int64]


class OutputSchema(BaseModel):
    seed_losses: Differentiable[Array[(8,), Float64]]
    displacement_min: Array[(8,), Float64]
    displacement_max: Array[(8,), Float64]
    velocity_min: Array[(8,), Float64]
    velocity_max: Array[(8,), Float64]
    stick_to_slip: Array[(8,), Int64]
    slip_to_stick: Array[(8,), Int64]
    representative_displacement: Array[(NUM_STEPS,), Float64]
    representative_velocity: Array[(NUM_STEPS,), Float64]
    representative_slip: Array[(NUM_STEPS,), Int64]


def apply(inputs: InputSchema) -> OutputSchema:
    result = evaluate_batch(inputs.q, np.asarray(inputs.seeds))
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


def _fd(inputs: InputSchema) -> np.ndarray:
    return crn_fd_jacobian(inputs.q, np.asarray(inputs.seeds))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    if jvp_inputs != {"q"} or jvp_outputs != {"seed_losses"}:
        raise ValueError("stick_slip_fem differentiates seed_losses with respect to q")
    return {"seed_losses": _fd(inputs) @ np.asarray(tangent_vector["q"])}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    if vjp_inputs != {"q"} or vjp_outputs != {"seed_losses"}:
        raise ValueError("stick_slip_fem differentiates seed_losses with respect to q")
    return {
        "q": np.asarray(cotangent_vector["seed_losses"]) @ _fd(inputs)
    }


def abstract_eval(abstract_inputs):
    del abstract_inputs
    return {
        "seed_losses": ShapeDType(shape=(8,), dtype="float64"),
        "displacement_min": ShapeDType(shape=(8,), dtype="float64"),
        "displacement_max": ShapeDType(shape=(8,), dtype="float64"),
        "velocity_min": ShapeDType(shape=(8,), dtype="float64"),
        "velocity_max": ShapeDType(shape=(8,), dtype="float64"),
        "stick_to_slip": ShapeDType(shape=(8,), dtype="int64"),
        "slip_to_stick": ShapeDType(shape=(8,), dtype="int64"),
        "representative_displacement": ShapeDType(
            shape=(NUM_STEPS,), dtype="float64"
        ),
        "representative_velocity": ShapeDType(
            shape=(NUM_STEPS,), dtype="float64"
        ),
        "representative_slip": ShapeDType(
            shape=(NUM_STEPS,), dtype="int64"
        ),
    }
