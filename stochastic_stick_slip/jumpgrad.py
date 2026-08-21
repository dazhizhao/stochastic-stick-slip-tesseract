"""Condition-aware JumpGrad controller and frozen Wu-V2 hard physics."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from stochastic_stick_slip.model import STEPS_PER_PERIOD
from stochastic_stick_slip.wu_v2 import (
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
)
from stochastic_stick_slip.wu_v2_markov import (
    FD_EPSILON,
    NUM_STEPS,
    generate_hard_preload_history,
    markov_uniform_bank,
    mechanics_forward,
    steady_state_amplitude,
)


OMEGA_R_RATIO = 1.19
OMEGA_R = OMEGA_R_RATIO * SYSTEM.omega_1
WU_AMPLITUDE = 0.01
WU_PHASE = 4.516039439535327
FIXED_Q = np.asarray(
    [-10.665739565561044, -6.033414703985564], dtype=np.float64
)

TRAINING_CONDITIONS = np.asarray(
    [(amplitude, frequency) for amplitude in (0.8, 1.0, 1.2, 1.4)
     for frequency in (1.00, 1.04)],
    dtype=np.float64,
)
HELD_OUT_CONDITIONS = np.asarray(
    [(amplitude, frequency) for amplitude in (0.9, 1.1, 1.3, 1.5)
     for frequency in (0.98, 1.06)],
    dtype=np.float64,
)

AUDIT_STREAM = 9
TRAINING_STREAM = 10
MONITOR_STREAM = 11
HELD_OUT_STREAM = 12

CONTROLLER_PARAMETER_LAYOUT = (
    ("0.weight", (16, 2)),
    ("0.bias", (16,)),
    ("2.weight", (16, 16)),
    ("2.bias", (16,)),
    ("4.weight", (2, 16)),
    ("4.bias", (2,)),
)
NUM_CONTROLLER_PARAMETERS = sum(
    torch.Size(shape).numel() for _, shape in CONTROLLER_PARAMETER_LAYOUT
)
assert NUM_CONTROLLER_PARAMETERS == 354
assert NUM_STEPS == DIAGNOSTIC_NUM_PERIODS * STEPS_PER_PERIOD == 2400


def build_jumpgrad_controller() -> nn.Sequential:
    """Build the registered neutral-initialized 2-16-16-2 MLP."""
    torch.manual_seed(0)
    controller = nn.Sequential(
        nn.Linear(2, 16, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(16, 16, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(16, 2, dtype=torch.float64),
    )
    nn.init.zeros_(controller[-1].weight)
    nn.init.zeros_(controller[-1].bias)
    return controller


def flatten_jumpgrad_parameters(controller: nn.Module) -> torch.Tensor:
    """Flatten controller parameters in the declared functional order."""
    parameters = dict(controller.named_parameters())
    expected = tuple(name for name, _ in CONTROLLER_PARAMETER_LAYOUT)
    if tuple(parameters) != expected:
        raise ValueError("JumpGrad controller parameter layout changed")
    return torch.cat([parameters[name].reshape(-1) for name in expected])


def jumpgrad_parameter_dict(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map a differentiable flat theta vector to the MLP parameter tree."""
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


def functional_jumpgrad_controller(
    controller: nn.Module,
    theta: torch.Tensor,
    descriptors: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the condition-aware MLP from a flat parameter vector."""
    return torch.func.functional_call(
        controller,
        jumpgrad_parameter_dict(theta),
        (descriptors,),
        strict=True,
    )


def condition_descriptors(conditions: np.ndarray) -> np.ndarray:
    """Return the registered normalized amplitude/frequency descriptors."""
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("conditions must have shape (batch, 2)")
    return np.column_stack(
        (values[:, 0] - 1.0, (values[:, 1] - 1.0) / 0.1)
    )


def jumpgrad_uniform_bank(
    num_conditions: int,
    num_realizations: int,
    stream_id: int,
    iteration: int = 0,
) -> np.ndarray:
    """Return a fixed Wu-V2 tape bank for the first registered condition rows."""
    if not 1 <= num_conditions <= len(TRAINING_CONDITIONS):
        raise ValueError("num_conditions must be between 1 and 8")
    return markov_uniform_bank(
        num_realizations=num_realizations,
        stream_id=stream_id,
        iteration=iteration,
    )[:num_conditions]


def _condition_inputs(conditions: jax.Array):
    conditions = jnp.asarray(conditions, dtype=jnp.float64)
    omega = OMEGA_R * conditions[:, 1]
    time_step = 2.0 * jnp.pi / (omega * STEPS_PER_PERIOD)
    indices = jnp.arange(1, NUM_STEPS + 1, dtype=jnp.float64)
    times = time_step[:, None] * indices[None, :]
    forcing = (
        FORCING_AMPLITUDE
        * conditions[:, 0, None]
        * jnp.sin(omega[:, None] * times)
    )
    return omega, time_step, times, forcing


def _generate_histories_impl(q, conditions, tapes):
    omega, time_step, times, _ = _condition_inputs(conditions)
    return jax.vmap(
        generate_hard_preload_history,
        in_axes=(0, 0, 0, 0, 0),
    )(q, times, tapes, omega, time_step)


GENERATE_JUMPGRAD_HISTORIES = jax.jit(_generate_histories_impl)


def _mechanics_by_condition(forcing, preload, time_step):
    def one_condition(condition_forcing, condition_preload, condition_step):
        forcing_bank = jnp.broadcast_to(
            condition_forcing,
            (condition_preload.shape[0], condition_forcing.shape[0]),
        )
        return mechanics_forward(
            forcing_bank, condition_preload, condition_step
        )

    return jax.vmap(one_condition)(forcing, preload, time_step)


MECHANICS_BY_CONDITION = jax.jit(_mechanics_by_condition)


def _evaluate_jumpgrad_impl(q, conditions, tapes):
    omega, time_step, times, forcing = _condition_inputs(conditions)
    _, preload, transition_counts, high_fraction = jax.vmap(
        generate_hard_preload_history,
        in_axes=(0, 0, 0, 0, 0),
    )(q, times, tapes, omega, time_step)
    displacement = _mechanics_by_condition(
        forcing, preload, time_step
    )[0]
    trajectory_objectives = steady_state_amplitude(displacement)
    condition_objectives = jnp.mean(trajectory_objectives, axis=1)
    return (
        condition_objectives,
        trajectory_objectives,
        transition_counts,
        high_fraction,
    )


EVALUATE_JUMPGRAD_BANK = jax.jit(_evaluate_jumpgrad_impl)


def _validate_bank_inputs(q, conditions, tapes):
    q = np.asarray(q, dtype=np.float64)
    conditions = np.asarray(conditions, dtype=np.float64)
    tapes = np.asarray(tapes, dtype=np.float64)
    batch_size = len(conditions)
    if q.shape != (batch_size, 2):
        raise ValueError("q must have shape (condition, 2)")
    if conditions.ndim != 2 or conditions.shape[1] != 2:
        raise ValueError("conditions must have shape (condition, 2)")
    if tapes.ndim != 4 or tapes.shape != (
        batch_size,
        tapes.shape[1],
        NUM_STEPS + 1,
        2,
    ):
        raise ValueError(
            "tapes must have shape (condition, realization, 2401, 2)"
        )
    return q, conditions, tapes


def evaluate_jumpgrad_bank(q, conditions, tapes) -> dict[str, np.ndarray]:
    """Evaluate condition-wise Wu-V2 objectives and Markov diagnostics."""
    q, conditions, tapes = _validate_bank_inputs(q, conditions, tapes)
    outputs = EVALUATE_JUMPGRAD_BANK(
        jnp.asarray(q), jnp.asarray(conditions), jnp.asarray(tapes)
    )
    names = (
        "objectives",
        "trajectory_objectives",
        "transition_counts",
        "high_fraction",
    )
    result = {name: np.asarray(value) for name, value in zip(names, outputs)}
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise FloatingPointError("JumpGrad bank output is non-finite")
    return result


def generate_jumpgrad_histories(q, conditions, tapes) -> dict[str, np.ndarray]:
    """Generate hard histories without running mechanics."""
    q, conditions, tapes = _validate_bank_inputs(q, conditions, tapes)
    outputs = GENERATE_JUMPGRAD_HISTORIES(
        jnp.asarray(q), jnp.asarray(conditions), jnp.asarray(tapes)
    )
    names = ("modes", "preload", "transition_counts", "high_fraction")
    return {name: np.asarray(value) for name, value in zip(names, outputs)}


def crn_fd_condition_gradient(q, conditions, tapes) -> dict[str, np.ndarray]:
    """Return the two-coordinate, same-tape centered FD for every row."""
    q, conditions, tapes = _validate_bank_inputs(q, conditions, tapes)
    derivatives = np.empty_like(q)
    plus_objectives = np.empty_like(q)
    minus_objectives = np.empty_like(q)
    for coefficient in range(2):
        plus = q.copy()
        minus = q.copy()
        plus[:, coefficient] += FD_EPSILON
        minus[:, coefficient] -= FD_EPSILON
        plus_value = evaluate_jumpgrad_bank(plus, conditions, tapes)[
            "objectives"
        ]
        minus_value = evaluate_jumpgrad_bank(minus, conditions, tapes)[
            "objectives"
        ]
        plus_objectives[:, coefficient] = plus_value
        minus_objectives[:, coefficient] = minus_value
        derivatives[:, coefficient] = (
            plus_value - minus_value
        ) / (2.0 * FD_EPSILON)
    return {
        "gradient": derivatives,
        "plus_objectives": plus_objectives,
        "minus_objectives": minus_objectives,
    }


def _weighted_objective(q, conditions, tapes, cotangent):
    return jnp.vdot(
        _evaluate_jumpgrad_impl(q, conditions, tapes)[0], cotangent
    )


DIRECT_AD_VALUE_AND_GRAD = jax.jit(
    jax.value_and_grad(_weighted_objective, argnums=0)
)


def direct_ad_physics_gradient(q, conditions, tapes, cotangent):
    """Return raw AD through the true hard mode history."""
    q, conditions, tapes = _validate_bank_inputs(q, conditions, tapes)
    cotangent = np.asarray(cotangent, dtype=np.float64)
    if cotangent.shape != (len(conditions),):
        raise ValueError("cotangent must have shape (condition,)")
    value, gradient = DIRECT_AD_VALUE_AND_GRAD(
        jnp.asarray(q),
        jnp.asarray(conditions),
        jnp.asarray(tapes),
        jnp.asarray(cotangent),
    )
    return float(value), np.asarray(gradient)


def deterministic_condition_objectives(
    conditions: np.ndarray,
    method: str,
) -> np.ndarray:
    """Evaluate constant passive or frozen continuous Wu 2omega control."""
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("conditions must have shape (condition, 2)")
    omega, time_step, times, forcing = _condition_inputs(jnp.asarray(values))
    if method == "passive":
        scalar = jnp.full(times.shape, REFERENCE_PRELOAD, dtype=jnp.float64)
    elif method == "wu_continuous_2omega":
        scalar = REFERENCE_PRELOAD + WU_AMPLITUDE * jnp.sin(
            2.0 * omega[:, None] * times + WU_PHASE
        )
    else:
        raise ValueError("method must be passive or wu_continuous_2omega")
    preload = jnp.repeat(scalar[:, None, :, None], 2, axis=-1)
    displacement = MECHANICS_BY_CONDITION(forcing, preload, time_step)[0]
    objectives = np.asarray(steady_state_amplitude(displacement)[:, 0])
    if not np.all(np.isfinite(objectives)):
        raise FloatingPointError("deterministic condition response is non-finite")
    return objectives


def q_polar_rows(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude and wrapped coefficient phase for q rows."""
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("q must have shape (batch, 2)")
    magnitude = np.linalg.norm(values, axis=1)
    phase = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * np.pi)
    phase = np.where(magnitude == 0.0, 0.0, phase)
    return magnitude, phase
