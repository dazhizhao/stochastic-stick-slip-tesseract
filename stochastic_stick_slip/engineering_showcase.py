"""Locked low-damping, resonant binding for the engineering showcase."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    FD_RELATIVE_EPSILON,
    FORCING_AMPLITUDE,
    NUM_FOURIER_COEFFICIENTS,
    NUM_STEPS,
    BatchResult,
    TrajectoryResult,
    build_batch_simulator,
    build_trajectory_simulator,
    forcing_parameters,
    preload_history_with_basis,
)
from stochastic_stick_slip.showcase import (
    CONTACT_COLUMNS,
    NUM_ELEMENTS_X,
    NUM_ELEMENTS_Y,
    SYSTEM,
    full_nodal_field,
)


FIRST_FREQUENCY_RATIO = 1.0
SECOND_FREQUENCY_RATIO = 1.35

FOURIER_BASIS = jnp.stack(
    (
        jnp.ones(NUM_STEPS, dtype=jnp.float64),
        jnp.cos(FIRST_FREQUENCY_RATIO * SYSTEM.omega_1 * SYSTEM.times),
        jnp.sin(FIRST_FREQUENCY_RATIO * SYSTEM.omega_1 * SYSTEM.times),
        jnp.cos(SECOND_FREQUENCY_RATIO * SYSTEM.omega_1 * SYSTEM.times),
        jnp.sin(SECOND_FREQUENCY_RATIO * SYSTEM.omega_1 * SYSTEM.times),
    ),
    axis=1,
)
_SIMULATE_BATCH = build_batch_simulator(SYSTEM, FOURIER_BASIS)
_SIMULATE_TRAJECTORY = build_trajectory_simulator(SYSTEM, FOURIER_BASIS)


def forcing_history(seed: int) -> np.ndarray:
    """Return the frozen 1.00/1.35-omega forcing for one seed."""
    amplitudes, phases = forcing_parameters(seed)
    times = np.asarray(SYSTEM.times)
    return FORCING_AMPLITUDE * (
        amplitudes[0]
        * np.sin(FIRST_FREQUENCY_RATIO * SYSTEM.omega_1 * times + phases[0])
        + 0.6
        * amplitudes[1]
        * np.sin(SECOND_FREQUENCY_RATIO * SYSTEM.omega_1 * times + phases[1])
    )


def forcing_batch(seeds: np.ndarray) -> jax.Array:
    return jnp.asarray(
        np.stack([forcing_history(int(seed)) for seed in seeds]),
        dtype=jnp.float64,
    )


def preload_history(base_preload, coefficients):
    return preload_history_with_basis(base_preload, coefficients, FOURIER_BASIS)


def evaluate_controlled_batch(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> BatchResult:
    displacement, velocity, slip, stick_to_slip, slip_to_stick = _SIMULATE_BATCH(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(coefficients, dtype=jnp.float64),
        forcing_batch(seeds),
    )
    return BatchResult(
        losses=jnp.mean(displacement**2, axis=1),
        displacement=displacement,
        velocity=velocity,
        slip=slip,
        stick_to_slip=jnp.sum(stick_to_slip, axis=1),
        slip_to_stick=jnp.sum(slip_to_stick, axis=1),
    )


def crn_fd_controlled_q_jacobian(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
    epsilon_multiplier: float = 1.0,
) -> np.ndarray:
    q_array = np.asarray(q, dtype=np.float64)
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    epsilons = FD_RELATIVE_EPSILON * q_array * epsilon_multiplier
    columns = []
    for index, epsilon in enumerate(epsilons):
        plus = q_array.copy()
        minus = q_array.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_losses = np.asarray(
            evaluate_controlled_batch(plus, coefficients_array, seeds).losses
        )
        minus_losses = np.asarray(
            evaluate_controlled_batch(minus, coefficients_array, seeds).losses
        )
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def crn_fd_coefficient_jacobian(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
    epsilon: float = COEFFICIENT_FD_EPSILON,
) -> np.ndarray:
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    columns = []
    for index in range(NUM_FOURIER_COEFFICIENTS):
        plus = coefficients_array.copy()
        minus = coefficients_array.copy()
        plus[:, index] += epsilon
        minus[:, index] -= epsilon
        plus_losses = np.asarray(evaluate_controlled_batch(q, plus, seeds).losses)
        minus_losses = np.asarray(evaluate_controlled_batch(q, minus, seeds).losses)
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def evaluate_full_trajectory(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seed: int,
) -> TrajectoryResult:
    outputs = _SIMULATE_TRAJECTORY(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(forcing_history(seed), dtype=jnp.float64),
    )
    return TrajectoryResult(
        displacement=outputs[0],
        velocity=outputs[1],
        slip=outputs[2],
        stick_to_slip=jnp.sum(outputs[3], axis=0),
        slip_to_stick=jnp.sum(outputs[4], axis=0),
    )
