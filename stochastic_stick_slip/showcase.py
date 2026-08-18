"""Locked 32x4 mechanics binding for the final Hackathon showcase."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    FD_RELATIVE_EPSILON,
    NUM_FOURIER_COEFFICIENTS,
    NUM_STEPS,
    BatchResult,
    TrajectoryResult,
    _assemble_system,
    build_batch_simulator,
    build_fourier_basis,
    build_trajectory_simulator,
    evaluate_controlled_batch_for_system,
    evaluate_trajectory_for_system,
    preload_history_with_basis,
)


NUM_ELEMENTS_X = 32
NUM_ELEMENTS_Y = 4
CONTACT_COLUMNS = (22, 30)

SYSTEM = _assemble_system(
    num_elements_x=NUM_ELEMENTS_X,
    num_elements_y=NUM_ELEMENTS_Y,
    contact_columns=CONTACT_COLUMNS,
)
FOURIER_BASIS = build_fourier_basis(SYSTEM)
_SIMULATE_BATCH = build_batch_simulator(SYSTEM, FOURIER_BASIS)
_SIMULATE_TRAJECTORY = build_trajectory_simulator(SYSTEM, FOURIER_BASIS)


def preload_history(base_preload, coefficients):
    return preload_history_with_basis(
        base_preload, coefficients, FOURIER_BASIS
    )


def evaluate_controlled_batch(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> BatchResult:
    return evaluate_controlled_batch_for_system(
        q, coefficients, seeds, SYSTEM, _SIMULATE_BATCH
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
        plus_losses = np.asarray(
            evaluate_controlled_batch(q, plus, seeds).losses
        )
        minus_losses = np.asarray(
            evaluate_controlled_batch(q, minus, seeds).losses
        )
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def evaluate_full_trajectory(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seed: int,
) -> TrajectoryResult:
    return evaluate_trajectory_for_system(
        q,
        coefficients,
        seed,
        SYSTEM,
        _SIMULATE_TRAJECTORY,
    )


def full_nodal_field(free_field: np.ndarray | jax.Array) -> np.ndarray:
    """Restore zero-valued fixed DOFs and reshape a free field to nodes."""
    free_array = np.asarray(free_field)
    full_shape = free_array.shape[:-1] + (SYSTEM.num_total_dofs,)
    full = np.zeros(full_shape, dtype=np.float64)
    full[..., SYSTEM.free_dofs] = free_array
    return full.reshape(free_array.shape[:-1] + (len(SYSTEM.points), 2))
