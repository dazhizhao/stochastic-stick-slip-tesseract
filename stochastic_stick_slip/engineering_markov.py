"""Locked 32x4 Markov-jump engineering mechanics for Gate A."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.engineering_showcase import (
    FOURIER_BASIS,
    SYSTEM,
    forcing_batch,
)
from stochastic_stick_slip.markov_jump import generate_markov_preload_history
from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    NUM_STEPS,
    TRAINING_SEEDS,
    build_mechanics_batch_simulator,
)


DAMPING = 0.10
PRELOAD_LOW = 0.02
PRELOAD_HIGH = 0.06
BETA = 1.0
PERIOD_1 = 2.0 * np.pi / SYSTEM.omega_1
LAMBDA_0 = 1.0 / PERIOD_1
FD_EPSILON = COEFFICIENT_FD_EPSILON
GATE_A_FORCING_SEEDS = TRAINING_SEEDS.copy()
MARKOV_BASE_SEED = 20260819
MARKOV_STREAM = 4
MARKOV_ITERATION = 0


@dataclass(frozen=True)
class MarkovBatchResult:
    losses: jax.Array
    displacement: jax.Array
    velocity: jax.Array
    slip: jax.Array
    stick_to_slip: jax.Array
    slip_to_stick: jax.Array
    modes: jax.Array
    preload: jax.Array
    transition_counts: jax.Array
    high_mode_fraction: jax.Array


def markov_uniform_bank(num_realizations: int) -> np.ndarray:
    """Return the fixed Gate A tape bank as [condition,R,time+1,contact]."""
    return np.stack(
        [
            np.stack(
                [
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                MARKOV_BASE_SEED,
                                MARKOV_STREAM,
                                MARKOV_ITERATION,
                                int(seed),
                                realization,
                            ]
                        )
                    ).uniform(0.0, 1.0, size=(NUM_STEPS + 1, 2))
                    for realization in range(num_realizations)
                ]
            )
            for seed in GATE_A_FORCING_SEEDS
        ]
    )


MECHANICS_SIMULATOR = build_mechanics_batch_simulator(SYSTEM)


def _simulate_markov_bank(coefficients, forcing, uniforms):
    modes, preload, transition_counts, high_mode_fraction = (
        generate_markov_preload_history(
            coefficients,
            FOURIER_BASIS,
            uniforms,
            SYSTEM.time_step,
            LAMBDA_0,
            BETA,
            PRELOAD_LOW,
            PRELOAD_HIGH,
        )
    )
    num_conditions, num_realizations = preload.shape[:2]
    repeated_forcing = jnp.broadcast_to(
        forcing[:, None, :],
        (num_conditions, num_realizations, NUM_STEPS),
    )
    outputs = MECHANICS_SIMULATOR(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        repeated_forcing.reshape((-1, NUM_STEPS)),
        preload.reshape((-1, NUM_STEPS, 2)),
    )

    reshaped = []
    for output in outputs:
        reshaped.append(
            output.reshape((num_conditions, num_realizations) + output.shape[1:])
        )
    return (*reshaped, modes, preload, transition_counts, high_mode_fraction)


SIMULATE_MARKOV_BANK = jax.jit(_simulate_markov_bank)


def evaluate_markov_bank(
    coefficients: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
) -> MarkovBatchResult:
    outputs = SIMULATE_MARKOV_BANK(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    (
        displacement,
        velocity,
        slip,
        stick_to_slip,
        slip_to_stick,
        modes,
        preload,
        transition_counts,
        high_mode_fraction,
    ) = outputs
    return MarkovBatchResult(
        losses=jnp.mean(displacement**2, axis=2),
        displacement=displacement,
        velocity=velocity,
        slip=slip,
        stick_to_slip=jnp.sum(stick_to_slip, axis=2),
        slip_to_stick=jnp.sum(slip_to_stick, axis=2),
        modes=modes,
        preload=preload,
        transition_counts=transition_counts,
        high_mode_fraction=high_mode_fraction,
    )


def shared_coefficient_objective(shared_coefficients, forcing, uniforms):
    coefficients = jnp.broadcast_to(
        shared_coefficients[None, :],
        (forcing.shape[0], shared_coefficients.shape[0]),
    )
    displacement = SIMULATE_MARKOV_BANK(coefficients, forcing, uniforms)[0]
    return jnp.mean(displacement**2)


_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(shared_coefficient_objective))


def direct_ad_objective_and_gradient(shared_coefficients, forcing, uniforms):
    value, gradient = _VALUE_AND_GRAD(
        jnp.asarray(shared_coefficients, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    return float(value), np.asarray(gradient)


def gate_a_forcing() -> jax.Array:
    return forcing_batch(GATE_A_FORCING_SEEDS)
