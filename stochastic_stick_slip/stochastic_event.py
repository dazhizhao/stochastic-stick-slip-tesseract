"""Independent hard stochastic friction-state variant for the S1 probe."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    CONTACT_STIFFNESS,
    FRICTION_COEFFICIENT,
    NUM_STEPS,
    _factor_solve,
    _select_contact_regime,
    forcing_batch_for_system,
    preload_history_with_basis,
)
from stochastic_stick_slip.showcase import FOURIER_BASIS, SYSTEM


WEAK_MULTIPLIER = 0.85
STRONG_MULTIPLIER = 1.15
WEAK_PROBABILITY_REFERENCE = 0.04
WEAK_PROBABILITY_SCALE = 0.01


@dataclass(frozen=True)
class StochasticEventBatchResult:
    losses: jax.Array
    displacement: jax.Array
    velocity: jax.Array
    slip: jax.Array
    stick_to_slip: jax.Array
    slip_to_stick: jax.Array
    weak_state: jax.Array
    weak_selections: jax.Array
    strong_selections: jax.Array
    renewals: jax.Array


def friction_uniform_history(seed: int) -> np.ndarray:
    """Return the hidden friction draws from a stream independent of forcing."""
    generator = np.random.default_rng(
        np.random.SeedSequence([int(seed), 1])
    )
    return generator.uniform(0.0, 1.0, size=(NUM_STEPS, 2))


def friction_uniform_batch(seeds: np.ndarray) -> jax.Array:
    return jnp.asarray(
        np.stack([friction_uniform_history(int(seed)) for seed in seeds]),
        dtype=jnp.float64,
    )


def stochastic_inputs(seeds: np.ndarray) -> tuple[jax.Array, jax.Array]:
    """Precompute all random inputs outside the differentiable simulator."""
    return (
        forcing_batch_for_system(seeds, SYSTEM),
        friction_uniform_batch(seeds),
    )


def _weak_probability(preload: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(
        (WEAK_PROBABILITY_REFERENCE - preload) / WEAK_PROBABILITY_SCALE
    )


def _build_stochastic_event_simulator():
    def simulate_batch_impl(q, coefficients, forcing, uniforms):
        damping, base_preload = q
        dt = SYSTEM.time_step
        mass = SYSTEM.mass
        stiffness = SYSTEM.stiffness
        load = SYSTEM.load
        observation = SYSTEM.observation
        contacts = SYSTEM.contacts
        preload = preload_history_with_basis(
            base_preload, coefficients, FOURIER_BASIS
        )

        effective_matrix = stiffness + mass / dt**2 + damping * mass / dt
        cholesky_factor = jnp.linalg.cholesky(effective_matrix)
        contact_response = _factor_solve(cholesky_factor, contacts)
        contact_compliance = contacts.T @ contact_response
        zero_displacement = jnp.zeros(stiffness.shape[0], dtype=jnp.float64)

        def simulate_seed(seed_forcing, seed_preload, seed_uniforms):
            initial_weak = seed_uniforms[0] < _weak_probability(seed_preload[0])
            initial_state = (
                zero_displacement,
                zero_displacement,
                jnp.zeros(2, dtype=jnp.float64),
                jnp.zeros(2, dtype=jnp.bool_),
                initial_weak,
            )

            def step(state, step_inputs):
                (
                    previous,
                    previous_previous,
                    slider_position,
                    was_slipping,
                    weak_state,
                ) = state
                external_force, current_preload, current_uniform = step_inputs
                multiplier = jnp.where(
                    weak_state, WEAK_MULTIPLIER, STRONG_MULTIPLIER
                )
                history = (
                    mass @ (2.0 * previous - previous_previous) / dt**2
                    + damping * mass @ previous / dt
                )
                free_solution = _factor_solve(
                    cholesky_factor, history + load * external_force
                )
                free_contact_displacement = contacts.T @ free_solution
                contact_force, contact_displacement, regime = (
                    _select_contact_regime(
                        free_contact_displacement,
                        slider_position,
                        contact_compliance,
                        FRICTION_COEFFICIENT
                        * current_preload
                        * multiplier,
                    )
                )
                displacement_vector = (
                    free_solution + contact_response @ contact_force
                )
                displacement = observation @ displacement_vector
                previous_displacement = observation @ previous
                velocity = (displacement - previous_displacement) / dt
                is_slipping = regime != 0
                next_slider_position = jnp.where(
                    is_slipping,
                    contact_displacement + contact_force / CONTACT_STIFFNESS,
                    slider_position,
                )
                stick_to_slip = jnp.logical_and(
                    jnp.logical_not(was_slipping), is_slipping
                )
                slip_to_stick = jnp.logical_and(
                    was_slipping, jnp.logical_not(is_slipping)
                )
                selected_weak = current_uniform < _weak_probability(
                    current_preload
                )
                next_weak_state = jnp.where(
                    slip_to_stick, selected_weak, weak_state
                )
                next_state = (
                    displacement_vector,
                    previous,
                    next_slider_position,
                    is_slipping,
                    next_weak_state,
                )
                output = (
                    displacement,
                    velocity,
                    is_slipping,
                    stick_to_slip,
                    slip_to_stick,
                    weak_state,
                    jnp.logical_and(slip_to_stick, selected_weak),
                    jnp.logical_and(slip_to_stick, jnp.logical_not(selected_weak)),
                )
                return next_state, output

            _, outputs = jax.lax.scan(
                step,
                initial_state,
                (seed_forcing, seed_preload, seed_uniforms),
            )
            return outputs

        return jax.vmap(simulate_seed)(forcing, preload, uniforms)

    return jax.jit(simulate_batch_impl)


SIMULATE_BATCH = _build_stochastic_event_simulator()


def evaluate_with_inputs(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
) -> StochasticEventBatchResult:
    outputs = SIMULATE_BATCH(
        jnp.asarray(q, dtype=jnp.float64),
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
        weak_state,
        renewal_weak,
        renewal_strong,
    ) = outputs
    initial_weak = weak_state[:, 0]
    return StochasticEventBatchResult(
        losses=jnp.mean(displacement**2, axis=1),
        displacement=displacement,
        velocity=velocity,
        slip=slip,
        stick_to_slip=jnp.sum(stick_to_slip, axis=1),
        slip_to_stick=jnp.sum(slip_to_stick, axis=1),
        weak_state=weak_state,
        weak_selections=(
            initial_weak.astype(jnp.int64)
            + jnp.sum(renewal_weak, axis=1, dtype=jnp.int64)
        ),
        strong_selections=(
            jnp.logical_not(initial_weak).astype(jnp.int64)
            + jnp.sum(renewal_strong, axis=1, dtype=jnp.int64)
        ),
        renewals=jnp.sum(slip_to_stick, axis=1, dtype=jnp.int64),
    )


def evaluate_controlled_batch(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> StochasticEventBatchResult:
    forcing, uniforms = stochastic_inputs(seeds)
    return evaluate_with_inputs(q, coefficients, forcing, uniforms)


def centered_fd_coefficient_jacobian(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
    epsilon: float = COEFFICIENT_FD_EPSILON,
) -> np.ndarray:
    """Return finite-radius per-seed sensitivities with shared random inputs."""
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    columns = []
    for index in range(coefficients_array.shape[1]):
        plus = coefficients_array.copy()
        minus = coefficients_array.copy()
        plus[:, index] += epsilon
        minus[:, index] -= epsilon
        plus_losses = np.asarray(
            evaluate_with_inputs(q, plus, forcing, uniforms).losses
        )
        minus_losses = np.asarray(
            evaluate_with_inputs(q, minus, forcing, uniforms).losses
        )
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def _mean_objective(coefficients, q, forcing, uniforms):
    displacement = SIMULATE_BATCH(q, coefficients, forcing, uniforms)[0]
    return jnp.mean(displacement**2)


_DIRECT_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_mean_objective))


def direct_ad_batch_objective_and_gradient(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
) -> tuple[float, np.ndarray]:
    """Return branchwise AD through the exact stochastic hard simulator."""
    value, gradient = _DIRECT_VALUE_AND_GRAD(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    return float(value), np.asarray(gradient)
