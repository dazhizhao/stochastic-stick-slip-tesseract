"""Hard two-state Markov controller and CRN finite differences."""

from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from wu2019.dynamics import DEFAULT_SETTINGS, SimulationSettings
from wu2019.newmark import simulate_summary_raw


N_LOW = 30.0
N_HIGH = 50.0
BETA = 1.0
FD_EPSILON = 0.02
NUM_COEFFICIENTS = 5


@dataclass(frozen=True)
class MarkovEvaluation:
    objective: float
    amplitudes: np.ndarray
    frequency_means: np.ndarray
    modes: np.ndarray
    preload: np.ndarray
    transition_counts: np.ndarray
    high_mode_fraction: np.ndarray


@dataclass(frozen=True)
class CRNFiniteDifference:
    epsilon: float
    gradient: np.ndarray
    plus_objectives: np.ndarray
    minus_objectives: np.ndarray
    mode_difference_counts: np.ndarray


def phase_basis(settings: SimulationSettings = DEFAULT_SETTINGS) -> np.ndarray:
    phase = (
        2.0
        * np.pi
        * np.arange(1, settings.num_steps + 1, dtype=np.float64)
        / settings.steps_per_period
    )
    return np.column_stack(
        (
            np.ones_like(phase),
            np.cos(phase),
            np.sin(phase),
            np.cos(2.0 * phase),
            np.sin(2.0 * phase),
        )
    )


def uniform_bank(
    num_realizations: int,
    base_seed: int,
    settings: SimulationSettings = DEFAULT_SETTINGS,
) -> np.ndarray:
    """Return explicit phase-index tapes with shape [R, num_steps + 1]."""
    return np.stack(
        [
            np.random.default_rng(base_seed + realization).uniform(
                size=settings.num_steps + 1
            )
            for realization in range(num_realizations)
        ]
    )


def _generate_markov_history(coefficients, basis, uniforms, steps_per_period):
    signal = basis @ coefficients
    policy = jnp.tanh(signal)
    probability_low_to_high = 1.0 - jnp.exp(
        -jnp.exp(BETA * policy) / steps_per_period
    )
    probability_high_to_low = 1.0 - jnp.exp(
        -jnp.exp(-BETA * policy) / steps_per_period
    )

    initial_high = uniforms[:, 0] < 0.5
    transition_uniforms = jnp.moveaxis(uniforms[:, 1:], 1, 0)

    def transition(current_high, inputs):
        current_uniform, probability_lh, probability_hl = inputs
        probability = jnp.where(
            current_high, probability_hl, probability_lh
        )
        next_high = jnp.logical_xor(
            current_high, current_uniform < probability
        )
        return next_high, next_high

    _, endpoint_modes = jax.lax.scan(
        transition,
        initial_high,
        (
            transition_uniforms,
            probability_low_to_high,
            probability_high_to_low,
        ),
    )
    modes = jnp.moveaxis(endpoint_modes, 0, 1)
    previous_modes = jnp.concatenate(
        (initial_high[:, None], modes[:, :-1]), axis=1
    )
    transition_counts = jnp.sum(modes != previous_modes, axis=1)
    high_mode_fraction = jnp.mean(modes, axis=1, dtype=jnp.float64)
    preload = jnp.where(modes, N_HIGH, N_LOW)
    return modes, preload, transition_counts, high_mode_fraction


@partial(
    jax.jit,
    static_argnames=("steps_per_period", "measurement_periods"),
)
def _evaluate_raw(
    coefficients,
    omegas,
    basis,
    uniforms,
    steps_per_period: int,
    measurement_periods: int,
):
    modes, preload, transition_counts, high_mode_fraction = (
        _generate_markov_history(
            coefficients, basis, uniforms, steps_per_period
        )
    )
    num_frequencies = omegas.shape[0]
    num_realizations = uniforms.shape[0]
    repeated_omegas = jnp.repeat(omegas, num_realizations)
    repeated_preload = jnp.broadcast_to(
        preload[None, :, :],
        (num_frequencies, num_realizations, preload.shape[1]),
    ).reshape((num_frequencies * num_realizations, preload.shape[1]))
    summaries = jax.vmap(
        lambda omega, normal_force: simulate_summary_raw(
            omega,
            normal_force,
            steps_per_period,
            measurement_periods,
        )
    )(repeated_omegas, repeated_preload)
    amplitudes = summaries[0].reshape(
        (num_frequencies, num_realizations)
    )
    frequency_means = jnp.mean(amplitudes, axis=1)
    objective = jnp.max(frequency_means)
    return (
        objective,
        amplitudes,
        frequency_means,
        modes,
        preload,
        transition_counts,
        high_mode_fraction,
    )


def evaluate_markov(
    coefficients,
    omegas,
    uniforms,
    settings: SimulationSettings = DEFAULT_SETTINGS,
) -> MarkovEvaluation:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if coefficients.shape != (NUM_COEFFICIENTS,):
        raise ValueError("Markov coefficients must have shape (5,)")
    if uniforms.ndim != 2 or uniforms.shape[1] != settings.num_steps + 1:
        raise ValueError("Markov uniform bank has the wrong shape")
    outputs = _evaluate_raw(
        jnp.asarray(coefficients),
        jnp.asarray(omegas),
        jnp.asarray(phase_basis(settings)),
        jnp.asarray(uniforms),
        settings.steps_per_period,
        settings.measurement_periods,
    )
    arrays = [np.asarray(value) for value in outputs]
    return MarkovEvaluation(
        objective=float(arrays[0]),
        amplitudes=arrays[1],
        frequency_means=arrays[2],
        modes=arrays[3],
        preload=arrays[4],
        transition_counts=arrays[5],
        high_mode_fraction=arrays[6],
    )


@partial(
    jax.jit,
    static_argnames=("steps_per_period", "measurement_periods"),
)
def _direct_value_and_gradient(
    coefficients,
    omegas,
    basis,
    uniforms,
    steps_per_period: int,
    measurement_periods: int,
):
    objective = lambda values: _evaluate_raw(
        values,
        omegas,
        basis,
        uniforms,
        steps_per_period,
        measurement_periods,
    )[0]
    return jax.value_and_grad(objective)(coefficients)


def direct_ad_objective_and_gradient(
    coefficients,
    omegas,
    uniforms,
    settings: SimulationSettings = DEFAULT_SETTINGS,
):
    value, gradient = _direct_value_and_gradient(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(omegas, dtype=jnp.float64),
        jnp.asarray(phase_basis(settings)),
        jnp.asarray(uniforms, dtype=jnp.float64),
        settings.steps_per_period,
        settings.measurement_periods,
    )
    return float(value), np.asarray(gradient)


def crn_centered_finite_difference(
    coefficients,
    omegas,
    uniforms,
    epsilon: float = FD_EPSILON,
    settings: SimulationSettings = DEFAULT_SETTINGS,
) -> CRNFiniteDifference:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    gradients = np.empty(NUM_COEFFICIENTS, dtype=np.float64)
    plus_objectives = np.empty(NUM_COEFFICIENTS, dtype=np.float64)
    minus_objectives = np.empty(NUM_COEFFICIENTS, dtype=np.float64)
    mode_difference_counts = np.empty(NUM_COEFFICIENTS, dtype=np.int64)
    for index in range(NUM_COEFFICIENTS):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_result = evaluate_markov(
            plus, omegas, uniforms, settings
        )
        minus_result = evaluate_markov(
            minus, omegas, uniforms, settings
        )
        plus_objectives[index] = plus_result.objective
        minus_objectives[index] = minus_result.objective
        gradients[index] = (
            plus_result.objective - minus_result.objective
        ) / (2.0 * epsilon)
        mode_difference_counts[index] = np.count_nonzero(
            plus_result.modes != minus_result.modes
        )
    return CRNFiniteDifference(
        epsilon=epsilon,
        gradient=gradients,
        plus_objectives=plus_objectives,
        minus_objectives=minus_objectives,
        mode_difference_counts=mode_difference_counts,
    )
