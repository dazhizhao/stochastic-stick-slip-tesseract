"""Causal state-aware hard Markov controller for the Wu benchmark."""

from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from wu2019.dynamics import (
    DEFAULT_PARAMETERS,
    DEFAULT_SETTINGS,
    SimulationSettings,
)
from wu2019.markov import (
    BETA,
    FD_EPSILON,
    N_HIGH,
    N_LOW,
    phase_basis,
)
from wu2019.newmark import _advance_newmark_state, _initial_newmark_state


NUM_STATE_AWARE_COEFFICIENTS = 7
DISPLACEMENT_SCALE = 2.4e-3
VELOCITY_SCALE = 0.48
PHASE2_COEFFICIENTS = np.array(
    [
        0.13100082433292837,
        0.10486369345810433,
        0.2783754374253086,
        -0.3888951380644648,
        0.3409590405511461,
    ],
    dtype=np.float64,
)
INITIAL_STATE_AWARE_COEFFICIENTS = np.concatenate(
    (PHASE2_COEFFICIENTS, np.zeros(2, dtype=np.float64))
)


@dataclass(frozen=True)
class StateAwareEvaluation:
    objective: float
    amplitudes: np.ndarray
    frequency_means: np.ndarray
    transition_counts: np.ndarray
    high_mode_fraction: np.ndarray


@dataclass(frozen=True)
class StateAwareReplay:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    modes: np.ndarray
    preload: np.ndarray
    score: np.ndarray


@dataclass(frozen=True)
class StateAwareFiniteDifference:
    epsilon: float
    gradient: np.ndarray
    plus_objectives: np.ndarray
    minus_objectives: np.ndarray
    mode_difference_counts: np.ndarray
    history_omega: float


def _transition_probability(score, current_high, steps_per_period):
    policy = jnp.tanh(score)
    probability_low_to_high = 1.0 - jnp.exp(
        -jnp.exp(BETA * policy) / steps_per_period
    )
    probability_high_to_low = 1.0 - jnp.exp(
        -jnp.exp(-BETA * policy) / steps_per_period
    )
    return jnp.where(
        current_high,
        probability_high_to_low,
        probability_low_to_high,
    )


def _state_aware_score(coefficients, basis_row, mechanics_state):
    displacement, velocity = mechanics_state[:2]
    return (
        coefficients[:5] @ basis_row
        + coefficients[5] * velocity / VELOCITY_SCALE
        + coefficients[6] * displacement / DISPLACEMENT_SCALE
    )


def _simulate_summary_single(
    coefficients,
    omega,
    basis,
    uniforms,
    steps_per_period,
    num_periods,
    measurement_periods,
):
    dt = 2.0 * jnp.pi / (omega * steps_per_period)
    initial_high = uniforms[0] < 0.5
    initial_carry = (
        _initial_newmark_state(),
        initial_high,
        jnp.asarray(-jnp.inf, dtype=jnp.float64),
        jnp.zeros(measurement_periods, dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(0, dtype=jnp.int64),
    )
    indices = jnp.arange(basis.shape[0], dtype=jnp.int64)

    def step(carry, inputs):
        (
            mechanics_state,
            current_high,
            period_maximum,
            measured_maxima,
            transition_count,
            high_count,
        ) = carry
        index, basis_row, current_uniform = inputs
        score = _state_aware_score(
            coefficients, basis_row, mechanics_state
        )
        probability = _transition_probability(
            score, current_high, steps_per_period
        )
        next_high = jnp.logical_xor(
            current_high, current_uniform < probability
        )
        normal_force = jnp.where(next_high, N_HIGH, N_LOW)
        time = dt * (index + 1)
        external_force = (
            DEFAULT_PARAMETERS.forcing_amplitude * jnp.sin(omega * time)
        )
        next_mechanics_state, _ = _advance_newmark_state(
            mechanics_state,
            external_force,
            normal_force,
            dt,
        )
        next_period_maximum = jnp.maximum(
            period_maximum, next_mechanics_state[0]
        )
        end_of_period = (index + 1) % steps_per_period == 0
        period_number = (index + 1) // steps_per_period
        measured_period = jnp.logical_and(
            end_of_period,
            period_number > num_periods - measurement_periods,
        )
        measurement_index = jnp.clip(
            period_number
            - (num_periods - measurement_periods)
            - 1,
            0,
            measurement_periods - 1,
        )
        current_measured_maximum = measured_maxima[measurement_index]
        next_measured_maxima = measured_maxima.at[measurement_index].set(
            jnp.where(
                measured_period,
                next_period_maximum,
                current_measured_maximum,
            )
        )
        next_period_maximum = jnp.where(
            end_of_period, -jnp.inf, next_period_maximum
        )
        next_carry = (
            next_mechanics_state,
            next_high,
            next_period_maximum,
            next_measured_maxima,
            transition_count + (next_high != current_high),
            high_count + next_high,
        )
        return next_carry, None

    final_carry, _ = jax.lax.scan(
        step,
        initial_carry,
        (indices, basis, uniforms[1:]),
    )
    measured_maxima = final_carry[3]
    transition_count = final_carry[4]
    high_count = final_carry[5]
    return (
        jnp.mean(measured_maxima),
        transition_count,
        high_count / basis.shape[0],
    )


@partial(
    jax.jit,
    static_argnames=(
        "steps_per_period",
        "num_periods",
        "measurement_periods",
    ),
)
def _evaluate_raw(
    coefficients,
    omegas,
    basis,
    uniforms,
    steps_per_period,
    num_periods,
    measurement_periods,
):
    num_frequencies = omegas.shape[0]
    num_realizations = uniforms.shape[0]
    repeated_omegas = jnp.repeat(omegas, num_realizations)
    repeated_uniforms = jnp.broadcast_to(
        uniforms[None, :, :],
        (num_frequencies, num_realizations, uniforms.shape[1]),
    ).reshape((num_frequencies * num_realizations, uniforms.shape[1]))
    outputs = jax.vmap(
        lambda omega, tape: _simulate_summary_single(
            coefficients,
            omega,
            basis,
            tape,
            steps_per_period,
            num_periods,
            measurement_periods,
        )
    )(repeated_omegas, repeated_uniforms)
    amplitudes, transition_counts, high_mode_fraction = [
        output.reshape((num_frequencies, num_realizations))
        for output in outputs
    ]
    frequency_means = jnp.mean(amplitudes, axis=1)
    objective = jnp.max(frequency_means)
    return (
        objective,
        amplitudes,
        frequency_means,
        transition_counts,
        high_mode_fraction,
    )


def evaluate_state_aware(
    coefficients,
    omegas,
    uniforms,
    settings: SimulationSettings = DEFAULT_SETTINGS,
) -> StateAwareEvaluation:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if coefficients.shape != (NUM_STATE_AWARE_COEFFICIENTS,):
        raise ValueError("state-aware coefficients must have shape (7,)")
    if uniforms.ndim != 2 or uniforms.shape[1] != settings.num_steps + 1:
        raise ValueError("state-aware uniform bank has the wrong shape")
    outputs = _evaluate_raw(
        jnp.asarray(coefficients),
        jnp.asarray(omegas),
        jnp.asarray(phase_basis(settings)),
        jnp.asarray(uniforms),
        settings.steps_per_period,
        settings.num_periods,
        settings.measurement_periods,
    )
    arrays = [np.asarray(value) for value in outputs]
    return StateAwareEvaluation(
        objective=float(arrays[0]),
        amplitudes=arrays[1],
        frequency_means=arrays[2],
        transition_counts=arrays[3],
        high_mode_fraction=arrays[4],
    )


def _replay_single(
    coefficients,
    omega,
    basis,
    uniforms,
    steps_per_period,
):
    dt = 2.0 * jnp.pi / (omega * steps_per_period)
    initial_carry = (_initial_newmark_state(), uniforms[0] < 0.5)
    indices = jnp.arange(basis.shape[0], dtype=jnp.int64)

    def step(carry, inputs):
        mechanics_state, current_high = carry
        index, basis_row, current_uniform = inputs
        score = _state_aware_score(
            coefficients, basis_row, mechanics_state
        )
        probability = _transition_probability(
            score, current_high, steps_per_period
        )
        next_high = jnp.logical_xor(
            current_high, current_uniform < probability
        )
        normal_force = jnp.where(next_high, N_HIGH, N_LOW)
        time = dt * (index + 1)
        external_force = (
            DEFAULT_PARAMETERS.forcing_amplitude * jnp.sin(omega * time)
        )
        next_mechanics_state, _ = _advance_newmark_state(
            mechanics_state,
            external_force,
            normal_force,
            dt,
        )
        output = (
            time,
            next_mechanics_state[0],
            next_mechanics_state[1],
            next_high,
            normal_force,
            score,
        )
        return (next_mechanics_state, next_high), output

    _, outputs = jax.lax.scan(
        step,
        initial_carry,
        (indices, basis, uniforms[1:]),
    )
    return outputs


_REPLAY_JIT = jax.jit(_replay_single, static_argnames=("steps_per_period",))


def replay_state_aware(
    coefficients,
    omega,
    uniforms,
    settings: SimulationSettings = DEFAULT_SETTINGS,
) -> StateAwareReplay:
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if uniforms.shape != (settings.num_steps + 1,):
        raise ValueError("state-aware replay tape has the wrong shape")
    outputs = _REPLAY_JIT(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(omega, dtype=jnp.float64),
        jnp.asarray(phase_basis(settings)),
        jnp.asarray(uniforms),
        settings.steps_per_period,
    )
    arrays = [np.asarray(value) for value in outputs]
    return StateAwareReplay(*arrays)


@partial(
    jax.jit,
    static_argnames=(
        "steps_per_period",
        "num_periods",
        "measurement_periods",
    ),
)
def _direct_value_and_gradient(
    coefficients,
    omegas,
    basis,
    uniforms,
    steps_per_period,
    num_periods,
    measurement_periods,
):
    objective = lambda values: _evaluate_raw(
        values,
        omegas,
        basis,
        uniforms,
        steps_per_period,
        num_periods,
        measurement_periods,
    )[0]
    return jax.value_and_grad(objective)(coefficients)


def direct_ad_state_aware(
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
        settings.num_periods,
        settings.measurement_periods,
    )
    return float(value), np.asarray(gradient)


def crn_fd_state_aware(
    coefficients,
    omegas,
    uniforms,
    epsilon: float = FD_EPSILON,
    settings: SimulationSettings = DEFAULT_SETTINGS,
    check_mode_history: bool = True,
) -> StateAwareFiniteDifference:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    gradients = np.empty(NUM_STATE_AWARE_COEFFICIENTS, dtype=np.float64)
    plus_objectives = np.empty_like(gradients)
    minus_objectives = np.empty_like(gradients)
    mode_difference_counts = np.empty(
        NUM_STATE_AWARE_COEFFICIENTS, dtype=np.int64
    )
    history_omega = float(omegas[len(omegas) // 2])
    for index in range(NUM_STATE_AWARE_COEFFICIENTS):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_result = evaluate_state_aware(
            plus, omegas, uniforms, settings
        )
        minus_result = evaluate_state_aware(
            minus, omegas, uniforms, settings
        )
        plus_objectives[index] = plus_result.objective
        minus_objectives[index] = minus_result.objective
        gradients[index] = (
            plus_result.objective - minus_result.objective
        ) / (2.0 * epsilon)
        if check_mode_history:
            differences = 0
            for tape in uniforms:
                plus_replay = replay_state_aware(
                    plus, history_omega, tape, settings
                )
                minus_replay = replay_state_aware(
                    minus, history_omega, tape, settings
                )
                differences += np.count_nonzero(
                    plus_replay.modes != minus_replay.modes
                )
            mode_difference_counts[index] = differences
        else:
            mode_difference_counts[index] = -1
    return StateAwareFiniteDifference(
        epsilon=epsilon,
        gradient=gradients,
        plus_objectives=plus_objectives,
        minus_objectives=minus_objectives,
        mode_difference_counts=mode_difference_counts,
        history_omega=history_omega,
    )
