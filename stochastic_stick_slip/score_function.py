"""Score-function gradients for the independent S3 stochastic-event probe."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.model import NUM_STEPS, preload_history_with_basis
from stochastic_stick_slip.showcase import FOURIER_BASIS
from stochastic_stick_slip.stochastic_event import (
    SIMULATE_BATCH,
    WEAK_PROBABILITY_REFERENCE,
    WEAK_PROBABILITY_SCALE,
)


EVENT_BASE_SEED = 20260819
ITERATION_ZERO_STREAM = 0
REFERENCE_STREAM = 1
TRAINING_STREAM = 2
EVALUATION_STREAM = 3


def friction_uniform_realization_bank(
    seeds: np.ndarray,
    num_realizations: int,
    stream_id: int,
    iteration: int = 0,
) -> np.ndarray:
    """Return deterministic event draws with shape [condition,R,time,contact]."""
    return np.stack(
        [
            np.stack(
                [
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                EVENT_BASE_SEED,
                                int(stream_id),
                                int(iteration),
                                int(seed),
                                realization,
                            ]
                        )
                    ).uniform(0.0, 1.0, size=(NUM_STEPS, 2))
                    for realization in range(num_realizations)
                ]
            )
            for seed in np.asarray(seeds)
        ]
    )


def leave_one_out_baseline(losses: jax.Array) -> jax.Array:
    """Return the mean of the other event realizations along the last axis."""
    losses = jnp.asarray(losses, dtype=jnp.float64)
    if losses.shape[-1] < 2:
        raise ValueError("leave-one-out baseline requires at least two realizations")
    return (jnp.sum(losses, axis=-1, keepdims=True) - losses) / (
        losses.shape[-1] - 1
    )


def _condition_trajectories(q, coefficient, forcing, uniforms):
    num_realizations = uniforms.shape[0]
    coefficients = jnp.repeat(coefficient[None, :], num_realizations, axis=0)
    forcing_batch = jnp.repeat(forcing[None, :], num_realizations, axis=0)
    outputs = SIMULATE_BATCH(q, coefficients, forcing_batch, uniforms)
    displacement = outputs[0]
    slip = outputs[2]
    weak_state = outputs[5]
    losses = jnp.mean(displacement**2, axis=1)

    preload = preload_history_with_basis(q[1], coefficients, FOURIER_BASIS)
    probability_logit = (
        WEAK_PROBABILITY_REFERENCE - preload
    ) / WEAK_PROBABILITY_SCALE
    probability = jax.nn.sigmoid(probability_logit)
    selected_weak = uniforms < probability[:, :, None]
    selected_log_probability = jnp.where(
        selected_weak,
        jax.nn.log_sigmoid(probability_logit)[:, :, None],
        jax.nn.log_sigmoid(-probability_logit)[:, :, None],
    )

    previous_slip = jnp.concatenate(
        [jnp.zeros_like(slip[:, :1]), slip[:, :-1]], axis=1
    )
    renewals = jnp.logical_and(previous_slip, jnp.logical_not(slip))
    initial_log_probability = jnp.sum(selected_log_probability[:, 0], axis=1)
    renewal_log_probability = jnp.sum(
        jnp.where(renewals, selected_log_probability, 0.0), axis=(1, 2)
    )
    return (
        losses,
        initial_log_probability + renewal_log_probability,
        weak_state,
        slip,
        renewals,
    )


def _branchwise_condition_objective(coefficient, q, forcing, uniforms):
    losses = _condition_trajectories(q, coefficient, forcing, uniforms)[0]
    return jnp.mean(losses)


def _score_condition_surrogate(coefficient, q, forcing, uniforms):
    losses, log_probability, _, _, _ = _condition_trajectories(
        q, coefficient, forcing, uniforms
    )
    baseline = leave_one_out_baseline(losses)
    advantage = jax.lax.stop_gradient(losses - baseline)
    return jnp.mean(advantage * log_probability)


_BRANCHWISE_BATCH_VALUE_AND_GRAD = jax.jit(
    jax.vmap(
        jax.value_and_grad(_branchwise_condition_objective),
        in_axes=(0, None, 0, 0),
    )
)
_SCORE_BATCH_VALUE_AND_GRAD = jax.jit(
    jax.vmap(
        jax.value_and_grad(_score_condition_surrogate),
        in_axes=(0, None, 0, 0),
    )
)
_CONDITION_MEAN_LOSSES = jax.jit(
    jax.vmap(_branchwise_condition_objective, in_axes=(0, None, 0, 0))
)


def branchwise_condition_gradients(q, coefficients, forcing, uniforms):
    """Return per-condition mean losses and unweighted branchwise gradients."""
    values, gradients = _BRANCHWISE_BATCH_VALUE_AND_GRAD(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    return np.asarray(values), np.asarray(gradients)


def score_function_condition_gradients(q, coefficients, forcing, uniforms):
    """Return score-surrogate values and unweighted score gradients."""
    values, gradients = _SCORE_BATCH_VALUE_AND_GRAD(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    return np.asarray(values), np.asarray(gradients)


def condition_mean_losses(q, coefficients, forcing, uniforms) -> np.ndarray:
    """Return each forcing condition's objective averaged over realizations."""
    return np.asarray(
        _CONDITION_MEAN_LOSSES(
            jnp.asarray(coefficients, dtype=jnp.float64),
            jnp.asarray(q, dtype=jnp.float64),
            jnp.asarray(forcing, dtype=jnp.float64),
            jnp.asarray(uniforms, dtype=jnp.float64),
        )
    )


def condition_event_details(q, coefficient, forcing, uniforms):
    """Return one condition's losses, log probabilities, and hard states."""
    return tuple(
        np.asarray(value)
        for value in _condition_trajectories(
            jnp.asarray(q, dtype=jnp.float64),
            jnp.asarray(coefficient, dtype=jnp.float64),
            jnp.asarray(forcing, dtype=jnp.float64),
            jnp.asarray(uniforms, dtype=jnp.float64),
        )
    )


def mc_centered_fd_condition_gradients(
    q,
    coefficients,
    forcing,
    uniforms,
    epsilon: float,
) -> np.ndarray:
    """Return per-condition finite-radius sensitivities for five columns."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    columns = []
    for column in range(coefficients.shape[1]):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[:, column] += epsilon
        minus[:, column] -= epsilon
        plus_losses = condition_mean_losses(q, plus, forcing, uniforms)
        minus_losses = condition_mean_losses(q, minus, forcing, uniforms)
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)
