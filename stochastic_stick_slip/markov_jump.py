"""Hard two-state Markov preload histories for two friction contacts."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def markov_transition_probabilities(
    coefficients: jax.Array,
    fourier_basis: jax.Array,
    time_step: float,
    lambda_0: float,
    beta: float,
):
    """Return the per-step LOW/HIGH transition probabilities."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    signal = coefficients @ fourier_basis.T
    policy = jnp.tanh(signal)
    probability_low_to_high = 1.0 - jnp.exp(
        -lambda_0 * jnp.exp(beta * policy) * time_step
    )
    probability_high_to_low = 1.0 - jnp.exp(
        -lambda_0 * jnp.exp(-beta * policy) * time_step
    )
    return probability_low_to_high, probability_high_to_low


def generate_markov_preload_history(
    coefficients: jax.Array,
    fourier_basis: jax.Array,
    uniforms: jax.Array,
    time_step: float,
    lambda_0: float,
    beta: float,
    preload_low: float,
    preload_high: float,
):
    """Generate hard modes and endpoint preloads from explicit random tapes."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    uniforms = jnp.asarray(uniforms, dtype=jnp.float64)

    probability_low_to_high, probability_high_to_low = (
        markov_transition_probabilities(
            coefficients,
            fourier_basis,
            time_step,
            lambda_0,
            beta,
        )
    )

    initial_high = uniforms[:, :, 0, :] < 0.5
    transition_uniforms = jnp.moveaxis(uniforms[:, :, 1:, :], 2, 0)
    probability_low_to_high = jnp.moveaxis(probability_low_to_high, 1, 0)
    probability_high_to_low = jnp.moveaxis(probability_high_to_low, 1, 0)

    def transition(current_high, inputs):
        current_uniform, probability_lh, probability_hl = inputs
        transition_probability = jnp.where(
            current_high,
            probability_hl[:, None, None],
            probability_lh[:, None, None],
        )
        next_high = jnp.logical_xor(
            current_high, current_uniform < transition_probability
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
    modes = jnp.moveaxis(endpoint_modes, 0, 2)
    previous_modes = jnp.concatenate(
        (initial_high[:, :, None, :], modes[:, :, :-1, :]), axis=2
    )
    transition_counts = jnp.sum(modes != previous_modes, axis=2)
    high_mode_fraction = jnp.mean(modes, axis=2, dtype=jnp.float64)
    preload = jnp.where(modes, preload_high, preload_low)
    return modes, preload, transition_counts, high_mode_fraction
