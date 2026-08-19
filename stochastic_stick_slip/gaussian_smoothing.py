"""Fixed Monte-Carlo Gaussian smoothing for the S2 stochastic-event probe."""

import numpy as np

from stochastic_stick_slip.model import COEFFICIENT_FD_EPSILON
from stochastic_stick_slip.stochastic_event import evaluate_with_inputs


NUM_GAUSSIAN_DIRECTIONS = 8
GAUSSIAN_DIRECTION_SEED = 20260819
GAUSSIAN_SIGMA = COEFFICIENT_FD_EPSILON / np.sqrt(5.0)


def fixed_gaussian_directions() -> np.ndarray:
    """Return the frozen four-batch direction bank with shape [4,8,8,5]."""
    generator = np.random.default_rng(GAUSSIAN_DIRECTION_SEED)
    return generator.normal(size=(4, NUM_GAUSSIAN_DIRECTIONS, 8, 5))


def gaussian_smoothing_coefficient_gradient(
    q,
    coefficients,
    forcing,
    uniforms,
    directions,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Estimate per-seed sensitivities using antithetic hard forwards."""
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    directions_array = np.asarray(directions, dtype=np.float64)
    gradient = np.zeros_like(coefficients_array)
    for direction in directions_array:
        plus_losses = np.asarray(
            evaluate_with_inputs(
                q,
                coefficients_array + sigma * direction,
                forcing,
                uniforms,
            ).losses
        )
        minus_losses = np.asarray(
            evaluate_with_inputs(
                q,
                coefficients_array - sigma * direction,
                forcing,
                uniforms,
            ).losses
        )
        directional_sensitivity = (plus_losses - minus_losses) / (2.0 * sigma)
        gradient += directional_sensitivity[:, None] * direction
    return gradient / len(directions_array)
