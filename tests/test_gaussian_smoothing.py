import numpy as np

from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.gaussian_smoothing import (
    GAUSSIAN_SIGMA,
    fixed_gaussian_directions,
    gaussian_smoothing_coefficient_gradient,
)
from stochastic_stick_slip.stochastic_event import (
    evaluate_with_inputs,
    stochastic_inputs,
)


BASE_Q = np.array([0.2, 0.04], dtype=np.float64)


def test_fixed_gaussian_directions_are_reproducible() -> None:
    first = fixed_gaussian_directions()
    second = fixed_gaussian_directions()
    assert first.shape == (4, 8, 8, 5)
    assert np.array_equal(first, second)


def test_gaussian_gradient_matches_antithetic_pair_and_repeats() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    coefficients = np.zeros((8, 5), dtype=np.float64)
    forcing, uniforms = stochastic_inputs(seeds)
    direction = fixed_gaussian_directions()[0, :1]
    gradient = gaussian_smoothing_coefficient_gradient(
        BASE_Q,
        coefficients,
        forcing,
        uniforms,
        direction,
    )
    plus_losses = np.asarray(
        evaluate_with_inputs(
            BASE_Q,
            coefficients + GAUSSIAN_SIGMA * direction[0],
            forcing,
            uniforms,
        ).losses
    )
    minus_losses = np.asarray(
        evaluate_with_inputs(
            BASE_Q,
            coefficients - GAUSSIAN_SIGMA * direction[0],
            forcing,
            uniforms,
        ).losses
    )
    expected = (
        (plus_losses - minus_losses)[:, None]
        * direction[0]
        / (2.0 * GAUSSIAN_SIGMA)
    )
    repeated = gaussian_smoothing_coefficient_gradient(
        BASE_Q,
        coefficients,
        forcing,
        uniforms,
        direction,
    )

    assert gradient.shape == (8, 5)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
    assert np.allclose(gradient, expected, rtol=1e-12, atol=1e-14)
    assert np.array_equal(gradient, repeated)
