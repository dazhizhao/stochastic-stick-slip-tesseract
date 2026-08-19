import numpy as np

from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.stochastic_event import (
    centered_fd_coefficient_jacobian,
    direct_ad_batch_objective_and_gradient,
    evaluate_controlled_batch,
    friction_uniform_history,
    stochastic_inputs,
)


BASE_Q = np.array([0.2, 0.04], dtype=np.float64)


def test_stochastic_event_reproducibility_and_state_support() -> None:
    seeds = H4_TRAINING_SEEDS
    coefficients = np.zeros((len(seeds), 5), dtype=np.float64)
    first = evaluate_controlled_batch(BASE_Q, coefficients, seeds)
    second = evaluate_controlled_batch(BASE_Q, coefficients, seeds)

    assert np.array_equal(friction_uniform_history(11), friction_uniform_history(11))
    assert np.array_equal(np.asarray(first.weak_state), np.asarray(second.weak_state))
    assert np.sum(np.asarray(first.weak_selections)) > 0
    assert np.sum(np.asarray(first.strong_selections)) > 0
    assert np.all(np.isfinite(np.asarray(first.losses)))


def test_stochastic_event_gradients_are_finite_and_nonzero() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    coefficients = np.zeros((8, 5), dtype=np.float64)
    forcing, uniforms = stochastic_inputs(seeds)
    objective, direct_gradient = direct_ad_batch_objective_and_gradient(
        BASE_Q, coefficients, forcing, uniforms
    )
    fd_gradient = centered_fd_coefficient_jacobian(
        BASE_Q, coefficients, forcing, uniforms
    )

    assert np.isfinite(objective)
    for gradient in (direct_gradient, fd_gradient):
        assert gradient.shape == (8, 5)
        assert np.all(np.isfinite(gradient))
        assert np.linalg.norm(gradient) > 0.0
