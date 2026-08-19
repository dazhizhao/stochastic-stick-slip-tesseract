import numpy as np

from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.score_function import (
    TRAINING_STREAM,
    branchwise_condition_gradients,
    condition_event_details,
    friction_uniform_realization_bank,
    leave_one_out_baseline,
    score_function_condition_gradients,
)
from stochastic_stick_slip.stochastic_event import stochastic_inputs


BASE_Q = np.array([0.2, 0.04], dtype=np.float64)


def test_event_bank_reproduces_states_and_log_probability() -> None:
    seeds = H4_TRAINING_SEEDS[:1]
    forcing = np.asarray(stochastic_inputs(seeds)[0][0])
    uniforms = friction_uniform_realization_bank(seeds, 2, TRAINING_STREAM, 3)[0]
    first = condition_event_details(
        BASE_Q, np.zeros(5), forcing, uniforms
    )
    second = condition_event_details(
        BASE_Q, np.zeros(5), forcing, uniforms
    )

    assert np.array_equal(first[2], second[2])
    assert np.array_equal(first[3], second[3])
    assert np.array_equal(first[1], second[1])


def test_log_probability_sum_is_finite() -> None:
    seeds = H4_TRAINING_SEEDS[:1]
    forcing = np.asarray(stochastic_inputs(seeds)[0][0])
    uniforms = friction_uniform_realization_bank(seeds, 2, TRAINING_STREAM, 0)[0]
    log_probability = condition_event_details(
        BASE_Q, np.zeros(5), forcing, uniforms
    )[1]
    assert np.all(np.isfinite(log_probability))


def test_score_function_gradient_is_finite_and_nonzero() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    forcing = np.asarray(stochastic_inputs(seeds)[0])
    uniforms = friction_uniform_realization_bank(seeds, 2, TRAINING_STREAM, 0)
    coefficients = np.zeros((8, 5), dtype=np.float64)
    _, gradient = score_function_condition_gradients(
        BASE_Q, coefficients, forcing, uniforms
    )
    _, branchwise = branchwise_condition_gradients(
        BASE_Q, coefficients, forcing, uniforms
    )

    assert gradient.shape == (8, 5)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
    assert branchwise.shape == (8, 5)


def test_leave_one_out_baseline_shape_and_scaling() -> None:
    losses = np.array([[1.0, 2.0, 4.0, 8.0]])
    baseline = np.asarray(leave_one_out_baseline(losses))
    expected = np.array([[14.0 / 3.0, 13.0 / 3.0, 11.0 / 3.0, 7.0 / 3.0]])

    assert baseline.shape == losses.shape
    assert np.allclose(baseline, expected, rtol=0.0, atol=1e-15)
    assert np.isclose(np.mean(baseline), np.mean(losses))
    assert np.isclose(np.sum(losses - baseline), 0.0)


def test_training_schedule_is_shared_and_iteration_dependent() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    first_path = friction_uniform_realization_bank(
        seeds, 4, TRAINING_STREAM, iteration=7
    )
    second_path = friction_uniform_realization_bank(
        seeds, 4, TRAINING_STREAM, iteration=7
    )
    next_iteration = friction_uniform_realization_bank(
        seeds, 4, TRAINING_STREAM, iteration=8
    )

    assert np.array_equal(first_path, second_path)
    assert not np.array_equal(first_path, next_iteration)
