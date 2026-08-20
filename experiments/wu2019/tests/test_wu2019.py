import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from wu2019.controller import constant_normal_force, harmonic_normal_force
from wu2019.dynamics import SimulationSettings
from wu2019.newmark import simulate_summary_batch, simulate_trajectory
from wu2019.markov import (
    crn_centered_finite_difference,
    direct_ad_objective_and_gradient,
    evaluate_markov,
    phase_basis,
    uniform_bank,
)
from wu2019.state_aware import (
    INITIAL_STATE_AWARE_COEFFICIENTS,
    PHASE2_COEFFICIENTS,
    crn_fd_state_aware,
    direct_ad_state_aware,
    evaluate_state_aware,
    replay_state_aware,
)


TEST_SETTINGS = SimulationSettings(
    steps_per_period=120,
    num_periods=30,
    measurement_periods=20,
)


def test_constant_normal_force_forward_is_finite() -> None:
    normal_force = constant_normal_force(40.0, TEST_SETTINGS)
    result = simulate_summary_batch(
        np.array([200.0]), normal_force, TEST_SETTINGS
    )
    assert np.all(np.isfinite(result.amplitude))
    assert result.amplitude[0] > 0.0
    assert result.friction_excess[0] <= 1e-10


def test_contact_sticks_slips_and_dissipates() -> None:
    normal_force = constant_normal_force(40.0, TEST_SETTINGS)
    trajectory = simulate_trajectory(202.0, normal_force, TEST_SETTINGS)
    measured = trajectory.slip[-20 * TEST_SETTINGS.steps_per_period :]
    assert np.any(measured)
    assert np.any(~measured)
    slider_increment = np.diff(
        np.concatenate(([0.0], trajectory.slider_displacement))
    )
    dissipated = np.sum(trajectory.friction_force * slider_increment)
    assert dissipated >= -1e-10


def test_harmonic_normal_force_stays_in_requested_range() -> None:
    normal_force = harmonic_normal_force(
        40.0, 10.0, 2, 4.4, TEST_SETTINGS
    )
    assert np.isclose(np.min(normal_force), 30.0, atol=0.02)
    assert np.isclose(np.max(normal_force), 50.0, atol=0.02)


def test_repeated_forward_is_exactly_reproducible() -> None:
    normal_force = harmonic_normal_force(
        40.0, 10.0, 2, 4.4, TEST_SETTINGS
    )
    first = simulate_summary_batch(
        np.array([200.0, 202.0]), normal_force, TEST_SETTINGS
    )
    second = simulate_summary_batch(
        np.array([200.0, 202.0]), normal_force, TEST_SETTINGS
    )
    assert np.array_equal(first.amplitude, second.amplitude)


def test_markov_tape_and_hard_preload_are_reproducible() -> None:
    first = uniform_bank(4, 20260820, TEST_SETTINGS)
    second = uniform_bank(4, 20260820, TEST_SETTINGS)
    assert np.array_equal(first, second)
    result = evaluate_markov(
        np.zeros(5), np.array([200.0]), first, TEST_SETTINGS
    )
    assert set(np.unique(result.preload)) == {30.0, 50.0}
    assert np.all(np.isfinite(result.amplitudes))


def test_hard_markov_direct_ad_is_zero_and_crn_fd_is_nonzero() -> None:
    omegas = np.array([198.0, 202.0, 206.0])
    uniforms = uniform_bank(4, 20260820, TEST_SETTINGS)
    coefficients = np.zeros(5)
    _, direct_gradient = direct_ad_objective_and_gradient(
        coefficients, omegas, uniforms, TEST_SETTINGS
    )
    first = crn_centered_finite_difference(
        coefficients, omegas, uniforms, settings=TEST_SETTINGS
    )
    second = crn_centered_finite_difference(
        coefficients, omegas, uniforms, settings=TEST_SETTINGS
    )
    assert np.max(np.abs(direct_gradient)) <= 1e-12
    assert np.all(np.isfinite(first.gradient))
    assert np.linalg.norm(first.gradient) > 0.0
    assert np.array_equal(first.gradient, second.gradient)
    assert np.any(first.mode_difference_counts > 0)


def test_zero_gain_state_aware_matches_periodic_markov() -> None:
    omegas = np.array([198.0, 202.0])
    uniforms = uniform_bank(2, 20260820, TEST_SETTINGS)
    periodic = evaluate_markov(
        PHASE2_COEFFICIENTS, omegas, uniforms, TEST_SETTINGS
    )
    state_aware = evaluate_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas,
        uniforms,
        TEST_SETTINGS,
    )
    assert np.allclose(
        state_aware.amplitudes,
        periodic.amplitudes,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.isclose(
        state_aware.objective,
        periodic.objective,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.array_equal(
        state_aware.transition_counts,
        np.broadcast_to(
            periodic.transition_counts,
            state_aware.transition_counts.shape,
        ),
    )
    replay = replay_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas[0],
        uniforms[0],
        TEST_SETTINGS,
    )
    assert np.array_equal(replay.modes, periodic.modes[0])
    assert np.array_equal(replay.preload, periodic.preload[0])


def test_state_aware_hard_gradient_is_real_and_reproducible() -> None:
    omegas = np.array([198.0, 202.0, 206.0])
    uniforms = uniform_bank(4, 20260820, TEST_SETTINGS)
    _, direct_gradient = direct_ad_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas,
        uniforms,
        TEST_SETTINGS,
    )
    first = crn_fd_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas,
        uniforms,
        settings=TEST_SETTINGS,
    )
    second = crn_fd_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas,
        uniforms,
        settings=TEST_SETTINGS,
    )
    assert np.max(np.abs(direct_gradient)) <= 1e-12
    assert np.all(np.isfinite(first.gradient))
    assert np.linalg.norm(first.gradient) > 0.0
    assert np.array_equal(first.gradient, second.gradient)
    assert np.any(np.abs(first.gradient[5:]) > 0.0)
    assert np.any(first.mode_difference_counts[5:] > 0)
    replay = replay_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        omegas[1],
        uniforms[0],
        TEST_SETTINGS,
    )
    assert set(np.unique(replay.preload)) == {30.0, 50.0}


def test_state_aware_score_uses_only_previous_mechanics_state() -> None:
    uniforms = uniform_bank(1, 20260820, TEST_SETTINGS)[0]
    coefficients = INITIAL_STATE_AWARE_COEFFICIENTS.copy()
    coefficients[5:] = [0.3, -0.2]
    replay = replay_state_aware(
        coefficients, 202.0, uniforms, TEST_SETTINGS
    )
    basis = phase_basis(TEST_SETTINGS)
    expected = basis @ coefficients[:5]
    expected[1:] += (
        coefficients[5] * replay.velocity[:-1] / 0.48
        + coefficients[6] * replay.displacement[:-1] / 0.0024
    )
    assert np.allclose(replay.score, expected, rtol=1e-12, atol=1e-14)
