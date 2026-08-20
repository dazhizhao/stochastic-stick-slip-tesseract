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
    uniform_bank,
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
