import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.engineering_showcase import SYSTEM
from stochastic_stick_slip.model import (
    build_variable_time_step_mechanics_batch_simulator,
)
from stochastic_stick_slip.wu2019_reproduction import (
    AMPLITUDE_RATIOS,
    CASES_PER_HARMONIC,
    FAST_PARAMETER_NAMES,
    FAST_SEED,
    HARMONIC_PHASES,
    TOTAL_HARMONIC_CASES,
    discrete_true_intervals,
    fast_parameter_samples,
    fourier_preload,
    positive_period_energy,
    select_budget_fast_n,
    single_harmonic_grid,
    wu_reference_table,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    excitation_grid,
    single_tone_forcing,
)


def test_fast_shape_seed_and_parameter_order() -> None:
    first = fast_parameter_samples(68)
    repeated = fast_parameter_samples(68)
    assert first.shape == (8 * 68, 8)
    assert np.array_equal(first, repeated)
    assert FAST_PARAMETER_NAMES == (
        "A1",
        "Phi1",
        "A2",
        "Phi2",
        "A3",
        "Phi3",
        "A4",
        "Phi4",
    )
    assert FAST_SEED == 20260821


def test_fourier_preload_is_shared_and_matches_the_formula() -> None:
    omega = 1.17 * SYSTEM.omega_1
    amplitudes = np.asarray([[0.01, 0.01, 0.01, 0.01]])
    phases = np.asarray([[0.0, 0.4, 0.8, 1.2]])
    preload = fourier_preload(omega, amplitudes, phases)
    _, times = excitation_grid(omega, DIAGNOSTIC_NUM_PERIODS)
    expected = 0.04 + sum(
        amplitudes[0, harmonic - 1]
        * np.sin(
            harmonic * omega * times + phases[0, harmonic - 1]
        )
        for harmonic in (1, 2, 3, 4)
    )
    assert preload.shape == (1, 2400, 2)
    assert np.allclose(preload[0, :, 0], expected)
    assert np.array_equal(preload[:, :, 0], preload[:, :, 1])
    assert np.min(preload) >= -1e-12


def test_harmonic_grid_has_the_registered_size_and_order() -> None:
    assert np.array_equal(AMPLITUDE_RATIOS, np.linspace(0.0, 0.25, 11))
    assert np.array_equal(
        HARMONIC_PHASES, 2.0 * np.pi * np.arange(64) / 64.0
    )
    amplitudes, phases, ratios, phase_values = single_harmonic_grid(2)
    assert CASES_PER_HARMONIC == 641
    assert TOTAL_HARMONIC_CASES == 2564
    assert amplitudes.shape == phases.shape == (641, 4)
    assert ratios[0] == 0.0
    assert phase_values[0] == 0.0
    assert np.count_nonzero(amplitudes[0]) == 0
    assert np.allclose(ratios[1:65], AMPLITUDE_RATIOS[1])
    assert np.array_equal(phase_values[1:65], HARMONIC_PHASES)


def test_local_control_tracks_current_frequency_at_fixed_phase() -> None:
    amplitude = 0.007
    phase = 4.4
    for omega in (0.9 * SYSTEM.omega_1, 1.1 * SYSTEM.omega_1):
        amplitudes = np.zeros((1, 4))
        phases = np.zeros_like(amplitudes)
        amplitudes[0, 1] = amplitude
        phases[0, 1] = phase
        preload = fourier_preload(omega, amplitudes, phases)
        _, times = excitation_grid(omega, DIAGNOSTIC_NUM_PERIODS)
        expected = 0.04 + amplitude * np.sin(2.0 * omega * times + phase)
        assert np.allclose(preload[0, :, 0], expected)


def test_positive_energy_uses_cycles_21_to_24() -> None:
    per_cycle = np.arange(1.0, 25.0)
    work = np.zeros((2, 2400, 2))
    for cycle, value in enumerate(per_cycle):
        work[:, cycle * 100, 0] = value
        work[:, cycle * 100, 1] = 2.0 * value
    mean_energy, cycles = positive_period_energy(work)
    assert cycles.shape == (2, 24)
    assert np.array_equal(cycles[0], 3.0 * per_cycle)
    assert np.all(mean_energy == 3.0 * np.mean([21.0, 22.0, 23.0, 24.0]))


def test_energy_diagnostic_does_not_change_legacy_mechanics_outputs() -> None:
    legacy = build_variable_time_step_mechanics_batch_simulator(SYSTEM)
    diagnostic = build_variable_time_step_mechanics_batch_simulator(
        SYSTEM, return_friction_work=True
    )
    omega = SYSTEM.omega_1
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
    )
    forcing = forcing[None, :]
    preload = np.full((1, forcing.shape[1], 2), 0.04)
    legacy_outputs = legacy(
        DAMPING,
        jnp.asarray(forcing),
        jnp.asarray(preload),
        jnp.asarray(time_step),
    )
    diagnostic_outputs = diagnostic(
        DAMPING,
        jnp.asarray(forcing),
        jnp.asarray(preload),
        jnp.asarray(time_step),
    )
    assert len(diagnostic_outputs) == len(legacy_outputs) + 1
    for reference, candidate in zip(
        legacy_outputs, diagnostic_outputs[:5], strict=True
    ):
        assert np.array_equal(np.asarray(reference), np.asarray(candidate))
    friction_work = np.asarray(diagnostic_outputs[-1])
    assert friction_work.shape == (1, 2400, 2)
    assert np.all(np.isfinite(friction_work))
    assert np.min(friction_work) >= 0.0


def test_runtime_budget_selects_maximum_or_stops_at_minimum() -> None:
    fast_timing = {
        key: {
            "first_call_seconds": 0.01,
            "median_steady_seconds": 0.01,
        }
        for key in ("batch_32", "batch_5", "batch_2", "batch_4_energy")
    }
    selected, _ = select_budget_fast_n(fast_timing)
    assert selected == 1000

    slow_timing = {
        key: {
            "first_call_seconds": 20.0,
            "median_steady_seconds": 20.0,
        }
        for key in ("batch_32", "batch_5", "batch_2", "batch_4_energy")
    }
    selected, estimate = select_budget_fast_n(slow_timing)
    assert selected is None
    assert estimate["with_margin_seconds"] > 1800.0


def test_discrete_intervals_do_not_bridge_gaps() -> None:
    ratios = np.asarray([0.6, 0.7, 0.8, 0.9, 1.0])
    mask = np.asarray([True, True, False, True, False])
    assert discrete_true_intervals(ratios, mask) == [[0.6, 0.7], [0.9, 0.9]]


def test_wu_reference_table_has_traceable_sections() -> None:
    references = wu_reference_table()
    for key in (
        "passive",
        "fast",
        "one_omega",
        "two_omega",
        "three_omega",
        "four_omega",
        "excitation",
        "energy",
    ):
        assert "source" in references[key] or "source_indices" in references[key]
