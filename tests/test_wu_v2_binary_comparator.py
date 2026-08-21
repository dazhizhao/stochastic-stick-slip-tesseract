import numpy as np

from scripts.run_wu_v2_binary_comparator import (
    BINARY_PHASES,
    EXPECTED_BINARY_PHASE,
    coefficient_phase_to_binary_phase,
    load_frozen_inputs,
    select_phase_by_local_peak,
)
from stochastic_stick_slip.wu_v2 import (
    DIAGNOSTIC_NUM_PERIODS,
    SYSTEM,
    excitation_grid,
)
from stochastic_stick_slip.wu_v2_markov import (
    PRELOAD_HIGH,
    PRELOAD_LOW,
    deterministic_binary_preload,
    deterministic_policy_hard_limit_preload,
)


def test_binary_phase_grid_is_exact() -> None:
    expected = 2.0 * np.pi * np.arange(64, dtype=np.float64) / 64.0
    assert np.array_equal(BINARY_PHASES, expected)


def test_vectorized_binary_preload_is_two_state_and_shared() -> None:
    omega = 1.13 * SYSTEM.omega_1
    phases = BINARY_PHASES[[0, 17, 46]]
    preload = deterministic_binary_preload(omega, phases)
    assert preload.shape == (3, 2400, 2)
    assert set(np.unique(preload)) == {PRELOAD_LOW, PRELOAD_HIGH}
    assert np.array_equal(preload[:, :, 0], preload[:, :, 1])
    scalar = deterministic_binary_preload(omega, phases[0])
    assert np.array_equal(scalar, preload[:1])


def test_binary_local_command_tracks_frequency_at_fixed_phase() -> None:
    phase = 0.41
    first_omega = 0.91 * SYSTEM.omega_1
    second_omega = 1.09 * SYSTEM.omega_1
    first = deterministic_binary_preload(first_omega, phase)
    second = deterministic_binary_preload(second_omega, phase)
    _, first_times = excitation_grid(first_omega, DIAGNOSTIC_NUM_PERIODS)
    _, second_times = excitation_grid(second_omega, DIAGNOSTIC_NUM_PERIODS)
    first_expected = np.where(
        np.sin(2.0 * first_omega * first_times + phase) >= 0.0,
        PRELOAD_HIGH,
        PRELOAD_LOW,
    )
    second_expected = np.where(
        np.sin(2.0 * second_omega * second_times + phase) >= 0.0,
        PRELOAD_HIGH,
        PRELOAD_LOW,
    )
    assert np.array_equal(first[0, :, 0], first_expected)
    assert np.array_equal(second[0, :, 0], second_expected)


def test_phase_selection_uses_local_peak_not_nominal_value() -> None:
    amplitudes = np.asarray(
        [
            [1.0, 2.0, 9.0],
            [4.0, 5.0, 4.0],
            [4.0, 5.0, 4.0],
        ]
    )
    best, local_peaks, peak_indices = select_phase_by_local_peak(amplitudes)
    assert int(np.argmin(amplitudes[:, 1])) == 0
    assert best == 1
    assert np.array_equal(local_peaks, [9.0, 5.0, 5.0])
    assert np.array_equal(peak_indices, [2, 1, 1])


def test_coefficient_phase_conversion_matches_direct_hard_limit() -> None:
    omega = 1.07 * SYSTEM.omega_1
    q_values = np.asarray(
        [
            [-3.514982271973227, -0.5337623522652377],
            [-10.665739565561044, -6.033414703985564],
        ]
    )
    coefficient_phases = np.mod(
        np.arctan2(q_values[:, 1], q_values[:, 0]), 2.0 * np.pi
    )
    binary_phases = np.asarray(
        [coefficient_phase_to_binary_phase(value) for value in coefficient_phases]
    )
    direct = deterministic_policy_hard_limit_preload(omega, q_values)
    converted = deterministic_binary_preload(omega, binary_phases)
    assert np.array_equal(direct, converted)
    assert np.allclose(
        binary_phases,
        [
            EXPECTED_BINARY_PHASE["stochastic_lr0p1"],
            EXPECTED_BINARY_PHASE["stochastic_lr1p0"],
        ],
        rtol=1e-12,
        atol=1e-14,
    )


def test_w2_loader_returns_frozen_references() -> None:
    frozen = load_frozen_inputs()
    assert frozen["frequency_ratios"].shape == (21,)
    assert frozen["passive_peak"] == 0.18748720511761083
    assert frozen["wu_continuous_peak"] == 0.1495599768466055
    assert frozen["wu_binary_peak"] == 0.14913023166130196
    assert np.array_equal(
        frozen["candidates"]["stochastic_lr1p0"]["q"],
        [-10.665739565561044, -6.033414703985564],
    )
