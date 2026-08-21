import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.engineering_showcase import SYSTEM
from stochastic_stick_slip.model import (
    NUM_STEPS,
    build_mechanics_batch_simulator,
    build_variable_time_step_mechanics_batch_simulator,
)
from stochastic_stick_slip.wu_v2 import (
    FORCING_AMPLITUDE,
    PHASES,
    constant_preload,
    cycle_amplitudes,
    excitation_grid,
    harmonic_preload,
    single_tone_forcing,
    steady_state_metrics,
)


def test_variable_time_step_matches_locked_mechanics() -> None:
    fixed = build_mechanics_batch_simulator(SYSTEM)
    variable = build_variable_time_step_mechanics_batch_simulator(SYSTEM)
    times = np.asarray(SYSTEM.times)
    forcing = (FORCING_AMPLITUDE * np.sin(SYSTEM.omega_1 * times))[None, :]
    preload = constant_preload(0.04)
    fixed_outputs = fixed(0.10, jnp.asarray(forcing), jnp.asarray(preload))
    variable_outputs = variable(
        0.10,
        jnp.asarray(forcing),
        jnp.asarray(preload),
        jnp.asarray(SYSTEM.time_step),
    )
    for index, (candidate, reference) in enumerate(
        zip(variable_outputs, fixed_outputs, strict=True)
    ):
        candidate_array = np.asarray(candidate)
        reference_array = np.asarray(reference)
        if index >= 2:
            assert np.array_equal(candidate_array, reference_array)
        else:
            assert np.allclose(
                candidate_array,
                reference_array,
                rtol=1e-10,
                atol=1e-11,
            )


def test_single_tone_and_harmonic_preload_are_frozen() -> None:
    omega = 1.23 * SYSTEM.omega_1
    time_step, times = excitation_grid(omega)
    returned_time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega
    )
    assert returned_time_step == time_step
    assert forcing.shape == (NUM_STEPS,)
    assert np.allclose(forcing, FORCING_AMPLITUDE * np.sin(omega * times))

    preload = harmonic_preload(0.04, omega, 2, PHASES[:2])
    assert preload.shape == (2, NUM_STEPS, 2)
    assert np.array_equal(preload[:, :, 0], preload[:, :, 1])
    assert np.min(preload) >= 0.03 - 1e-14
    assert np.max(preload) <= 0.05 + 1e-14
    assert np.allclose(np.mean(preload, axis=1), 0.04, atol=1e-14)


def test_registered_steady_state_metric_uses_cycles_5_to_8() -> None:
    requested = np.arange(1.0, 9.0)
    cycles = []
    for amplitude in requested:
        cycle = np.zeros(100)
        cycle[0] = -amplitude
        cycle[1] = amplitude
        cycles.append(cycle)
    displacement = np.concatenate(cycles)[None, :]

    amplitudes = cycle_amplitudes(displacement)
    objective, convergence, returned = steady_state_metrics(displacement)
    assert np.array_equal(amplitudes[0], requested)
    assert np.array_equal(returned, amplitudes)
    assert objective[0] == np.mean([5.0, 6.0, 7.0, 8.0])
    assert convergence[0] == abs(5.5 - 7.5) / 7.5
