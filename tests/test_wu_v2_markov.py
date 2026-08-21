import inspect

import jax.numpy as jnp
import numpy as np

import stochastic_stick_slip.wu_v2_markov as markov
from stochastic_stick_slip.model import STEPS_PER_PERIOD
from stochastic_stick_slip.wu_v2 import (
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    SYSTEM,
    single_tone_forcing,
)


def test_neutral_rates_and_hard_preloads_are_frozen() -> None:
    omega = 1.19 * SYSTEM.omega_1
    time_step, times = (
        2.0 * np.pi / (STEPS_PER_PERIOD * omega),
        np.asarray([0.1, 0.2]),
    )
    rate_lh, rate_hl = markov.transition_rates(
        jnp.zeros(2), jnp.asarray(times), jnp.asarray(omega)
    )
    assert np.array_equal(np.asarray(rate_lh), np.asarray(rate_hl))

    uniforms = jnp.asarray(
        [[[[0.25, 0.75], [0.99, 0.99], [0.99, 0.99]]]],
        dtype=jnp.float64,
    )
    modes, preload, _, _ = markov.generate_hard_preload_history(
        jnp.zeros(2),
        jnp.asarray(times),
        uniforms,
        jnp.asarray(omega),
        jnp.asarray(time_step),
    )
    assert np.array_equal(np.asarray(modes[0, 0, :, 0]), [True, True])
    assert np.array_equal(np.asarray(modes[0, 0, :, 1]), [False, False])
    assert set(np.unique(np.asarray(preload))) == {
        markov.PRELOAD_LOW,
        markov.PRELOAD_HIGH,
    }


def test_fixed_tapes_are_reproducible_and_contacts_are_independent() -> None:
    first = markov.markov_uniform_bank()
    repeated = markov.markov_uniform_bank()
    assert first.shape == (8, 4, 2401, 2)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first[..., 0], first[..., 1])
    assert np.any((first[..., 0, 0] < 0.5) != (first[..., 0, 1] < 0.5))


def test_wu_objective_uses_cycles_21_to_24() -> None:
    requested = np.arange(1.0, DIAGNOSTIC_NUM_PERIODS + 1.0)
    cycles = []
    for amplitude in requested:
        cycle = np.zeros(STEPS_PER_PERIOD)
        cycle[0] = -amplitude
        cycle[1] = amplitude
        cycles.append(cycle)
    displacement = jnp.asarray(np.concatenate(cycles)[None, :])
    objective = np.asarray(markov.steady_state_amplitude(displacement))
    assert objective[0] == np.mean([21.0, 22.0, 23.0, 24.0])


def test_crn_fd_reuses_the_same_tape(monkeypatch) -> None:
    uniforms = markov.markov_uniform_bank()[:1, :1]
    captured = []

    def fake_evaluate(q, forcing, received, times, omega, time_step):
        del forcing, times, omega, time_step
        captured.append(np.asarray(received).copy())
        q = np.asarray(q)
        return {
            "trajectory_objectives": np.asarray([[q[0] + 2.0 * q[1]]]),
            "modes": np.zeros((1, 1, 2, 2), dtype=bool),
        }

    monkeypatch.setattr(markov, "evaluate_markov_bank", fake_evaluate)
    result = markov.crn_centered_fd(
        np.zeros(2), np.zeros(2), uniforms, np.zeros(2), 1.0, 0.01
    )
    assert np.allclose(result["gradient"], [1.0, 2.0])
    assert len(captured) == 4
    assert all(np.array_equal(received, uniforms) for received in captured)


def test_reduced_bank_direct_ad_and_crn_fd_are_finite() -> None:
    omega = 1.19 * SYSTEM.omega_1
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
    )
    times = time_step * np.arange(1, markov.NUM_STEPS + 1)
    uniforms = markov.markov_uniform_bank()[:1, :1]
    evaluation = markov.evaluate_markov_bank(
        np.zeros(2), forcing, uniforms, times, omega, time_step
    )
    objective, gradient = markov.direct_ad_objective_and_gradient(
        np.zeros(2), forcing, uniforms, times, omega, time_step
    )
    fd = markov.crn_centered_fd(
        np.zeros(2), forcing, uniforms, times, omega, time_step
    )
    assert np.isfinite(objective)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) <= 1e-12
    assert np.all(np.isfinite(fd["gradient"]))

    parameters = list(inspect.signature(markov.fixed_history_objective).parameters)
    assert parameters == ["forcing", "preload", "time_step"]
    preload = np.asarray(evaluation["preload"])[0, 0]
    replays = [
        markov.fixed_history_objective(forcing, preload, time_step)
        for _ in (
            np.zeros(2),
            np.asarray([0.02, 0.0]),
            np.asarray([-0.02, 0.0]),
            np.asarray([0.0, 0.02]),
            np.asarray([0.0, -0.02]),
        )
    ]
    assert np.allclose(replays, replays[0], rtol=1e-12, atol=1e-14)
