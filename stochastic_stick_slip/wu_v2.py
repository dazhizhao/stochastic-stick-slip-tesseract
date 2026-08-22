"""Frozen Wu-style deterministic benchmark on the 32x4 Jenkins FEM."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.showcase import SYSTEM
from stochastic_stick_slip.model import (
    NUM_PERIODS,
    STEPS_PER_PERIOD,
    build_variable_time_step_mechanics_batch_simulator,
)


DAMPING = 0.10
FORCING_AMPLITUDE = 0.02
REFERENCE_PRELOAD = 0.04
PRELOAD_VALUES = np.arange(0.025, 0.055 + 1e-12, 0.005)
REPAIR_NUM_PERIODS = 16
REPAIR_PRELOAD_VALUES = np.arange(0.0, 0.060 + 1e-12, 0.005)
DIAGNOSTIC_NUM_PERIODS = 24
DIAGNOSTIC_PRELOAD_VALUES = np.arange(0.0, 0.060 + 1e-12, 0.010)
FINAL_REFINEMENT_RATIOS = np.linspace(1.150, 1.200, 11)
LOCAL_FRF_RATIOS = np.linspace(0.95, 1.05, 11)
COARSE_FREQUENCY_RATIOS = np.linspace(0.80, 1.60, 33)
FINE_FREQUENCY_HALF_WIDTH = 0.025
FINE_FREQUENCY_POINTS = 11
PHASES = 2.0 * np.pi * np.arange(32, dtype=np.float64) / 32.0
STEADY_STATE_TOLERANCE = 0.02
MINIMUM_PASSIVE_REDUCTION_PERCENT = 5.0
MINIMUM_ADDITIONAL_REDUCTION_POINTS = 2.0

MECHANICS_SIMULATOR = build_variable_time_step_mechanics_batch_simulator(
    SYSTEM
)


def excitation_grid(
    omega: float,
    num_periods: int = NUM_PERIODS,
) -> tuple[float, np.ndarray]:
    """Return the step and endpoint times for complete forcing cycles."""
    time_step = 2.0 * np.pi / (float(omega) * STEPS_PER_PERIOD)
    num_steps = int(num_periods) * STEPS_PER_PERIOD
    times = time_step * np.arange(1, num_steps + 1, dtype=np.float64)
    return time_step, times


def single_tone_forcing(
    amplitude: float,
    omega: float,
    num_periods: int = NUM_PERIODS,
) -> tuple[float, np.ndarray]:
    """Return F sin(omega t) on a frequency-specific cycle grid."""
    time_step, times = excitation_grid(omega, num_periods)
    return time_step, float(amplitude) * np.sin(float(omega) * times)


def constant_preload(
    value: float,
    batch_size: int = 1,
    num_periods: int = NUM_PERIODS,
) -> np.ndarray:
    """Return a shared constant command for both friction contacts."""
    num_steps = int(num_periods) * STEPS_PER_PERIOD
    return np.full(
        (batch_size, num_steps, 2), float(value), dtype=np.float64
    )


def harmonic_preload(
    optimum_preload: float,
    omega: float,
    harmonic: int,
    phases: np.ndarray,
    num_periods: int = NUM_PERIODS,
) -> np.ndarray:
    """Return the same zero-mean harmonic preload command at both contacts."""
    _, times = excitation_grid(omega, num_periods)
    phase_values = np.asarray(phases, dtype=np.float64)
    scalar = float(optimum_preload) * (
        1.0
        + 0.25
        * np.sin(
            harmonic * float(omega) * times[None, :]
            + phase_values[:, None]
        )
    )
    return np.repeat(scalar[:, :, None], 2, axis=2)


def simulate_preload_bank(
    omega: float,
    preload: np.ndarray,
    forcing_amplitude: float = FORCING_AMPLITUDE,
):
    """Run a bank sharing one single-tone forcing condition."""
    preload_array = np.asarray(preload, dtype=np.float64)
    if (
        preload_array.ndim != 3
        or preload_array.shape[2] != 2
        or preload_array.shape[1] % STEPS_PER_PERIOD != 0
    ):
        raise ValueError("preload must have shape (batch, periods*100, 2)")
    num_steps = preload_array.shape[1]
    num_periods = num_steps // STEPS_PER_PERIOD
    time_step, forcing = single_tone_forcing(
        forcing_amplitude, omega, num_periods
    )
    forcing_bank = np.broadcast_to(
        forcing, (preload_array.shape[0], num_steps)
    )
    return MECHANICS_SIMULATOR(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        jnp.asarray(forcing_bank, dtype=jnp.float64),
        jnp.asarray(preload_array, dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )


def cycle_amplitudes(displacement: np.ndarray | jax.Array) -> np.ndarray:
    """Return (max-min)/2 for each complete excitation cycle."""
    values = np.asarray(displacement, dtype=np.float64)
    if values.shape[-1] % STEPS_PER_PERIOD != 0:
        raise ValueError("displacement history must contain complete cycles")
    num_periods = values.shape[-1] // STEPS_PER_PERIOD
    cycles = values.reshape(
        values.shape[:-1] + (num_periods, STEPS_PER_PERIOD)
    )
    return 0.5 * (np.max(cycles, axis=-1) - np.min(cycles, axis=-1))


def steady_state_metrics(
    displacement: np.ndarray | jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A_ss and the registered cycles 5-6 versus 7-8 convergence."""
    amplitudes = cycle_amplitudes(displacement)
    first_window = np.mean(amplitudes[..., 4:6], axis=-1)
    second_window = np.mean(amplitudes[..., 6:8], axis=-1)
    objective = np.mean(amplitudes[..., 4:8], axis=-1)
    convergence = np.abs(first_window - second_window) / np.maximum(
        np.abs(second_window), 1e-15
    )
    return objective, convergence, amplitudes


def repair_steady_state_metrics(
    displacement: np.ndarray | jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the registered 16-cycle repair objective and steady error."""
    amplitudes = cycle_amplitudes(displacement)
    if amplitudes.shape[-1] != REPAIR_NUM_PERIODS:
        raise ValueError("Gate 0 repair requires exactly 16 cycles")
    previous_window = np.mean(amplitudes[..., 8:12], axis=-1)
    final_window = np.mean(amplitudes[..., 12:16], axis=-1)
    convergence = np.abs(final_window - previous_window) / np.maximum(
        np.abs(final_window), 1e-15
    )
    return final_window, convergence, amplitudes


def diagnostic_steady_state_metrics(
    displacement: np.ndarray | jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the registered 24-cycle diagnostic objective and steady error."""
    amplitudes = cycle_amplitudes(displacement)
    if amplitudes.shape[-1] != DIAGNOSTIC_NUM_PERIODS:
        raise ValueError("Passive FRF diagnosis requires exactly 24 cycles")
    previous_window = np.mean(amplitudes[..., 16:20], axis=-1)
    final_window = np.mean(amplitudes[..., 20:24], axis=-1)
    convergence = np.abs(final_window - previous_window) / np.maximum(
        np.abs(final_window), 1e-15
    )
    return final_window, convergence, amplitudes


def frf_peak_indices(steady_amplitudes: np.ndarray) -> np.ndarray:
    """Return the frequency-column maximum for each preload row."""
    values = np.asarray(steady_amplitudes, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("FRF amplitudes must have shape (preload, frequency)")
    return np.argmax(values, axis=1)
