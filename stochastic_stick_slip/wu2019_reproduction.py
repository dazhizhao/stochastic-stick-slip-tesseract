"""Frozen helpers for the Wu2019 design-method reproduction."""

from __future__ import annotations

import math

import numpy as np
from SALib.sample import fast_sampler

from stochastic_stick_slip.model import STEPS_PER_PERIOD
from stochastic_stick_slip.wu_v2 import (
    DIAGNOSTIC_NUM_PERIODS,
    REFERENCE_PRELOAD,
    excitation_grid,
)


FAST_PARAMETER_NAMES = (
    "A1",
    "Phi1",
    "A2",
    "Phi2",
    "A3",
    "Phi3",
    "A4",
    "Phi4",
)
FAST_BOUNDS = tuple(
    [0.0, 0.25 * REFERENCE_PRELOAD]
    if name.startswith("A")
    else [0.0, 2.0 * np.pi]
    for name in FAST_PARAMETER_NAMES
)
FAST_SEED = 20260821
FAST_INTERFERENCE = 4
FAST_MAX_N = 1000
FAST_MIN_N = 68
FAST_VARIABLES = len(FAST_PARAMETER_NAMES)
RUNTIME_BUDGET_SECONDS = 30.0 * 60.0
RUNTIME_MARGIN = 1.15

AMPLITUDE_RATIOS = np.linspace(0.0, 0.25, 11)
HARMONIC_PHASES = 2.0 * np.pi * np.arange(64, dtype=np.float64) / 64.0
HARMONICS = (1, 2, 3, 4)
CASES_PER_HARMONIC = 1 + (len(AMPLITUDE_RATIOS) - 1) * len(
    HARMONIC_PHASES
)
TOTAL_HARMONIC_CASES = len(HARMONICS) * CASES_PER_HARMONIC
LOCAL_FRF_RATIOS = np.linspace(0.90, 1.10, 21)
WORKING_FORCE_RATIOS = np.linspace(0.60, 1.80, 13)
WORKING_FRF_RATIOS = np.linspace(0.90, 1.10, 11)


def wu_reference_table() -> dict:
    """Return the small paper reference table with journal-page provenance."""
    return {
        "paper": {
            "citation": (
                "Wu et al., Journal of Sound and Vibration 459 (2019) 114850"
            ),
            "scope": "numerical SDOF results, not the experimental reduction",
        },
        "passive": {
            "normal_force_N": 40.0,
            "source": "Section 5.2, journal page 7; Fig. 4",
        },
        "fast": {
            "samples": 8000,
            "samples_per_parameter": 1000,
            "source": "Section 5.2, journal page 7",
            "figure_9_output": "maximum response over the frequency band",
            "figure_9_first_order": {
                "A1": 1.6e-4,
                "Phi1": 4.6e-4,
                "A2": 4.8e-2,
                "Phi2": 9.8e-4,
                "A3": 1.8e-2,
                "Phi3": 1.9e-3,
                "A4": 8.4e-4,
                "Phi4": 3.3e-3,
            },
            "figure_9_total_order": {
                "A1": 4.2e-2,
                "Phi1": 3.2e-2,
                "A2": 8.1e-1,
                "Phi2": 8.0e-1,
                "A3": 6.2e-2,
                "Phi3": 5.2e-2,
                "A4": 2.1e-1,
                "Phi4": 2.1e-1,
            },
            "source_indices": "Fig. 9, journal page 10",
        },
        "one_omega": {
            "result": "negligible improvement",
            "source": "Fig. 11 and Section 5.3, journal pages 11-12",
        },
        "two_omega": {
            "amplitude_N": 10.0,
            "amplitude_ratio": 0.25,
            "phase_rad": 4.4,
            "peak_reduction_percent": 21.0,
            "source": "Fig. 10 and Section 5.3, journal page 11",
        },
        "three_omega": {
            "result": "weak by FAST; no separate numerical optimum reported",
            "source": "Figs. 8-9, journal pages 9-10",
        },
        "four_omega": {
            "amplitude_N": 5.0,
            "amplitude_ratio": 0.125,
            "phase_rad": 5.0,
            "peak_reduction_percent": 9.0,
            "source": "Fig. 10 and Section 5.3, journal page 11",
        },
        "excitation": {
            "design_force_N": 10.0,
            "working_interval_N": [6.4, 17.8],
            "normalized_interval": [0.64, 1.78],
            "source": "Fig. 12 and Section 5.3, journal page 12",
        },
        "energy": {
            "result": "only even harmonics add period-averaged dissipation",
            "source": "Eqs. 20-27, journal pages 12-13",
        },
    }


def fast_problem() -> dict:
    return {
        "num_vars": FAST_VARIABLES,
        "names": list(FAST_PARAMETER_NAMES),
        "bounds": [list(bounds) for bounds in FAST_BOUNDS],
    }


def fast_parameter_samples(n: int) -> np.ndarray:
    if n < FAST_MIN_N:
        raise ValueError(f"FAST N must be at least {FAST_MIN_N}")
    return np.asarray(
        fast_sampler.sample(
            fast_problem(), n, M=FAST_INTERFERENCE, seed=FAST_SEED
        ),
        dtype=np.float64,
    )


def split_fast_parameters(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FAST_VARIABLES:
        raise ValueError("FAST samples must have shape (samples, 8)")
    return values[:, 0::2], values[:, 1::2]


def fourier_preload(
    omega: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    num_periods: int = DIAGNOSTIC_NUM_PERIODS,
) -> np.ndarray:
    """Build the shared four-harmonic normal-force command."""
    amplitude_values = np.asarray(amplitudes, dtype=np.float64)
    phase_values = np.asarray(phases, dtype=np.float64)
    if (
        amplitude_values.ndim != 2
        or amplitude_values.shape[1] != 4
        or phase_values.shape != amplitude_values.shape
    ):
        raise ValueError("amplitudes and phases must both have shape (batch, 4)")
    if np.any(amplitude_values < 0.0) or np.any(
        amplitude_values > 0.25 * REFERENCE_PRELOAD + 1e-15
    ):
        raise ValueError("harmonic amplitudes are outside the frozen bounds")
    _, times = excitation_grid(omega, num_periods)
    orders = np.arange(1.0, 5.0, dtype=np.float64)
    arguments = (
        orders[None, :, None] * float(omega) * times[None, None, :]
        + phase_values[:, :, None]
    )
    scalar = REFERENCE_PRELOAD + np.sum(
        amplitude_values[:, :, None] * np.sin(arguments), axis=1
    )
    if np.min(scalar) < -1e-12:
        raise FloatingPointError("Fourier preload became negative")
    return np.repeat(scalar[:, :, None], 2, axis=2)


def single_harmonic_grid(
    harmonic: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if harmonic not in HARMONICS:
        raise ValueError("harmonic must be one of 1, 2, 3, 4")
    ratios = [0.0]
    phases = [0.0]
    for ratio in AMPLITUDE_RATIOS[1:]:
        for phase in HARMONIC_PHASES:
            ratios.append(float(ratio))
            phases.append(float(phase))
    ratios_array = np.asarray(ratios, dtype=np.float64)
    phases_array = np.asarray(phases, dtype=np.float64)
    amplitudes = np.zeros((CASES_PER_HARMONIC, 4), dtype=np.float64)
    phase_matrix = np.zeros_like(amplitudes)
    amplitudes[:, harmonic - 1] = ratios_array * REFERENCE_PRELOAD
    phase_matrix[:, harmonic - 1] = phases_array
    return amplitudes, phase_matrix, ratios_array, phases_array


def positive_period_energy(
    dissipated_work: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cycles 21-24 mean energy and all per-cycle energies."""
    values = np.asarray(dissipated_work, dtype=np.float64)
    if values.shape[-2:] != (DIAGNOSTIC_NUM_PERIODS * STEPS_PER_PERIOD, 2):
        raise ValueError("dissipated work must have shape (..., 2400, 2)")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise FloatingPointError("dissipated work must be finite and non-negative")
    cycles = values.reshape(
        values.shape[:-2]
        + (DIAGNOSTIC_NUM_PERIODS, STEPS_PER_PERIOD, 2)
    ).sum(axis=(-1, -2))
    return np.mean(cycles[..., 20:24], axis=-1), cycles


def runtime_call_counts(fast_n: int) -> dict[str, int]:
    return {
        "batch_32": math.ceil(FAST_VARIABLES * fast_n / 32)
        + math.ceil(TOTAL_HARMONIC_CASES / 32),
        "batch_5": len(LOCAL_FRF_RATIOS),
        "batch_2": len(WORKING_FORCE_RATIOS) * len(WORKING_FRF_RATIOS),
        "batch_4_energy": 1,
    }


def projected_runtime_seconds(timing: dict, fast_n: int) -> dict:
    counts = runtime_call_counts(fast_n)
    raw = 0.0
    for key, count in counts.items():
        first = float(timing[key]["first_call_seconds"])
        median = float(timing[key]["median_steady_seconds"])
        raw += first + max(count - 1, 0) * median
    return {
        "call_counts": counts,
        "without_margin_seconds": raw,
        "with_margin_seconds": RUNTIME_MARGIN * raw,
    }


def select_budget_fast_n(timing: dict) -> tuple[int | None, dict]:
    for n in range(FAST_MAX_N, FAST_MIN_N - 1, -1):
        if n % 4:
            continue
        estimate = projected_runtime_seconds(timing, n)
        if estimate["with_margin_seconds"] <= RUNTIME_BUDGET_SECONDS:
            return n, estimate
    return None, projected_runtime_seconds(timing, FAST_MIN_N)


def discrete_true_intervals(
    ratios: np.ndarray, qualifying: np.ndarray
) -> list[list[float]]:
    ratio_values = np.asarray(ratios, dtype=np.float64)
    mask = np.asarray(qualifying, dtype=np.bool_)
    if ratio_values.ndim != 1 or mask.shape != ratio_values.shape:
        raise ValueError("ratios and qualifying must be matching vectors")
    intervals: list[list[float]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            intervals.append(
                [float(ratio_values[start]), float(ratio_values[end])]
            )
            start = None
    return intervals
