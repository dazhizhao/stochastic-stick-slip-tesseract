"""Reproduce the Wu2019 design method on the frozen JAX-FEM benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import fast

from stochastic_stick_slip.model import (
    STEPS_PER_PERIOD,
    build_variable_time_step_mechanics_batch_simulator,
)
from stochastic_stick_slip.wu2019_reproduction import (
    AMPLITUDE_RATIOS,
    CASES_PER_HARMONIC,
    FAST_INTERFERENCE,
    FAST_MAX_N,
    FAST_PARAMETER_NAMES,
    FAST_SEED,
    HARMONICS,
    HARMONIC_PHASES,
    LOCAL_FRF_RATIOS,
    RUNTIME_BUDGET_SECONDS,
    TOTAL_HARMONIC_CASES,
    WORKING_FORCE_RATIOS,
    WORKING_FRF_RATIOS,
    discrete_true_intervals,
    fast_parameter_samples,
    fast_problem,
    fourier_preload,
    positive_period_energy,
    projected_runtime_seconds,
    select_budget_fast_n,
    single_harmonic_grid,
    split_fast_parameters,
    wu_reference_table,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
    constant_preload,
    diagnostic_steady_state_metrics,
    simulate_preload_bank,
    single_tone_forcing,
)


GATE_0_PATH = ROOT / "outputs/wu_v2_gate0_final/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/wu2019_reproduction"
RESULTS_PATH = OUTPUT_DIRECTORY / "scorecard.json"
MARKDOWN_PATH = OUTPUT_DIRECTORY / "scorecard.md"
SUMMARY_FIGURE_PATH = OUTPUT_DIRECTORY / "wu_method_summary.png"
RANGE_FIGURE_PATH = OUTPUT_DIRECTORY / "excitation_range.png"
STOCHASTIC_PATHS = (
    ROOT / "outputs/wu_v2_gate_bc_100_lr0p1/results.json",
    ROOT / "outputs/wu_v2_gate_bc_100_lr1/results.json",
)

ENERGY_SIMULATOR = build_variable_time_step_mechanics_batch_simulator(
    SYSTEM, return_friction_work=True
)

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#747B84"
WEAK_COLOR = "#B9BEC4"
ONE_COLOR = "#9A7F72"
TWO_COLOR = "#2F668B"
THREE_COLOR = "#A6A0A8"
FOUR_COLOR = "#765C88"
HARMONIC_COLORS = (ONE_COLOR, TWO_COLOR, THREE_COLOR, FOUR_COLOR)


def _load_frozen_gate0() -> dict:
    result = json.loads(GATE_0_PATH.read_text())
    configuration = result["configuration"]
    passive = result["passive"]
    valid = (
        result["gate_0"]["result"] == "PASS"
        and configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
        and configuration["steps_per_period"] == STEPS_PER_PERIOD
        and np.isclose(configuration["damping"], DAMPING)
        and np.isclose(
            configuration["forcing_amplitude"], FORCING_AMPLITUDE
        )
        and np.isclose(passive["preload"], REFERENCE_PRELOAD)
        and np.isclose(passive["omega_r_ratio"], 1.19)
        and np.isclose(
            passive["steady_amplitude"],
            0.18748720511761185,
            rtol=0.0,
            atol=1e-14,
        )
    )
    if not valid:
        raise RuntimeError("Final Gate 0 does not match the frozen W1 benchmark")
    return {
        "omega_r": float(passive["omega_r"]),
        "omega_r_ratio": float(passive["omega_r_ratio"]),
        "passive_amplitude": float(passive["steady_amplitude"]),
        "passive_steady_error": float(passive["steady_error"]),
    }


def _finite_outputs(outputs, context: str) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(output) for output in outputs)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"Non-finite mechanics output: {context}")
    return arrays


def _evaluate_preload(
    omega: float,
    preload: np.ndarray,
    forcing_amplitude: float = FORCING_AMPLITUDE,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _finite_outputs(
        simulate_preload_bank(omega, preload, forcing_amplitude),
        f"omega={omega}, forcing={forcing_amplitude}",
    )
    objective, steady_error, cycle_amplitudes = (
        diagnostic_steady_state_metrics(arrays[0])
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (objective, steady_error, cycle_amplitudes)
    ):
        raise FloatingPointError("Non-finite W1 objective")
    return objective, steady_error


def _pad_rows(values: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    actual = len(values)
    if actual == 0:
        raise ValueError("cannot pad an empty batch")
    if actual == batch_size:
        return values, actual
    padding = np.repeat(values[-1:], batch_size - actual, axis=0)
    return np.concatenate((values, padding), axis=0), actual


def _evaluate_controls(
    omega: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    objectives = []
    steady_errors = []
    for start in range(0, len(amplitudes), batch_size):
        amplitude_chunk, actual = _pad_rows(
            amplitudes[start : start + batch_size], batch_size
        )
        phase_chunk, _ = _pad_rows(
            phases[start : start + batch_size], batch_size
        )
        preload = fourier_preload(omega, amplitude_chunk, phase_chunk)
        objective, steady_error = _evaluate_preload(omega, preload)
        objectives.extend(np.asarray(objective[:actual]).tolist())
        steady_errors.extend(np.asarray(steady_error[:actual]).tolist())
    return np.asarray(objectives), np.asarray(steady_errors)


def _timed_forward(batch_size: int, energy: bool) -> dict:
    amplitudes = np.zeros((batch_size, 4), dtype=np.float64)
    phases = np.zeros_like(amplitudes)
    amplitudes[:, 1] = 0.005
    phases[:, 1] = 4.4
    preload = fourier_preload(1.19 * SYSTEM.omega_1, amplitudes, phases)
    omega = 1.19 * SYSTEM.omega_1

    def execute() -> None:
        if energy:
            time_step, forcing = single_tone_forcing(
                FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
            )
            forcing_bank = np.broadcast_to(
                forcing, (batch_size, forcing.size)
            )
            outputs = ENERGY_SIMULATOR(
                jnp.asarray(DAMPING, dtype=jnp.float64),
                jnp.asarray(forcing_bank, dtype=jnp.float64),
                jnp.asarray(preload, dtype=jnp.float64),
                jnp.asarray(time_step, dtype=jnp.float64),
            )
            np.asarray(outputs[-1])
        else:
            outputs = simulate_preload_bank(omega, preload)
            np.asarray(outputs[0])

    started = time.perf_counter()
    execute()
    first = time.perf_counter() - started
    repeats = []
    for _ in range(3):
        started = time.perf_counter()
        execute()
        repeats.append(time.perf_counter() - started)
    return {
        "batch_size": batch_size,
        "first_call_seconds": first,
        "steady_call_seconds": repeats,
        "median_steady_seconds": float(np.median(repeats)),
    }


def _estimate_runtime() -> dict:
    timing = {
        "batch_32": _timed_forward(32, energy=False),
        "batch_5": _timed_forward(5, energy=False),
        "batch_2": _timed_forward(2, energy=False),
        "batch_4_energy": _timed_forward(4, energy=True),
    }
    selected_n, selected_estimate = select_budget_fast_n(timing)
    return {
        "timing": timing,
        "budget_seconds": RUNTIME_BUDGET_SECONDS,
        "margin_factor": 1.15,
        "wu_scale_projection": projected_runtime_seconds(timing, FAST_MAX_N),
        "selected_fast_n": selected_n,
        "selected_projection": selected_estimate,
        "runtime_gate": "PASS" if selected_n is not None else "FAIL",
    }


def _run_fast(omega: float, fast_n: int) -> dict:
    samples = fast_parameter_samples(fast_n)
    expected_shape = (8 * fast_n, 8)
    if samples.shape != expected_shape:
        raise AssertionError(
            f"FAST sample shape {samples.shape} does not match {expected_shape}"
        )
    output = np.empty(len(samples), dtype=np.float64)
    for start in range(0, len(samples), 32):
        sample_chunk, actual = _pad_rows(samples[start : start + 32], 32)
        amplitudes, phases = split_fast_parameters(sample_chunk)
        preload = fourier_preload(omega, amplitudes, phases)
        objective, _ = _evaluate_preload(omega, preload)
        output[start : start + actual] = objective[:actual]
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("FAST objective contains non-finite values")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sensitivity = fast.analyze(
            fast_problem(),
            output,
            M=FAST_INTERFERENCE,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
            seed=FAST_SEED,
        )
    indices = {
        name: {
            "S1": float(sensitivity["S1"][index]),
            "ST": float(sensitivity["ST"][index]),
            "S1_conf": float(sensitivity["S1_conf"][index]),
            "ST_conf": float(sensitivity["ST_conf"][index]),
        }
        for index, name in enumerate(FAST_PARAMETER_NAMES)
    }
    ranking = sorted(
        FAST_PARAMETER_NAMES,
        key=lambda name: indices[name]["ST"],
        reverse=True,
    )
    return {
        "scope": "resonance-only FAST reproduction",
        "mode": (
            "Wu-scale resonance-only FAST reproduction"
            if fast_n == FAST_MAX_N
            else "budget-limited resonance-only FAST reproduction"
        ),
        "N_per_parameter": fast_n,
        "num_parameter_samples": len(samples),
        "M": FAST_INTERFERENCE,
        "seed": FAST_SEED,
        "parameter_order": list(FAST_PARAMETER_NAMES),
        "indices": indices,
        "total_order_ranking": ranking,
        "objective_summary": {
            "minimum": float(np.min(output)),
            "mean": float(np.mean(output)),
            "maximum": float(np.max(output)),
            "population_std": float(np.std(output, ddof=0)),
        },
        "analysis_warnings": [str(item.message) for item in caught],
        "raw_objectives_committed": False,
    }


def _run_harmonic_search(omega: float, passive_amplitude: float) -> dict:
    results = {}
    for harmonic in HARMONICS:
        amplitudes, phases, ratios, phase_values = single_harmonic_grid(harmonic)
        objective, steady_error = _evaluate_controls(
            omega, amplitudes, phases, batch_size=32
        )
        best_index = int(np.argmin(objective))
        best_zero = best_index == 0
        results[str(harmonic)] = {
            "harmonic": harmonic,
            "amplitude_ratios": ratios.tolist(),
            "phases_rad": [
                None if index == 0 else float(value)
                for index, value in enumerate(phase_values)
            ],
            "steady_amplitudes": objective.tolist(),
            "steady_errors": steady_error.tolist(),
            "best_index": best_index,
            "best_amplitude_ratio": float(ratios[best_index]),
            "best_amplitude": float(
                ratios[best_index] * REFERENCE_PRELOAD
            ),
            "best_phase_rad": (
                None if best_zero else float(phase_values[best_index])
            ),
            "best_steady_amplitude_at_omega_r": float(objective[best_index]),
            "best_steady_error_at_omega_r": float(steady_error[best_index]),
            "single_frequency_reduction_percent": float(
                100.0
                * (passive_amplitude - objective[best_index])
                / passive_amplitude
            ),
        }
    return results


def _best_control_vectors(harmonic_search: dict) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = np.zeros((5, 4), dtype=np.float64)
    phases = np.zeros_like(amplitudes)
    for row, harmonic in enumerate(HARMONICS, start=1):
        best = harmonic_search[str(harmonic)]
        amplitudes[row, harmonic - 1] = best["best_amplitude"]
        phases[row, harmonic - 1] = best["best_phase_rad"] or 0.0
    return amplitudes, phases


def _run_local_frf(omega_r: float, harmonic_search: dict) -> dict:
    control_amplitudes, control_phases = _best_control_vectors(harmonic_search)
    amplitude_columns = []
    error_columns = []
    for ratio in LOCAL_FRF_RATIOS:
        omega = float(ratio * omega_r)
        preload = fourier_preload(omega, control_amplitudes, control_phases)
        objective, steady_error = _evaluate_preload(omega, preload)
        amplitude_columns.append(objective)
        error_columns.append(steady_error)
    amplitudes = np.column_stack(amplitude_columns)
    errors = np.column_stack(error_columns)
    methods = ("passive", "1omega", "2omega", "3omega", "4omega")
    entries = {}
    passive_peak = float(np.max(amplitudes[0]))
    for row, method in enumerate(methods):
        peak_index = int(np.argmax(amplitudes[row]))
        boundary = peak_index in (0, len(LOCAL_FRF_RATIOS) - 1)
        entries[method] = {
            "steady_amplitudes": amplitudes[row].tolist(),
            "steady_errors": errors[row].tolist(),
            "peak_index": peak_index,
            "peak_ratio": float(LOCAL_FRF_RATIOS[peak_index]),
            "peak_omega": float(LOCAL_FRF_RATIOS[peak_index] * omega_r),
            "peak_amplitude": float(amplitudes[row, peak_index]),
            "peak_steady_error": float(errors[row, peak_index]),
            "peak_at_boundary": boundary,
            "range_status": "range_insufficient" if boundary else "interior",
            "sampled_peak_reduction_vs_passive_percent": float(
                100.0
                * (passive_peak - amplitudes[row, peak_index])
                / passive_peak
            ),
        }
    return {
        "frequency_ratios": LOCAL_FRF_RATIOS.tolist(),
        "omegas": (LOCAL_FRF_RATIOS * omega_r).tolist(),
        "primary_comparison": "sampled local FRF peak",
        "methods": entries,
    }


def _run_energy(omega_r: float, harmonic_search: dict) -> dict:
    all_amplitudes, all_phases = _best_control_vectors(harmonic_search)
    selected_rows = np.asarray([0, 1, 2, 4])
    amplitudes = all_amplitudes[selected_rows]
    phases = all_phases[selected_rows]
    preload = fourier_preload(omega_r, amplitudes, phases)
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega_r, DIAGNOSTIC_NUM_PERIODS
    )
    forcing_bank = np.broadcast_to(forcing, (len(preload), forcing.size))
    outputs = _finite_outputs(
        ENERGY_SIMULATOR(
            jnp.asarray(DAMPING, dtype=jnp.float64),
            jnp.asarray(forcing_bank, dtype=jnp.float64),
            jnp.asarray(preload, dtype=jnp.float64),
            jnp.asarray(time_step, dtype=jnp.float64),
        ),
        "friction energy diagnostic",
    )
    mean_energy, cycle_energy = positive_period_energy(outputs[-1])
    passive = float(mean_energy[0])
    methods = ("passive", "1omega", "2omega", "4omega")
    return {
        "definition": "sum_contacts sum_steps abs(F_contact * delta_slider)",
        "window_cycles": [21, 22, 23, 24],
        "methods": {
            method: {
                "mean_energy_per_period": float(mean_energy[index]),
                "cycle_energies": cycle_energy[index].tolist(),
                "relative_change_vs_passive_percent": float(
                    100.0 * (mean_energy[index] - passive) / passive
                ),
            }
            for index, method in enumerate(methods)
        },
    }


def _run_working_range(omega_r: float, harmonic_search: dict) -> dict:
    best_two = harmonic_search["2"]
    passive_peaks = []
    active_peaks = []
    passive_peak_ratios = []
    active_peak_ratios = []
    passive_normalized = []
    active_normalized = []
    for force_ratio in WORKING_FORCE_RATIOS:
        force = float(force_ratio * FORCING_AMPLITUDE)
        amplitude_rows = []
        for frequency_ratio in WORKING_FRF_RATIOS:
            omega = float(frequency_ratio * omega_r)
            amplitudes = np.zeros((2, 4), dtype=np.float64)
            phases = np.zeros_like(amplitudes)
            amplitudes[1, 1] = best_two["best_amplitude"]
            phases[1, 1] = best_two["best_phase_rad"] or 0.0
            preload = fourier_preload(omega, amplitudes, phases)
            objective, _ = _evaluate_preload(omega, preload, force)
            amplitude_rows.append(objective)
        response = np.column_stack(amplitude_rows)
        indices = np.argmax(response, axis=1)
        peaks = response[np.arange(2), indices]
        peak_ratios = WORKING_FRF_RATIOS[indices]
        peak_omegas = peak_ratios * omega_r
        normalized = peak_omegas**2 * peaks / force
        passive_peaks.append(float(peaks[0]))
        active_peaks.append(float(peaks[1]))
        passive_peak_ratios.append(float(peak_ratios[0]))
        active_peak_ratios.append(float(peak_ratios[1]))
        passive_normalized.append(float(normalized[0]))
        active_normalized.append(float(normalized[1]))
    design_indices = np.flatnonzero(np.isclose(WORKING_FORCE_RATIOS, 1.0))
    if len(design_indices) != 1:
        raise AssertionError("working-range grid must contain F/F0=1 exactly once")
    reference = passive_normalized[int(design_indices[0])]
    qualifying = np.asarray(active_normalized) <= reference
    return {
        "force_ratios": WORKING_FORCE_RATIOS.tolist(),
        "frequency_ratios": WORKING_FRF_RATIOS.tolist(),
        "passive_raw_peaks": passive_peaks,
        "active_2omega_raw_peaks": active_peaks,
        "passive_peak_frequency_ratios": passive_peak_ratios,
        "active_2omega_peak_frequency_ratios": active_peak_ratios,
        "passive_normalized_peaks": passive_normalized,
        "active_2omega_normalized_peaks": active_normalized,
        "passive_design_point_reference": float(reference),
        "active_qualifying_mask": qualifying.tolist(),
        "active_discrete_intervals": discrete_true_intervals(
            WORKING_FORCE_RATIOS, qualifying
        ),
        "interpolation_or_extrapolation_used": False,
    }


def _load_stochastic_appendix(omega_r: float) -> list[dict]:
    appendix = []
    for path in STOCHASTIC_PATHS:
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        configuration = result["configuration"]
        optimization = result["optimization"]
        valid = (
            np.isclose(configuration["omega_r"], omega_r)
            and np.isclose(configuration["forcing_amplitude"], FORCING_AMPLITUDE)
            and configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
            and configuration["fixed_evaluation_realizations"] == 64
            and optimization["optimizer"]["num_updates"] == 100
        )
        if not valid:
            raise RuntimeError(f"Stochastic appendix mismatch: {path.name}")
        appendix.append(
            {
                "learning_rate": float(
                    optimization["optimizer"]["learning_rate"]
                ),
                "neutral_fixed_bank_amplitude": float(
                    optimization["initial_fixed_evaluation_amplitude"]
                ),
                "final_fixed_bank_amplitude": float(
                    optimization["final_fixed_evaluation_amplitude"]
                ),
                "relative_improvement_percent": float(
                    optimization["relative_improvement_percent"]
                ),
                "status": (
                    "provisional fixed-bank results; not part of Wu2019 "
                    "reproduction"
                ),
            }
        )
    return appendix


def _interpretation(
    fast_result: dict, local_frf: dict, energy: dict, working: dict
) -> dict:
    indices = fast_result["indices"]
    weak_names = ("A1", "Phi1", "A3", "Phi3")
    weak_max = max(indices[name]["ST"] for name in weak_names)
    two_dominant = min(indices["A2"]["ST"], indices["Phi2"]["ST"]) > weak_max
    four_secondary = min(indices["A4"]["ST"], indices["Phi4"]["ST"]) > weak_max
    reductions = {
        harmonic: local_frf["methods"][f"{harmonic}omega"][
            "sampled_peak_reduction_vs_passive_percent"
        ]
        for harmonic in HARMONICS
    }
    performance_order = reductions[2] > reductions[4] > max(
        reductions[1], reductions[3]
    )
    two_is_20_percent_scale = 10.0 <= reductions[2] <= 30.0
    qualifying = np.asarray(working["active_qualifying_mask"])
    ratios = np.asarray(working["force_ratios"])
    extends_both_sides = bool(
        np.any(qualifying & (ratios < 1.0))
        and np.any(qualifying & (ratios > 1.0))
    )
    local_peak_ranges_sufficient = not any(
        local_frf["methods"][method]["peak_at_boundary"]
        for method in ("passive", "2omega", "4omega")
    )
    passive_energy = energy["methods"]["passive"][
        "mean_energy_per_period"
    ]
    two_energy = energy["methods"]["2omega"]["mean_energy_per_period"]
    four_energy = energy["methods"]["4omega"]["mean_energy_per_period"]
    wu_energy_trend = two_energy > four_energy > passive_energy
    flags = {
        "2omega_FAST_dominant_over_1omega_3omega": bool(two_dominant),
        "4omega_FAST_secondary_over_1omega_3omega": bool(four_secondary),
        "local_peak_order_2_then_4_then_1_3": bool(performance_order),
        "2omega_local_peak_reduction_is_20_percent_scale": bool(
            two_is_20_percent_scale
        ),
        "active_working_range_extends_below_and_above_design": bool(
            extends_both_sides
        ),
        "passive_2omega_4omega_local_peak_ranges_sufficient": bool(
            local_peak_ranges_sufficient
        ),
        "dissipated_energy_order_2omega_then_4omega_then_passive": bool(
            wu_energy_trend
        ),
    }
    if all(flags.values()):
        category = "Strong reproduction"
    elif two_dominant and reductions[2] > max(reductions[1], reductions[3]):
        category = "Partial reproduction"
    else:
        category = "Divergent result"
    return {
        "category": category,
        "evidence_flags": flags,
        "not_a_pass_fail_gate": True,
        "phase_proximity_to_4p4_is_not_a_criterion": True,
    }


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)
    axis.tick_params(
        direction="in",
        top=False,
        right=False,
        width=1.0,
        colors=FRAME_COLOR,
    )


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.15,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _plot_summary(result: dict) -> None:
    _configure_plotting()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axis_a, axis_b, axis_c, axis_d = axes.ravel()

    names = list(FAST_PARAMETER_NAMES)
    total_indices = [result["fast"]["indices"][name]["ST"] for name in names]
    colors = [
        HARMONIC_COLORS[(index // 2)]
        if index // 2 in (1, 3)
        else WEAK_COLOR
        for index in range(len(names))
    ]
    axis_a.bar(np.arange(len(names)), total_indices, color=colors, width=0.72)
    axis_a.set(
        xticks=np.arange(len(names)),
        xticklabels=[r"$A_1$", r"$\phi_1$", r"$A_2$", r"$\phi_2$", r"$A_3$", r"$\phi_3$", r"$A_4$", r"$\phi_4$"],
        ylabel="FAST total index",
    )

    reductions = [
        result["local_frf"]["methods"][f"{harmonic}omega"][
            "sampled_peak_reduction_vs_passive_percent"
        ]
        for harmonic in HARMONICS
    ]
    axis_b.bar(
        np.arange(4), reductions, color=HARMONIC_COLORS, width=0.68
    )
    axis_b.axhline(0.0, color=FRAME_COLOR, linewidth=0.8)
    axis_b.set(
        xticks=np.arange(4),
        xticklabels=[r"$1\omega$", r"$2\omega$", r"$3\omega$", r"$4\omega$"],
        ylabel="Local peak reduction (%)",
    )

    local = result["local_frf"]
    for method, color, label, linewidth in (
        ("passive", PASSIVE_COLOR, "Passive", 1.6),
        ("1omega", ONE_COLOR, r"Best $1\omega$", 1.2),
        ("2omega", TWO_COLOR, r"Best $2\omega$", 2.1),
        ("3omega", THREE_COLOR, r"Best $3\omega$", 1.2),
        ("4omega", FOUR_COLOR, r"Best $4\omega$", 1.7),
    ):
        axis_c.plot(
            local["frequency_ratios"],
            local["methods"][method]["steady_amplitudes"],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=2.2,
            label=label,
        )
    axis_c.set(
        xlabel=r"Frequency ratio $\omega/\omega_r$",
        ylabel="Steady amplitude",
    )
    axis_c.legend(loc="best", ncol=2)

    energy_methods = ("passive", "1omega", "2omega", "4omega")
    passive_energy = result["friction_energy"]["methods"]["passive"][
        "mean_energy_per_period"
    ]
    energy_percent = [
        100.0
        * result["friction_energy"]["methods"][method][
            "mean_energy_per_period"
        ]
        / passive_energy
        for method in energy_methods
    ]
    axis_d.bar(
        np.arange(4),
        energy_percent,
        color=(PASSIVE_COLOR, ONE_COLOR, TWO_COLOR, FOUR_COLOR),
        width=0.68,
    )
    axis_d.axhline(100.0, color=FRAME_COLOR, linestyle="--", linewidth=0.9)
    axis_d.set(
        xticks=np.arange(4),
        xticklabels=["Passive", r"$1\omega$", r"$2\omega$", r"$4\omega$"],
        ylabel="Dissipated energy (% passive)",
    )

    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _style_axis(axis)
        _panel_label(axis, label)
    figure.savefig(SUMMARY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_working_range(result: dict) -> None:
    _configure_plotting()
    working = result["working_range"]
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.2), constrained_layout=True)
    axis.plot(
        working["force_ratios"],
        working["passive_normalized_peaks"],
        color=PASSIVE_COLOR,
        marker="o",
        linewidth=1.7,
        label="Passive",
    )
    axis.plot(
        working["force_ratios"],
        working["active_2omega_normalized_peaks"],
        color=TWO_COLOR,
        marker="s",
        linewidth=2.1,
        label=r"Best $2\omega$",
    )
    axis.axhline(
        working["passive_design_point_reference"],
        color=FRAME_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="Passive design reference",
    )
    for lower, upper in working["active_discrete_intervals"]:
        axis.axvspan(lower, upper, color=TWO_COLOR, alpha=0.10, linewidth=0)
    axis.set(
        xlabel=r"Excitation ratio $F/F_0$",
        ylabel=r"Normalized peak $\omega_p^2 A_p/F$",
        xlim=(float(WORKING_FORCE_RATIOS[0]), float(WORKING_FORCE_RATIOS[-1])),
    )
    axis.legend(loc="best")
    _style_axis(axis)
    figure.savefig(RANGE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _markdown(result: dict) -> str:
    lines = [
        "# Wu2019 vs JAX-FEM",
        "",
        "This is a Wu-method reproduction on the frozen 32x4 JAX-FEM/Jenkins benchmark, not a reproduction of the paper's SDOF/MHBM solver.",
        "",
        f"FAST scope: **{result['fast']['mode']}** (`N={result['fast']['N_per_parameter']}`, {result['fast']['num_parameter_samples']} total samples).",
        "",
        "## FAST sensitivity",
        "",
        "| Rank | Parameter | S1 | ST |",
        "|---:|---|---:|---:|",
    ]
    for rank, name in enumerate(result["fast"]["total_order_ranking"], start=1):
        entry = result["fast"]["indices"][name]
        lines.append(f"| {rank} | {name} | {entry['S1']:.6g} | {entry['ST']:.6g} |")
    lines.extend(
        [
            "",
            "The paper's Fig. 9 uses the maximum response over a frequency band; W1 uses the frozen-resonance amplitude, so the comparison is ordinal rather than a percentage-error comparison.",
            "",
            "## Harmonic comparison",
            "",
            "| Harmonic | Best A/A0 | Best phase (rad) | Reduction at omega_r (%) | Sampled local max ratio | Sampled local max reduction (%) | Range status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for harmonic in HARMONICS:
        search = result["harmonic_search"][str(harmonic)]
        local = result["local_frf"]["methods"][f"{harmonic}omega"]
        phase = search["best_phase_rad"]
        phase_text = "n/a" if phase is None else f"{phase:.6g}"
        lines.append(
            f"| {harmonic}omega | {search['best_amplitude_ratio']:.6g} | {phase_text} | "
            f"{search['single_frequency_reduction_percent']:.6g} | {local['peak_ratio']:.6g} | "
            f"{local['sampled_peak_reduction_vs_passive_percent']:.6g} | {local['range_status']} |"
        )
    lines.extend(
        [
            "",
            "## Friction energy",
            "",
            "| Control | Mean dissipated energy/period | Change vs passive (%) |",
            "|---|---:|---:|",
        ]
    )
    for method in ("passive", "1omega", "2omega", "4omega"):
        entry = result["friction_energy"]["methods"][method]
        lines.append(
            f"| {method} | {entry['mean_energy_per_period']:.8g} | {entry['relative_change_vs_passive_percent']:.6g} |"
        )
    intervals = result["working_range"]["active_discrete_intervals"]
    lines.extend(
        [
            "",
            "## Excitation working range",
            "",
            f"Discrete active intervals: `{intervals}`. No interpolation or extrapolation was used.",
            "",
            "## Interpretation",
            "",
            f"**{result['interpretation']['category']}**",
            "",
        ]
    )
    for name, value in result["interpretation"]["evidence_flags"].items():
        lines.append(f"- {name}: `{value}`")
    if result["stochastic_appendix"]:
        lines.extend(
            [
                "",
                "## Provisional stochastic appendix",
                "",
                "These fixed-bank results are not part of the Wu2019 reproduction.",
                "",
                "| Adam lr | Neutral | Final | Improvement (%) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for entry in result["stochastic_appendix"]:
            lines.append(
                f"| {entry['learning_rate']:.6g} | {entry['neutral_fixed_bank_amplitude']:.8g} | "
                f"{entry['final_fixed_bank_amplitude']:.8g} | {entry['relative_improvement_percent']:.6g} |"
            )
    return "\n".join(lines) + "\n"


def _timeout_handler(_signum, _frame) -> None:
    raise TimeoutError("W1 exceeded the registered 30-minute runtime budget")


def _run_science(fast_n: int, runtime_estimate: dict | None) -> dict:
    if fast_n < 68 or fast_n > FAST_MAX_N or fast_n % 4:
        raise ValueError("--fast-n must be a multiple of 4 in [68, 1000]")
    if runtime_estimate is not None:
        if runtime_estimate["runtime_gate"] != "PASS":
            raise RuntimeError("runtime estimate did not pass")
        if runtime_estimate["selected_fast_n"] != fast_n:
            raise RuntimeError("--fast-n does not match the timing-gate selection")
    frozen = _load_frozen_gate0()
    started = time.perf_counter()
    stage_seconds = {}

    print("W1 FAST", flush=True)
    stage = time.perf_counter()
    fast_result = _run_fast(frozen["omega_r"], fast_n)
    stage_seconds["fast"] = time.perf_counter() - stage

    print("W1 harmonic search", flush=True)
    stage = time.perf_counter()
    harmonic_search = _run_harmonic_search(
        frozen["omega_r"], frozen["passive_amplitude"]
    )
    stage_seconds["harmonic_search"] = time.perf_counter() - stage

    print("W1 local FRF", flush=True)
    stage = time.perf_counter()
    local_frf = _run_local_frf(frozen["omega_r"], harmonic_search)
    stage_seconds["local_frf"] = time.perf_counter() - stage

    print("W1 friction energy", flush=True)
    stage = time.perf_counter()
    energy = _run_energy(frozen["omega_r"], harmonic_search)
    stage_seconds["friction_energy"] = time.perf_counter() - stage

    print("W1 working range", flush=True)
    stage = time.perf_counter()
    working = _run_working_range(frozen["omega_r"], harmonic_search)
    stage_seconds["working_range"] = time.perf_counter() - stage

    interpretation = _interpretation(fast_result, local_frf, energy, working)
    total_seconds = time.perf_counter() - started
    return {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "preload_A0": REFERENCE_PRELOAD,
            "omega_r": frozen["omega_r"],
            "omega_r_ratio": frozen["omega_r_ratio"],
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": STEPS_PER_PERIOD,
            "objective_cycles": [21, 22, 23, 24],
            "steady_windows": [[17, 18, 19, 20], [21, 22, 23, 24]],
            "local_frf_ratios": LOCAL_FRF_RATIOS.tolist(),
            "working_force_ratios": WORKING_FORCE_RATIOS.tolist(),
            "working_frf_ratios": WORKING_FRF_RATIOS.tolist(),
        },
        "wu_references": wu_reference_table(),
        "frozen_passive": frozen,
        "runtime": {
            "budget_seconds": RUNTIME_BUDGET_SECONDS,
            "timing_gate": runtime_estimate,
            "stage_seconds": stage_seconds,
            "scientific_runner_seconds": total_seconds,
        },
        "fast": fast_result,
        "harmonic_search": harmonic_search,
        "local_frf": local_frf,
        "friction_energy": energy,
        "working_range": working,
        "interpretation": interpretation,
        "stochastic_appendix": _load_stochastic_appendix(
            frozen["omega_r"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--timing-file", type=Path)
    parser.add_argument("--fast-n", type=int)
    parser.add_argument("--render-existing", action="store_true")
    arguments = parser.parse_args()

    if arguments.render_existing:
        if (
            arguments.estimate_only
            or arguments.fast_n is not None
            or arguments.timing_file is not None
            or arguments.timing_output is not None
        ):
            parser.error("--render-existing cannot be combined with run options")
        result = json.loads(RESULTS_PATH.read_text())
        result["interpretation"] = _interpretation(
            result["fast"],
            result["local_frf"],
            result["friction_energy"],
            result["working_range"],
        )
        RESULTS_PATH.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        MARKDOWN_PATH.write_text(_markdown(result))
        _plot_summary(result)
        _plot_working_range(result)
        print(result["interpretation"]["category"])
        return 0

    if arguments.estimate_only:
        if arguments.fast_n is not None or arguments.timing_file is not None:
            parser.error("--estimate-only cannot be combined with full-run options")
        estimate = _estimate_runtime()
        payload = json.dumps(estimate, indent=2, allow_nan=False) + "\n"
        if arguments.timing_output is not None:
            arguments.timing_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.timing_output.write_text(payload)
        print(payload, end="")
        return 0 if estimate["runtime_gate"] == "PASS" else 2

    if arguments.fast_n is None:
        parser.error("full science mode requires --fast-n")
    runtime_estimate = None
    if arguments.timing_file is not None:
        runtime_estimate = json.loads(arguments.timing_file.read_text())

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, RUNTIME_BUDGET_SECONDS)
    try:
        result = _run_science(arguments.fast_n, runtime_estimate)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    MARKDOWN_PATH.write_text(_markdown(result))
    _plot_summary(result)
    _plot_working_range(result)

    print("## W1 complete")
    print(f"FAST_N={arguments.fast_n}")
    print(f"FAST_samples={result['fast']['num_parameter_samples']}")
    print(f"classification={result['interpretation']['category']}")
    print(
        "runtime_seconds="
        f"{result['runtime']['scientific_runner_seconds']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
