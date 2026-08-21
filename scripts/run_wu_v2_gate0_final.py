"""Run the final Wu-style passive-peak and harmonic-authority gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FINAL_REFINEMENT_RATIOS,
    FORCING_AMPLITUDE,
    LOCAL_FRF_RATIOS,
    MINIMUM_ADDITIONAL_REDUCTION_POINTS,
    MINIMUM_PASSIVE_REDUCTION_PERCENT,
    PHASES,
    STEADY_STATE_TOLERANCE,
    SYSTEM,
    constant_preload,
    diagnostic_steady_state_metrics,
    frf_peak_indices,
    harmonic_preload,
    simulate_preload_bank,
)


OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_gate0_final"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "gate0_final_summary.png"
DIAGNOSTIC_RESULTS_PATH = (
    ROOT / "outputs/wu_v2_passive_diagnostic/results.json"
)

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#707780"
ONE_COLOR = "#A55C45"
TWO_COLOR = "#2D6388"
ACCENT_COLOR = "#7B5485"


def _load_frozen_preload() -> float:
    diagnostic = json.loads(DIAGNOSTIC_RESULTS_PATH.read_text())
    comparison = diagnostic["comparison"]
    preload = float(comparison["peak_best_preload"])
    if not comparison["peak_best_interior"] or not np.isclose(
        preload, 0.04, rtol=0.0, atol=1e-15
    ):
        raise RuntimeError("A0-D did not freeze the expected interior N*=0.04")
    return preload


def _evaluate(
    omega: float, preload: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outputs = simulate_preload_bank(omega, preload)
    output_arrays = tuple(np.asarray(output) for output in outputs)
    if not all(np.all(np.isfinite(output)) for output in output_arrays):
        raise RuntimeError(f"Non-finite mechanics output at omega={omega}")
    displacement = output_arrays[0]
    objective, steady_error, cycle_amplitudes = (
        diagnostic_steady_state_metrics(displacement)
    )
    if not (
        np.all(np.isfinite(objective))
        and np.all(np.isfinite(steady_error))
        and np.all(np.isfinite(cycle_amplitudes))
    ):
        raise RuntimeError(f"Non-finite Gate 0 metric at omega={omega}")
    return displacement, objective, steady_error, cycle_amplitudes


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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
        axis="both",
        direction="in",
        top=False,
        right=False,
        width=1.0,
        colors=FRAME_COLOR,
    )


def _plot_refinement(axis, results: dict) -> None:
    refinement = results["refinement"]
    axis.plot(
        refinement["frequency_ratios"],
        refinement["steady_amplitudes"],
        color=PASSIVE_COLOR,
        marker="o",
        markersize=3.8,
        linewidth=1.7,
    )
    axis.scatter(
        [refinement["omega_r_ratio"]],
        [refinement["peak_amplitude"]],
        color=ACCENT_COLOR,
        edgecolor="white",
        linewidth=0.5,
        s=32,
        zorder=3,
    )
    axis.set(
        xlabel=r"Frequency ratio $\omega/\omega_1$",
        ylabel="Steady amplitude",
    )


def _plot(results: dict) -> None:
    _configure_plotting()
    if results["phase_sweep"] is None:
        figure, axis = plt.subplots(
            1, 1, figsize=(3.6, 3.2), constrained_layout=True
        )
        _plot_refinement(axis, results)
        _style_axis(axis)
        axis.text(
            -0.15,
            1.04,
            "a",
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    else:
        figure = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.18))
        axis_a = figure.add_subplot(grid[0, 0])
        axis_c = figure.add_subplot(grid[0, 1])
        axis_b = figure.add_subplot(grid[1, :])
        _plot_refinement(axis_a, results)

        phase_sweep = results["phase_sweep"]
        phase_fraction = np.asarray(phase_sweep["phase_fraction"])
        axis_b.axhline(
            0.0, color=PASSIVE_COLOR, linestyle="--", linewidth=1.2
        )
        axis_b.plot(
            phase_fraction,
            phase_sweep["one_omega"]["relative_change_percent"],
            color=ONE_COLOR,
            linewidth=1.8,
            label=r"$1\omega$",
        )
        axis_b.plot(
            phase_fraction,
            phase_sweep["two_omega"]["relative_change_percent"],
            color=TWO_COLOR,
            linewidth=2.2,
            label=r"$2\omega$",
        )
        axis_b.set(
            xlabel=r"Phase $\phi/2\pi$",
            ylabel="Change from passive (%)",
            xlim=(0.0, 1.0),
            xticks=np.linspace(0.0, 1.0, 5),
        )
        axis_b.legend(loc="best")

        local = results["local_frf"]
        for key, color, label, marker in (
            ("passive", PASSIVE_COLOR, "Passive", "o"),
            ("one_omega", ONE_COLOR, r"Best $1\omega$", "^"),
            ("two_omega", TWO_COLOR, r"Best $2\omega$", "s"),
        ):
            axis_c.plot(
                local["frequency_ratios"],
                local[key]["steady_amplitudes"],
                color=color,
                marker=marker,
                markersize=3.4,
                linewidth=1.7,
                label=label,
            )
        axis_c.set(
            xlabel=r"Frequency ratio $\omega/\omega_r$",
            ylabel="Steady amplitude",
        )
        axis_c.legend(loc="best")

        for label, axis in (("a", axis_a), ("b", axis_b), ("c", axis_c)):
            _style_axis(axis)
            axis.text(
                -0.14,
                1.04,
                label,
                transform=axis.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
            )
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    _plot(results)


def _gate_summary(gates: dict[str, bool | None]) -> tuple[list[str], list[str]]:
    failed = [name for name, value in gates.items() if value is False]
    skipped = [name for name, value in gates.items() if value is None]
    return failed, skipped


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frozen_preload = _load_frozen_preload()
    gates: dict[str, bool | None] = {
        "refined_peak_interior": None,
        "passive_steady": None,
        "best_1omega_steady": None,
        "best_2omega_steady": None,
        "2omega_reduction_at_least_5_percent": None,
        "2omega_advantage_at_least_2_points": None,
        "2omega_has_better_and_worse_phases": None,
        "passive_local_peak_interior": None,
        "best_2omega_local_peak_interior": None,
        "passive_local_peak_steady": None,
        "best_2omega_local_peak_steady": None,
        "best_2omega_local_peak_below_passive": None,
    }

    refinement_amplitudes = []
    refinement_errors = []
    refinement_cycles = []
    refinement_preload = constant_preload(
        frozen_preload, num_periods=DIAGNOSTIC_NUM_PERIODS
    )
    for ratio in FINAL_REFINEMENT_RATIOS:
        _, objective, steady_error, cycles = _evaluate(
            float(ratio * SYSTEM.omega_1), refinement_preload
        )
        refinement_amplitudes.append(float(objective[0]))
        refinement_errors.append(float(steady_error[0]))
        refinement_cycles.append(cycles[0].tolist())
    refinement_amplitudes_array = np.asarray(refinement_amplitudes)
    refinement_peak_index = int(np.argmax(refinement_amplitudes_array))
    omega_r_ratio = float(FINAL_REFINEMENT_RATIOS[refinement_peak_index])
    omega_r = float(omega_r_ratio * SYSTEM.omega_1)
    gates["refined_peak_interior"] = bool(
        0 < refinement_peak_index < len(FINAL_REFINEMENT_RATIOS) - 1
    )

    refinement = {
        "frequency_ratios": FINAL_REFINEMENT_RATIOS.tolist(),
        "omegas": (FINAL_REFINEMENT_RATIOS * SYSTEM.omega_1).tolist(),
        "steady_amplitudes": refinement_amplitudes,
        "steady_errors": refinement_errors,
        "cycle_amplitudes": refinement_cycles,
        "peak_index": refinement_peak_index,
        "omega_r_ratio": omega_r_ratio,
        "omega_r": omega_r,
        "peak_amplitude": float(
            refinement_amplitudes_array[refinement_peak_index]
        ),
        "peak_steady_error": refinement_errors[refinement_peak_index],
        "peak_at_boundary": not gates["refined_peak_interior"],
    }
    configuration = {
        "mesh": "32x4 QUAD4",
        "num_free_dofs": SYSTEM.num_free_dofs,
        "damping": DAMPING,
        "forcing_amplitude": FORCING_AMPLITUDE,
        "num_periods": DIAGNOSTIC_NUM_PERIODS,
        "steps_per_period": 100,
        "previous_window_cycles": [17, 18, 19, 20],
        "final_window_cycles": [21, 22, 23, 24],
        "steady_tolerance": STEADY_STATE_TOLERANCE,
        "frozen_preload": frozen_preload,
        "frozen_preload_source": (
            "outputs/wu_v2_passive_diagnostic/results.json"
        ),
        "refinement_ratios": FINAL_REFINEMENT_RATIOS.tolist(),
        "local_frf_ratios": LOCAL_FRF_RATIOS.tolist(),
        "phases": PHASES.tolist(),
        "modulation_depth": 0.25,
    }

    if not gates["refined_peak_interior"]:
        failed, skipped = _gate_summary(gates)
        results = {
            "configuration": configuration,
            "refinement": refinement,
            "passive": None,
            "phase_sweep": None,
            "local_frf": None,
            "gates": gates,
            "gate_0": {
                "result": "FAIL",
                "reason": "Passive resonance refinement range was insufficient.",
                "failed_checks": failed,
                "skipped_checks": skipped,
            },
        }
        _write_results(results)
        print("Final Gate 0: FAIL")
        print(results["gate_0"]["reason"])
        return 1

    passive_displacement, passive_objective, passive_error, passive_cycles = (
        _evaluate(omega_r, refinement_preload)
    )
    passive_amplitude = float(passive_objective[0])
    passive_steady_error = float(passive_error[0])
    passive = {
        "preload": frozen_preload,
        "omega_r_ratio": omega_r_ratio,
        "omega_r": omega_r,
        "steady_amplitude": passive_amplitude,
        "steady_error": passive_steady_error,
        "peak_displacement_final_four_cycles": float(
            np.max(np.abs(passive_displacement[0, -400:]))
        ),
        "cycle_amplitudes": passive_cycles[0].tolist(),
    }
    gates["passive_steady"] = bool(
        passive_steady_error <= STEADY_STATE_TOLERANCE
    )
    if not gates["passive_steady"]:
        failed, skipped = _gate_summary(gates)
        results = {
            "configuration": configuration,
            "refinement": refinement,
            "passive": passive,
            "phase_sweep": None,
            "local_frf": None,
            "gates": gates,
            "gate_0": {
                "result": "FAIL",
                "reason": "Passive reference did not satisfy the steady criterion.",
                "failed_checks": failed,
                "skipped_checks": skipped,
            },
        }
        _write_results(results)
        print("Final Gate 0: FAIL")
        print(results["gate_0"]["reason"])
        return 1

    one_preload = harmonic_preload(
        frozen_preload, omega_r, 1, PHASES, DIAGNOSTIC_NUM_PERIODS
    )
    two_preload = harmonic_preload(
        frozen_preload, omega_r, 2, PHASES, DIAGNOSTIC_NUM_PERIODS
    )
    _, phase_objective, phase_errors, phase_cycles = _evaluate(
        omega_r, np.concatenate((one_preload, two_preload), axis=0)
    )
    one_amplitudes = phase_objective[: len(PHASES)]
    two_amplitudes = phase_objective[len(PHASES) :]
    one_errors = phase_errors[: len(PHASES)]
    two_errors = phase_errors[len(PHASES) :]
    one_cycles = phase_cycles[: len(PHASES)]
    two_cycles = phase_cycles[len(PHASES) :]
    one_best_index = int(np.argmin(one_amplitudes))
    one_worst_index = int(np.argmax(one_amplitudes))
    two_best_index = int(np.argmin(two_amplitudes))
    two_worst_index = int(np.argmax(two_amplitudes))
    one_improvement = 100.0 * (
        passive_amplitude - one_amplitudes[one_best_index]
    ) / passive_amplitude
    two_improvement = 100.0 * (
        passive_amplitude - two_amplitudes[two_best_index]
    ) / passive_amplitude
    additional_improvement = float(two_improvement - one_improvement)

    gates.update(
        {
            "best_1omega_steady": bool(
                one_errors[one_best_index] <= STEADY_STATE_TOLERANCE
            ),
            "best_2omega_steady": bool(
                two_errors[two_best_index] <= STEADY_STATE_TOLERANCE
            ),
            "2omega_reduction_at_least_5_percent": bool(
                two_improvement >= MINIMUM_PASSIVE_REDUCTION_PERCENT
            ),
            "2omega_advantage_at_least_2_points": bool(
                additional_improvement >= MINIMUM_ADDITIONAL_REDUCTION_POINTS
            ),
            "2omega_has_better_and_worse_phases": bool(
                np.min(two_amplitudes) < passive_amplitude
                and np.max(two_amplitudes) > passive_amplitude
            ),
        }
    )
    phase_sweep = {
        "phases": PHASES.tolist(),
        "phase_fraction": (PHASES / (2.0 * np.pi)).tolist(),
        "one_omega": {
            "steady_amplitudes": one_amplitudes.tolist(),
            "steady_errors": one_errors.tolist(),
            "cycle_amplitudes": one_cycles.tolist(),
            "relative_change_percent": (
                100.0 * (one_amplitudes - passive_amplitude) / passive_amplitude
            ).tolist(),
            "best_index": one_best_index,
            "best_phase": float(PHASES[one_best_index]),
            "best_phase_fraction": float(
                PHASES[one_best_index] / (2.0 * np.pi)
            ),
            "best_amplitude": float(one_amplitudes[one_best_index]),
            "best_steady_error": float(one_errors[one_best_index]),
            "best_improvement_percent": float(one_improvement),
            "worst_index": one_worst_index,
            "worst_phase": float(PHASES[one_worst_index]),
            "worst_phase_fraction": float(
                PHASES[one_worst_index] / (2.0 * np.pi)
            ),
            "worst_amplitude": float(one_amplitudes[one_worst_index]),
        },
        "two_omega": {
            "steady_amplitudes": two_amplitudes.tolist(),
            "steady_errors": two_errors.tolist(),
            "cycle_amplitudes": two_cycles.tolist(),
            "relative_change_percent": (
                100.0 * (two_amplitudes - passive_amplitude) / passive_amplitude
            ).tolist(),
            "best_index": two_best_index,
            "best_phase": float(PHASES[two_best_index]),
            "best_phase_fraction": float(
                PHASES[two_best_index] / (2.0 * np.pi)
            ),
            "best_amplitude": float(two_amplitudes[two_best_index]),
            "best_steady_error": float(two_errors[two_best_index]),
            "best_improvement_percent": float(two_improvement),
            "worst_index": two_worst_index,
            "worst_phase": float(PHASES[two_worst_index]),
            "worst_phase_fraction": float(
                PHASES[two_worst_index] / (2.0 * np.pi)
            ),
            "worst_amplitude": float(two_amplitudes[two_worst_index]),
        },
        "2omega_vs_1omega_improvement_points": additional_improvement,
    }

    local_amplitude_columns = []
    local_error_columns = []
    for ratio in LOCAL_FRF_RATIOS:
        current_omega = float(ratio * omega_r)
        local_bank = np.concatenate(
            (
                constant_preload(
                    frozen_preload, num_periods=DIAGNOSTIC_NUM_PERIODS
                ),
                harmonic_preload(
                    frozen_preload,
                    current_omega,
                    1,
                    np.asarray([PHASES[one_best_index]]),
                    DIAGNOSTIC_NUM_PERIODS,
                ),
                harmonic_preload(
                    frozen_preload,
                    current_omega,
                    2,
                    np.asarray([PHASES[two_best_index]]),
                    DIAGNOSTIC_NUM_PERIODS,
                ),
            ),
            axis=0,
        )
        _, objective, steady_error, _ = _evaluate(current_omega, local_bank)
        local_amplitude_columns.append(objective)
        local_error_columns.append(steady_error)
    local_amplitudes = np.column_stack(local_amplitude_columns)
    local_errors = np.column_stack(local_error_columns)
    local_peak_indices = frf_peak_indices(local_amplitudes)
    rows = np.arange(3)
    local_peak_amplitudes = local_amplitudes[rows, local_peak_indices]
    local_peak_errors = local_errors[rows, local_peak_indices]
    local_peak_ratios = LOCAL_FRF_RATIOS[local_peak_indices]
    local_peak_omegas = local_peak_ratios * omega_r
    local_boundary = (local_peak_indices == 0) | (
        local_peak_indices == len(LOCAL_FRF_RATIOS) - 1
    )

    gates.update(
        {
            "passive_local_peak_interior": bool(not local_boundary[0]),
            "best_2omega_local_peak_interior": bool(not local_boundary[2]),
            "passive_local_peak_steady": bool(
                local_peak_errors[0] <= STEADY_STATE_TOLERANCE
            ),
            "best_2omega_local_peak_steady": bool(
                local_peak_errors[2] <= STEADY_STATE_TOLERANCE
            ),
            "best_2omega_local_peak_below_passive": bool(
                local_peak_amplitudes[2] < local_peak_amplitudes[0]
            ),
        }
    )

    method_keys = ("passive", "one_omega", "two_omega")
    local_methods = {}
    for index, key in enumerate(method_keys):
        local_methods[key] = {
            "steady_amplitudes": local_amplitudes[index].tolist(),
            "steady_errors": local_errors[index].tolist(),
            "peak_index": int(local_peak_indices[index]),
            "peak_ratio": float(local_peak_ratios[index]),
            "peak_omega": float(local_peak_omegas[index]),
            "peak_amplitude": float(local_peak_amplitudes[index]),
            "peak_steady_error": float(local_peak_errors[index]),
            "peak_at_boundary": bool(local_boundary[index]),
            "range_status": (
                "range_insufficient" if local_boundary[index] else "interior"
            ),
        }
    local_frf = {
        "frequency_ratios": LOCAL_FRF_RATIOS.tolist(),
        "omegas": (LOCAL_FRF_RATIOS * omega_r).tolist(),
        **local_methods,
        "passive_to_2omega_peak_reduction_percent": (
            None
            if local_boundary[0] or local_boundary[2]
            else float(
                100.0
                * (local_peak_amplitudes[0] - local_peak_amplitudes[2])
                / local_peak_amplitudes[0]
            )
        ),
        "2omega_vs_1omega_peak_reduction_percent": (
            None
            if local_boundary[1]
            else float(
                100.0
                * (local_peak_amplitudes[1] - local_peak_amplitudes[2])
                / local_peak_amplitudes[1]
            )
        ),
    }

    passed = bool(all(value is True for value in gates.values()))
    failed, skipped = _gate_summary(gates)
    reason = (
        "All pre-registered Final Gate 0 checks passed."
        if passed
        else "One or more pre-registered Final Gate 0 checks failed."
    )
    results = {
        "configuration": configuration,
        "refinement": refinement,
        "passive": passive,
        "phase_sweep": phase_sweep,
        "local_frf": local_frf,
        "gates": gates,
        "gate_0": {
            "result": "PASS" if passed else "FAIL",
            "reason": reason,
            "failed_checks": failed,
            "skipped_checks": skipped,
        },
    }
    _write_results(results)

    print("## Final Gate 0")
    print(f"N_star={frozen_preload:.12g}")
    print(f"omega_r={omega_r:.12g} ({omega_r_ratio:.6g} omega_1)")
    print(
        f"passive_A={passive_amplitude:.12g} "
        f"steady_error={passive_steady_error:.6g}"
    )
    print(
        f"best_1omega_phase={PHASES[one_best_index]:.6g} "
        f"A={one_amplitudes[one_best_index]:.12g} "
        f"improvement={one_improvement:.6g}% "
        f"steady_error={one_errors[one_best_index]:.6g}"
    )
    print(
        f"best_2omega_phase={PHASES[two_best_index]:.6g} "
        f"A={two_amplitudes[two_best_index]:.12g} "
        f"improvement={two_improvement:.6g}% "
        f"steady_error={two_errors[two_best_index]:.6g}"
    )
    for key in method_keys:
        entry = local_frf[key]
        print(
            f"local_{key}_peak_ratio={entry['peak_ratio']:.6g} "
            f"A={entry['peak_amplitude']:.12g} "
            f"steady_error={entry['peak_steady_error']:.6g} "
            f"status={entry['range_status']}"
        )
    print(f"Final Gate 0: {results['gate_0']['result']}")
    print(f"failed_checks={failed}")
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
