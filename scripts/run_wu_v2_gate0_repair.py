"""Run the frozen 16-cycle repair of the Wu-style engineering gate."""

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
    COARSE_FREQUENCY_RATIOS,
    DAMPING,
    FINE_FREQUENCY_HALF_WIDTH,
    FINE_FREQUENCY_POINTS,
    FORCING_AMPLITUDE,
    MINIMUM_ADDITIONAL_REDUCTION_POINTS,
    MINIMUM_PASSIVE_REDUCTION_PERCENT,
    PHASES,
    REFERENCE_PRELOAD,
    REPAIR_NUM_PERIODS,
    REPAIR_PRELOAD_VALUES,
    STEADY_STATE_TOLERANCE,
    SYSTEM,
    constant_preload,
    harmonic_preload,
    repair_steady_state_metrics,
    simulate_preload_bank,
)


OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_gate0_repair"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "gate0_repair_summary.png"

FRAME_COLOR = "#20242A"
NEUTRAL_COLOR = "#666B73"
FIRST_COLOR = "#A36A3F"
SECOND_COLOR = "#27628D"
ACCENT_COLOR = "#8F355E"


def _response_for_frequency_ratio(ratio: float) -> dict:
    omega = float(ratio * SYSTEM.omega_1)
    outputs = simulate_preload_bank(
        omega,
        constant_preload(
            REFERENCE_PRELOAD, num_periods=REPAIR_NUM_PERIODS
        ),
    )
    objective, convergence, amplitudes = repair_steady_state_metrics(
        outputs[0]
    )
    return {
        "ratio": float(ratio),
        "omega": omega,
        "steady_amplitude": float(objective[0]),
        "steady_error": float(convergence[0]),
        "cycle_amplitudes": amplitudes[0].tolist(),
    }


def _frequency_scan(ratios: np.ndarray) -> list[dict]:
    return [_response_for_frequency_ratio(float(ratio)) for ratio in ratios]


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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


def _plot(results: dict) -> None:
    _configure_plotting()
    figure = plt.figure(figsize=(8.0, 6.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.18))
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]

    resonance = results["resonance"]
    axes[0].plot(
        resonance["frequency_ratios"],
        resonance["steady_amplitudes"],
        color=SECOND_COLOR,
        linewidth=1.8,
    )
    axes[0].scatter(
        [resonance["omega_r_ratio"]],
        [resonance["peak_amplitude"]],
        color=ACCENT_COLOR,
        s=32,
        zorder=3,
    )
    axes[0].set(
        xlabel=r"Frequency ratio $\omega/\omega_1$",
        ylabel="Steady amplitude",
    )

    passive = results["passive_preload"]
    axes[1].plot(
        passive["preload_values"],
        passive["steady_amplitudes"],
        color=NEUTRAL_COLOR,
        marker="o",
        markersize=4.5,
        linewidth=1.7,
    )
    axes[1].scatter(
        [passive["n_star"]],
        [passive["n_star_amplitude"]],
        color=ACCENT_COLOR,
        s=32,
        zorder=3,
    )
    axes[1].set(xlabel="Constant preload", ylabel="Steady amplitude")

    axes[2].axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.4)
    phase = results["phase_sweep"]
    if phase is not None:
        axes[2].plot(
            phase["phases"],
            phase["one_omega_relative_change_percent"],
            color=FIRST_COLOR,
            linewidth=1.7,
            label=r"$1\omega$",
        )
        axes[2].plot(
            phase["phases"],
            phase["two_omega_relative_change_percent"],
            color=SECOND_COLOR,
            linewidth=2.0,
            label=r"$2\omega$",
        )
        axes[2].legend(loc="best")
    axes[2].set(
        xlabel="Phase (rad)",
        ylabel="Change from passive (%)",
        xlim=(0.0, 2.0 * np.pi),
        xticks=np.arange(0.0, 2.0 * np.pi + 1e-12, np.pi / 2.0),
        xticklabels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )

    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        _style_axis(axis)
        axis.text(
            -0.13,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _failure_classes(gates: dict) -> list[str]:
    classes = []
    steady_checks = (
        "passive_steady",
        "best_1omega_steady",
        "best_2omega_steady",
    )
    if any(gates.get(name) is False for name in steady_checks):
        classes.append("A: 16 periods did not satisfy the 2% steady criterion")
    if (
        gates["passive_optimum_interior"] is False
        or gates["passive_local_minimum"] is False
    ):
        classes.append("B: the passive optimum remained monotonic or on a boundary")
    performance_checks = (
        "2omega_reduction_at_least_5_percent",
        "2omega_advantage_at_least_2_points",
        "2omega_has_better_and_worse_phases",
    )
    if any(gates.get(name) is False for name in performance_checks):
        classes.append("C: the registered 2omega advantage was not retained")
    return classes


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    coarse = _frequency_scan(COARSE_FREQUENCY_RATIOS)
    coarse_amplitudes = np.asarray(
        [point["steady_amplitude"] for point in coarse]
    )
    coarse_peak_index = int(np.argmax(coarse_amplitudes))
    coarse_peak_ratio = float(COARSE_FREQUENCY_RATIOS[coarse_peak_index])
    fine_ratios = np.linspace(
        coarse_peak_ratio - FINE_FREQUENCY_HALF_WIDTH,
        coarse_peak_ratio + FINE_FREQUENCY_HALF_WIDTH,
        FINE_FREQUENCY_POINTS,
    )
    fine = _frequency_scan(fine_ratios)
    fine_amplitudes = np.asarray(
        [point["steady_amplitude"] for point in fine]
    )
    fine_peak_index = int(np.argmax(fine_amplitudes))
    omega_r_ratio = float(fine_ratios[fine_peak_index])
    omega_r = float(omega_r_ratio * SYSTEM.omega_1)

    combined = {point["ratio"]: point for point in coarse + fine}
    frequency_points = [combined[key] for key in sorted(combined)]

    passive_outputs = simulate_preload_bank(
        omega_r,
        np.concatenate(
            [
                constant_preload(value, num_periods=REPAIR_NUM_PERIODS)
                for value in REPAIR_PRELOAD_VALUES
            ],
            axis=0,
        ),
    )
    passive_objectives, passive_errors, passive_cycles = (
        repair_steady_state_metrics(passive_outputs[0])
    )
    n_star_index = int(np.argmin(passive_objectives))
    n_star = float(REPAIR_PRELOAD_VALUES[n_star_index])
    passive_amplitude = float(passive_objectives[n_star_index])

    gates: dict[str, bool | None] = {
        "all_resonance_values_finite": bool(
            np.all(np.isfinite(coarse_amplitudes))
            and np.all(np.isfinite(fine_amplitudes))
        ),
        "coarse_peak_interior": 0 < coarse_peak_index < len(coarse) - 1,
        "fine_peak_interior": 0 < fine_peak_index < len(fine) - 1,
        "all_passive_values_finite": bool(
            np.all(np.isfinite(passive_objectives))
            and np.all(np.isfinite(passive_errors))
        ),
        "passive_optimum_interior": 0
        < n_star_index
        < len(REPAIR_PRELOAD_VALUES) - 1,
        "passive_local_minimum": bool(
            0 < n_star_index < len(REPAIR_PRELOAD_VALUES) - 1
            and passive_objectives[n_star_index]
            <= passive_objectives[n_star_index - 1]
            and passive_objectives[n_star_index]
            <= passive_objectives[n_star_index + 1]
        ),
        "passive_steady": bool(
            passive_errors[n_star_index] <= STEADY_STATE_TOLERANCE
        ),
        "best_1omega_steady": None,
        "best_2omega_steady": None,
        "2omega_reduction_at_least_5_percent": None,
        "2omega_advantage_at_least_2_points": None,
        "2omega_has_better_and_worse_phases": None,
    }
    passive_preconditions = all(
        gates[name]
        for name in (
            "all_resonance_values_finite",
            "coarse_peak_interior",
            "fine_peak_interior",
            "all_passive_values_finite",
            "passive_optimum_interior",
            "passive_local_minimum",
            "passive_steady",
        )
    )

    phase_results = None
    if passive_preconditions:
        one_preload = harmonic_preload(
            n_star, omega_r, 1, PHASES, REPAIR_NUM_PERIODS
        )
        two_preload = harmonic_preload(
            n_star, omega_r, 2, PHASES, REPAIR_NUM_PERIODS
        )
        phase_outputs = simulate_preload_bank(
            omega_r, np.concatenate((one_preload, two_preload), axis=0)
        )
        phase_objectives, phase_errors, phase_cycles = (
            repair_steady_state_metrics(phase_outputs[0])
        )
        one_objectives = phase_objectives[: len(PHASES)]
        two_objectives = phase_objectives[len(PHASES) :]
        one_errors = phase_errors[: len(PHASES)]
        two_errors = phase_errors[len(PHASES) :]
        one_cycles = phase_cycles[: len(PHASES)]
        two_cycles = phase_cycles[len(PHASES) :]
        one_best_index = int(np.argmin(one_objectives))
        two_best_index = int(np.argmin(two_objectives))
        one_best_amplitude = float(one_objectives[one_best_index])
        two_best_amplitude = float(two_objectives[two_best_index])
        one_reduction = 100.0 * (
            passive_amplitude - one_best_amplitude
        ) / passive_amplitude
        two_reduction = 100.0 * (
            passive_amplitude - two_best_amplitude
        ) / passive_amplitude
        additional_reduction = two_reduction - one_reduction
        one_relative_change = 100.0 * (
            one_objectives - passive_amplitude
        ) / passive_amplitude
        two_relative_change = 100.0 * (
            two_objectives - passive_amplitude
        ) / passive_amplitude

        gates.update(
            {
                "best_1omega_steady": bool(
                    one_errors[one_best_index] <= STEADY_STATE_TOLERANCE
                ),
                "best_2omega_steady": bool(
                    two_errors[two_best_index] <= STEADY_STATE_TOLERANCE
                ),
                "2omega_reduction_at_least_5_percent": bool(
                    two_reduction >= MINIMUM_PASSIVE_REDUCTION_PERCENT
                ),
                "2omega_advantage_at_least_2_points": bool(
                    additional_reduction
                    >= MINIMUM_ADDITIONAL_REDUCTION_POINTS
                ),
                "2omega_has_better_and_worse_phases": bool(
                    np.min(two_objectives) < passive_amplitude
                    and np.max(two_objectives) > passive_amplitude
                ),
            }
        )
        phase_results = {
            "phases": PHASES.tolist(),
            "one_omega_amplitudes": one_objectives.tolist(),
            "two_omega_amplitudes": two_objectives.tolist(),
            "one_omega_relative_change_percent": one_relative_change.tolist(),
            "two_omega_relative_change_percent": two_relative_change.tolist(),
            "one_omega_steady_error": one_errors.tolist(),
            "two_omega_steady_error": two_errors.tolist(),
            "one_omega_cycle_amplitudes": one_cycles.tolist(),
            "two_omega_cycle_amplitudes": two_cycles.tolist(),
            "best_1omega_index": one_best_index,
            "best_1omega_phase": float(PHASES[one_best_index]),
            "best_1omega_amplitude": one_best_amplitude,
            "best_1omega_improvement_percent": one_reduction,
            "best_1omega_steady_error": float(one_errors[one_best_index]),
            "best_2omega_index": two_best_index,
            "best_2omega_phase": float(PHASES[two_best_index]),
            "best_2omega_amplitude": two_best_amplitude,
            "best_2omega_improvement_percent": two_reduction,
            "best_2omega_steady_error": float(two_errors[two_best_index]),
            "2omega_vs_1omega_improvement_points": additional_reduction,
            "worst_2omega_amplitude": float(np.max(two_objectives)),
        }

    passed = bool(phase_results is not None and all(gates.values()))
    failed_checks = [
        name for name, result in gates.items() if result is False
    ]
    skipped_checks = [name for name, result in gates.items() if result is None]
    failure_classes = [] if passed else _failure_classes(gates)
    if passed:
        reason = "All pre-registered Gate 0-R checks passed."
    elif not passive_preconditions:
        reason = (
            "Gate 0-R failed before harmonic sweeps because the passive "
            "comparator was not valid."
        )
    else:
        reason = "Gate 0-R failed after the harmonic phase sweeps."

    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "reference_preload": REFERENCE_PRELOAD,
            "num_periods": REPAIR_NUM_PERIODS,
            "steps_per_period": 100,
            "previous_window_cycles": [9, 10, 11, 12],
            "final_window_cycles": [13, 14, 15, 16],
            "steady_tolerance": STEADY_STATE_TOLERANCE,
            "coarse_frequency_ratios": COARSE_FREQUENCY_RATIOS.tolist(),
            "fine_half_width": FINE_FREQUENCY_HALF_WIDTH,
            "fine_points": FINE_FREQUENCY_POINTS,
            "preload_values": REPAIR_PRELOAD_VALUES.tolist(),
            "phase_grid": PHASES.tolist(),
            "modulation_depth": 0.25,
            "minimum_passive_reduction_percent": (
                MINIMUM_PASSIVE_REDUCTION_PERCENT
            ),
            "minimum_additional_reduction_points": (
                MINIMUM_ADDITIONAL_REDUCTION_POINTS
            ),
        },
        "resonance": {
            "omega_1": SYSTEM.omega_1,
            "coarse_peak_index": coarse_peak_index,
            "coarse_peak_ratio": coarse_peak_ratio,
            "fine_peak_index": fine_peak_index,
            "omega_r_ratio": omega_r_ratio,
            "omega_r": omega_r,
            "peak_amplitude": float(fine_amplitudes[fine_peak_index]),
            "frequency_ratios": [point["ratio"] for point in frequency_points],
            "omegas": [point["omega"] for point in frequency_points],
            "steady_amplitudes": [
                point["steady_amplitude"] for point in frequency_points
            ],
            "steady_error": [
                point["steady_error"] for point in frequency_points
            ],
            "cycle_amplitudes": [
                point["cycle_amplitudes"] for point in frequency_points
            ],
        },
        "passive_preload": {
            "preload_values": REPAIR_PRELOAD_VALUES.tolist(),
            "steady_amplitudes": passive_objectives.tolist(),
            "steady_error": passive_errors.tolist(),
            "cycle_amplitudes": passive_cycles.tolist(),
            "n_star_index": n_star_index,
            "n_star": n_star,
            "n_star_amplitude": passive_amplitude,
            "n_star_steady_error": float(passive_errors[n_star_index]),
            "interior": gates["passive_optimum_interior"],
            "local_minimum": gates["passive_local_minimum"],
        },
        "phase_sweep": phase_results,
        "gates": gates,
        "gate_0_repair": {
            "result": "PASS" if passed else "FAIL",
            "phase_sweeps_run": phase_results is not None,
            "failed_checks": failed_checks,
            "skipped_checks": skipped_checks,
            "failure_classes": failure_classes,
            "reason": reason,
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    _plot(results)

    print("## Gate 0-R")
    print(f"omega_1={SYSTEM.omega_1:.12g}")
    print(f"omega_r={omega_r:.12g} ({omega_r_ratio:.6g} omega_1)")
    for preload, objective, steady_error in zip(
        REPAIR_PRELOAD_VALUES,
        passive_objectives,
        passive_errors,
        strict=True,
    ):
        print(
            f"N={preload:.3f} A_ss={objective:.12g} "
            f"steady_error={steady_error:.6g}"
        )
    print(
        f"N_star={n_star:.12g} interior={gates['passive_optimum_interior']} "
        f"local_minimum={gates['passive_local_minimum']}"
    )
    if phase_results is not None:
        print(
            f"best_1omega_phase={phase_results['best_1omega_phase']:.6g} "
            f"A_ss={phase_results['best_1omega_amplitude']:.12g} "
            f"improvement={phase_results['best_1omega_improvement_percent']:.6g}% "
            f"steady_error={phase_results['best_1omega_steady_error']:.6g}"
        )
        print(
            f"best_2omega_phase={phase_results['best_2omega_phase']:.6g} "
            f"A_ss={phase_results['best_2omega_amplitude']:.12g} "
            f"improvement={phase_results['best_2omega_improvement_percent']:.6g}% "
            f"steady_error={phase_results['best_2omega_steady_error']:.6g}"
        )
        print(
            "2omega_vs_1omega_improvement_points="
            f"{phase_results['2omega_vs_1omega_improvement_points']:.6g}"
        )
    else:
        print("harmonic_phase_sweeps=SKIPPED")
    print(f"Gate 0-R: {results['gate_0_repair']['result']}")
    print(f"reason={reason}")
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
