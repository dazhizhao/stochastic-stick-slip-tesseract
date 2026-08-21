"""Run the pre-registered Wu-style V2 engineering-authority gate."""

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
    PRELOAD_VALUES,
    REFERENCE_PRELOAD,
    STEADY_STATE_TOLERANCE,
    SYSTEM,
    constant_preload,
    harmonic_preload,
    simulate_preload_bank,
    steady_state_metrics,
)


OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_gate0"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "gate0_summary.png"

FRAME_COLOR = "#20242A"
NEUTRAL_COLOR = "#666B73"
FIRST_COLOR = "#A36A3F"
SECOND_COLOR = "#27628D"
ACCENT_COLOR = "#8F355E"


def _response_for_frequency_ratio(ratio: float):
    omega = float(ratio * SYSTEM.omega_1)
    outputs = simulate_preload_bank(
        omega, constant_preload(REFERENCE_PRELOAD)
    )
    objective, convergence, amplitudes = steady_state_metrics(outputs[0])
    return {
        "ratio": float(ratio),
        "omega": omega,
        "steady_amplitude": float(objective[0]),
        "convergence": float(convergence[0]),
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

    phase = results["phase_sweep"]
    axes[2].axhline(
        passive["n_star_amplitude"],
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.4,
        label="Passive",
    )
    axes[2].plot(
        phase["phases"],
        phase["one_omega_amplitudes"],
        color=FIRST_COLOR,
        linewidth=1.7,
        label=r"$1\omega$",
    )
    axes[2].plot(
        phase["phases"],
        phase["two_omega_amplitudes"],
        color=SECOND_COLOR,
        linewidth=2.0,
        label=r"$2\omega$",
    )
    axes[2].set(
        xlabel="Phase (rad)",
        ylabel="Steady amplitude",
        xlim=(0.0, 2.0 * np.pi),
        xticks=np.arange(0.0, 2.0 * np.pi + 1e-12, np.pi / 2.0),
        xticklabels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axes[2].legend(loc="best")

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


def _write_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")


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
            [constant_preload(value) for value in PRELOAD_VALUES], axis=0
        ),
    )
    passive_objectives, passive_convergence, passive_cycles = (
        steady_state_metrics(passive_outputs[0])
    )
    n_star_index = int(np.argmin(passive_objectives))
    n_star = float(PRELOAD_VALUES[n_star_index])

    one_preload = harmonic_preload(n_star, omega_r, 1, PHASES)
    two_preload = harmonic_preload(n_star, omega_r, 2, PHASES)
    phase_outputs = simulate_preload_bank(
        omega_r, np.concatenate((one_preload, two_preload), axis=0)
    )
    phase_objectives, phase_convergence, phase_cycles = steady_state_metrics(
        phase_outputs[0]
    )
    one_objectives = phase_objectives[: len(PHASES)]
    two_objectives = phase_objectives[len(PHASES) :]
    one_convergence = phase_convergence[: len(PHASES)]
    two_convergence = phase_convergence[len(PHASES) :]
    one_cycles = phase_cycles[: len(PHASES)]
    two_cycles = phase_cycles[len(PHASES) :]
    one_best_index = int(np.argmin(one_objectives))
    two_best_index = int(np.argmin(two_objectives))

    passive_amplitude = float(passive_objectives[n_star_index])
    one_best_amplitude = float(one_objectives[one_best_index])
    two_best_amplitude = float(two_objectives[two_best_index])
    one_reduction = 100.0 * (
        passive_amplitude - one_best_amplitude
    ) / passive_amplitude
    two_reduction = 100.0 * (
        passive_amplitude - two_best_amplitude
    ) / passive_amplitude
    additional_reduction = two_reduction - one_reduction

    gates = {
        "all_values_finite": bool(
            np.all(np.isfinite(coarse_amplitudes))
            and np.all(np.isfinite(fine_amplitudes))
            and np.all(np.isfinite(passive_objectives))
            and np.all(np.isfinite(phase_objectives))
        ),
        "coarse_peak_interior": 0 < coarse_peak_index < len(coarse) - 1,
        "fine_peak_interior": 0 < fine_peak_index < len(fine) - 1,
        "passive_optimum_interior": 0 < n_star_index < len(PRELOAD_VALUES) - 1,
        "passive_local_minimum": bool(
            0 < n_star_index < len(PRELOAD_VALUES) - 1
            and passive_objectives[n_star_index]
            < passive_objectives[n_star_index - 1]
            and passive_objectives[n_star_index]
            < passive_objectives[n_star_index + 1]
        ),
        "passive_steady": bool(
            passive_convergence[n_star_index] <= STEADY_STATE_TOLERANCE
        ),
        "best_1omega_steady": bool(
            one_convergence[one_best_index] <= STEADY_STATE_TOLERANCE
        ),
        "best_2omega_steady": bool(
            two_convergence[two_best_index] <= STEADY_STATE_TOLERANCE
        ),
        "2omega_reduction_at_least_5_percent": bool(
            two_reduction >= MINIMUM_PASSIVE_REDUCTION_PERCENT
        ),
        "2omega_advantage_at_least_2_points": bool(
            additional_reduction >= MINIMUM_ADDITIONAL_REDUCTION_POINTS
        ),
        "2omega_has_better_and_worse_phases": bool(
            np.min(two_objectives) < passive_amplitude
            and np.max(two_objectives) > passive_amplitude
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    authority_gates = {
        "2omega_reduction_at_least_5_percent",
        "2omega_advantage_at_least_2_points",
    }
    if not failed:
        reason = "All pre-registered Wu-style engineering-authority checks passed."
    elif set(failed).issubset(authority_gates):
        reason = (
            "The pre-registered amplitude-authority threshold was not met; "
            "this does not by itself invalidate the Wu-style design."
        )
    else:
        reason = "Gate 0 failed: " + ", ".join(failed) + "."

    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "reference_preload": REFERENCE_PRELOAD,
            "num_cycles": 8,
            "steps_per_cycle": 100,
            "steady_cycles": [5, 6, 7, 8],
            "convergence_windows": [[5, 6], [7, 8]],
            "convergence_tolerance": STEADY_STATE_TOLERANCE,
            "coarse_frequency_ratios": COARSE_FREQUENCY_RATIOS.tolist(),
            "fine_half_width": FINE_FREQUENCY_HALF_WIDTH,
            "fine_points": FINE_FREQUENCY_POINTS,
            "preload_values": PRELOAD_VALUES.tolist(),
            "phase_grid": PHASES.tolist(),
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
            "convergence": [point["convergence"] for point in frequency_points],
            "cycle_amplitudes": [
                point["cycle_amplitudes"] for point in frequency_points
            ],
        },
        "passive_preload": {
            "preload_values": PRELOAD_VALUES.tolist(),
            "steady_amplitudes": passive_objectives.tolist(),
            "convergence": passive_convergence.tolist(),
            "cycle_amplitudes": passive_cycles.tolist(),
            "n_star_index": n_star_index,
            "n_star": n_star,
            "n_star_amplitude": passive_amplitude,
        },
        "phase_sweep": {
            "phases": PHASES.tolist(),
            "one_omega_amplitudes": one_objectives.tolist(),
            "two_omega_amplitudes": two_objectives.tolist(),
            "one_omega_convergence": one_convergence.tolist(),
            "two_omega_convergence": two_convergence.tolist(),
            "one_omega_cycle_amplitudes": one_cycles.tolist(),
            "two_omega_cycle_amplitudes": two_cycles.tolist(),
            "best_1omega_index": one_best_index,
            "best_1omega_phase": float(PHASES[one_best_index]),
            "best_1omega_amplitude": one_best_amplitude,
            "best_1omega_reduction_percent": one_reduction,
            "best_2omega_index": two_best_index,
            "best_2omega_phase": float(PHASES[two_best_index]),
            "best_2omega_amplitude": two_best_amplitude,
            "best_2omega_reduction_percent": two_reduction,
            "additional_reduction_points": additional_reduction,
            "worst_2omega_amplitude": float(np.max(two_objectives)),
        },
        "gates": gates,
        "gate_0": {
            "result": "PASS" if not failed else "FAIL",
            "failed_checks": failed,
            "reason": reason,
        },
    }
    _write_results(results)
    _plot(results)

    print("## Gate 0")
    print(f"omega_1={SYSTEM.omega_1:.12g}")
    print(f"omega_r={omega_r:.12g} ({omega_r_ratio:.6g} omega_1)")
    print(f"N_star={n_star:.12g}")
    print(f"passive_A_ss={passive_amplitude:.12g}")
    print(
        f"best_1omega_A_ss={one_best_amplitude:.12g} "
        f"reduction={one_reduction:.6g}% phase={PHASES[one_best_index]:.6g}"
    )
    print(
        f"best_2omega_A_ss={two_best_amplitude:.12g} "
        f"reduction={two_reduction:.6g}% phase={PHASES[two_best_index]:.6g}"
    )
    print(f"additional_reduction_points={additional_reduction:.6g}")
    print(f"Gate 0: {results['gate_0']['result']}")
    print(f"reason={reason}")
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
