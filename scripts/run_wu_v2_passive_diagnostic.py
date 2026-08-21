"""Diagnose fixed-frequency versus per-preload resonance comparators."""

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
    DIAGNOSTIC_NUM_PERIODS,
    DIAGNOSTIC_PRELOAD_VALUES,
    FORCING_AMPLITUDE,
    STEADY_STATE_TOLERANCE,
    SYSTEM,
    constant_preload,
    diagnostic_steady_state_metrics,
    frf_peak_indices,
    simulate_preload_bank,
)


OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_passive_diagnostic"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "passive_frf_diagnostic.png"
REPAIR_RESULTS_PATH = ROOT / "outputs/wu_v2_gate0_repair/results.json"

FRAME_COLOR = "#20242A"
FIXED_COLOR = "#A55C45"
PEAK_COLOR = "#2D6388"
HIGHLIGHT_COLORS = {
    0: "#2D6388",
    2: "#A55C45",
    4: "#3F7A67",
    6: "#7B5485",
}
SECONDARY_STYLES = {1: "--", 3: "-.", 5: ":"}


def _constant_preload_bank() -> np.ndarray:
    return np.concatenate(
        [
            constant_preload(
                preload, num_periods=DIAGNOSTIC_NUM_PERIODS
            )
            for preload in DIAGNOSTIC_PRELOAD_VALUES
        ],
        axis=0,
    )


def _evaluate_frequency(
    omega: float, preload_bank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outputs = simulate_preload_bank(omega, preload_bank)
    if not all(np.all(np.isfinite(np.asarray(output))) for output in outputs):
        raise RuntimeError(f"Non-finite mechanics output at omega={omega}")
    objectives, steady_errors, _ = diagnostic_steady_state_metrics(outputs[0])
    if not (
        np.all(np.isfinite(objectives))
        and np.all(np.isfinite(steady_errors))
    ):
        raise RuntimeError(f"Non-finite diagnostic metric at omega={omega}")
    return objectives, steady_errors


def _load_fixed_frequency() -> tuple[float, float]:
    repair = json.loads(REPAIR_RESULTS_PATH.read_text())
    return (
        float(repair["resonance"]["omega_r"]),
        float(repair["resonance"]["omega_r_ratio"]),
    )


def _classify(
    peak_indices: np.ndarray,
    peak_steady_errors: np.ndarray,
    fixed_best_index: int,
    peak_best_index: int,
) -> tuple[str, str]:
    peak_at_boundary = (peak_indices == 0) | (
        peak_indices == len(COARSE_FREQUENCY_RATIOS) - 1
    )
    if np.any(peak_at_boundary):
        return "inconclusive", "FRF range insufficient for diagnosis."

    unsteady_count = int(
        np.count_nonzero(peak_steady_errors > STEADY_STATE_TOLERANCE)
    )
    if unsteady_count >= 4:
        return (
            "inconclusive",
            f"Steady-state evidence insufficient at {unsteady_count}/7 FRF peaks.",
        )

    peak_best_interior = (
        0 < peak_best_index < len(DIAGNOSTIC_PRELOAD_VALUES) - 1
    )
    if peak_best_interior and fixed_best_index != peak_best_index:
        return (
            "fixed_frequency_comparator_confounded",
            "Fixed-frequency and own-peak comparators select different preloads, "
            "while the own-peak optimum is interior.",
        )
    if not peak_best_interior:
        return (
            "no_interior_passive_optimum",
            "The per-preload resonance-peak objective is minimized on the "
            "preload boundary.",
        )
    return (
        "interior_optimum_confirmed",
        "Both 24-period comparators select the same interior preload; the old "
        "Gate 0-R boundary result was affected by transient horizon.",
    )


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
            "legend.fontsize": 7.5,
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
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
        constrained_layout=True,
    )

    frequency_ratios = np.asarray(results["configuration"]["frequency_ratios"])
    for index, entry in enumerate(results["per_preload"]):
        highlighted = index in HIGHLIGHT_COLORS
        color = HIGHLIGHT_COLORS.get(index, "#B4BAC2")
        linestyle = "-" if highlighted else SECONDARY_STYLES[index]
        linewidth = 1.8 if highlighted else 1.15
        axes[0].plot(
            frequency_ratios,
            entry["frf_amplitudes"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"N={entry['preload']:.2f}",
            zorder=2 if highlighted else 1,
        )
        axes[0].scatter(
            [entry["peak_ratio"]],
            [entry["peak_amplitude"]],
            color=color,
            edgecolor="white",
            linewidth=0.45,
            s=25 if highlighted else 18,
            zorder=3,
        )
    axes[0].set(
        xlabel=r"Frequency ratio $\omega/\omega_1$",
        ylabel="Steady amplitude",
        xlim=(float(frequency_ratios[0]), float(frequency_ratios[-1])),
    )
    axes[0].legend(loc="best", ncol=2, handlelength=2.4, columnspacing=0.9)

    comparison = results["comparison"]
    preloads = np.asarray(results["configuration"]["preload_values"])
    axes[1].plot(
        preloads,
        comparison["fixed_amplitudes"],
        color=FIXED_COLOR,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Fixed frequency",
    )
    axes[1].plot(
        preloads,
        comparison["peak_amplitudes"],
        color=PEAK_COLOR,
        marker="s",
        markersize=4.2,
        linewidth=1.8,
        label="Own resonance peak",
    )
    axes[1].set(xlabel="Preload N", ylabel="Steady amplitude")
    axes[1].legend(loc="best")

    for label, axis in zip(("a", "b"), axes, strict=True):
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


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    omega_fixed, omega_fixed_ratio = _load_fixed_frequency()
    preload_bank = _constant_preload_bank()

    frf_objective_columns = []
    frf_error_columns = []
    for ratio in COARSE_FREQUENCY_RATIOS:
        objective, steady_error = _evaluate_frequency(
            float(ratio * SYSTEM.omega_1), preload_bank
        )
        frf_objective_columns.append(objective)
        frf_error_columns.append(steady_error)
    frf_amplitudes = np.column_stack(frf_objective_columns)
    frf_steady_errors = np.column_stack(frf_error_columns)

    fixed_amplitudes, fixed_steady_errors = _evaluate_frequency(
        omega_fixed, preload_bank
    )
    peak_indices = frf_peak_indices(frf_amplitudes)
    rows = np.arange(len(DIAGNOSTIC_PRELOAD_VALUES))
    peak_amplitudes = frf_amplitudes[rows, peak_indices]
    peak_steady_errors = frf_steady_errors[rows, peak_indices]
    peak_ratios = COARSE_FREQUENCY_RATIOS[peak_indices]
    peak_omegas = peak_ratios * SYSTEM.omega_1
    peak_at_boundary = (peak_indices == 0) | (
        peak_indices == len(COARSE_FREQUENCY_RATIOS) - 1
    )

    fixed_best_index = int(np.argmin(fixed_amplitudes))
    peak_best_index = int(np.argmin(peak_amplitudes))
    classification, reason = _classify(
        peak_indices,
        peak_steady_errors,
        fixed_best_index,
        peak_best_index,
    )
    unsteady_peak_count = int(
        np.count_nonzero(peak_steady_errors > STEADY_STATE_TOLERANCE)
    )

    per_preload = []
    for index, preload in enumerate(DIAGNOSTIC_PRELOAD_VALUES):
        per_preload.append(
            {
                "preload": float(preload),
                "frf_amplitudes": frf_amplitudes[index].tolist(),
                "frf_steady_errors": frf_steady_errors[index].tolist(),
                "peak_index": int(peak_indices[index]),
                "peak_ratio": float(peak_ratios[index]),
                "peak_omega": float(peak_omegas[index]),
                "peak_amplitude": float(peak_amplitudes[index]),
                "peak_steady_error": float(peak_steady_errors[index]),
                "peak_at_frequency_boundary": bool(peak_at_boundary[index]),
                "fixed_amplitude": float(fixed_amplitudes[index]),
                "fixed_steady_error": float(fixed_steady_errors[index]),
            }
        )

    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": 100,
            "previous_window_cycles": [17, 18, 19, 20],
            "final_window_cycles": [21, 22, 23, 24],
            "steady_tolerance": STEADY_STATE_TOLERANCE,
            "preload_values": DIAGNOSTIC_PRELOAD_VALUES.tolist(),
            "frequency_ratios": COARSE_FREQUENCY_RATIOS.tolist(),
            "omegas": (COARSE_FREQUENCY_RATIOS * SYSTEM.omega_1).tolist(),
            "omega_1": SYSTEM.omega_1,
            "omega_fixed": omega_fixed,
            "omega_fixed_ratio": omega_fixed_ratio,
            "omega_fixed_source": "outputs/wu_v2_gate0_repair/results.json",
        },
        "per_preload": per_preload,
        "comparison": {
            "fixed_amplitudes": fixed_amplitudes.tolist(),
            "fixed_steady_errors": fixed_steady_errors.tolist(),
            "fixed_best_index": fixed_best_index,
            "fixed_best_preload": float(
                DIAGNOSTIC_PRELOAD_VALUES[fixed_best_index]
            ),
            "peak_amplitudes": peak_amplitudes.tolist(),
            "peak_steady_errors": peak_steady_errors.tolist(),
            "peak_best_index": peak_best_index,
            "peak_best_preload": float(
                DIAGNOSTIC_PRELOAD_VALUES[peak_best_index]
            ),
            "peak_best_interior": bool(
                0 < peak_best_index < len(DIAGNOSTIC_PRELOAD_VALUES) - 1
            ),
            "comparators_select_different_preloads": bool(
                fixed_best_index != peak_best_index
            ),
        },
        "diagnosis": {
            "classification": classification,
            "reason": reason,
            "all_dynamics_finite": True,
            "any_peak_at_frequency_boundary": bool(np.any(peak_at_boundary)),
            "unsteady_peak_count": unsteady_peak_count,
            "peak_ratio_min": float(np.min(peak_ratios)),
            "peak_ratio_max": float(np.max(peak_ratios)),
            "peak_ratio_span": float(np.ptp(peak_ratios)),
            "fixed_best_changed_from_gate0_repair": bool(
                fixed_best_index != 0
            ),
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    _plot(results)

    print("## Passive FRF comparator diagnosis")
    print(f"omega_1={SYSTEM.omega_1:.12g}")
    print(f"omega_fixed={omega_fixed:.12g} ({omega_fixed_ratio:.6g} omega_1)")
    for entry in per_preload:
        print(
            f"N={entry['preload']:.2f} "
            f"peak_ratio={entry['peak_ratio']:.6g} "
            f"A_peak={entry['peak_amplitude']:.12g} "
            f"peak_steady_error={entry['peak_steady_error']:.6g} "
            f"A_fixed={entry['fixed_amplitude']:.12g} "
            f"fixed_steady_error={entry['fixed_steady_error']:.6g} "
            f"boundary={entry['peak_at_frequency_boundary']}"
        )
    print(f"classification={classification}")
    print(f"reason={reason}")
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
