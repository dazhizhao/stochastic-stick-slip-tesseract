"""Optimize a shared hard Markov controller on the Wu 2019 benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from wu2019.controller import constant_normal_force, harmonic_normal_force
from wu2019.dynamics import DEFAULT_SETTINGS, dense_frequency_grid
from wu2019.markov import (
    FD_EPSILON,
    NUM_COEFFICIENTS,
    crn_centered_finite_difference,
    direct_ad_objective_and_gradient,
    evaluate_markov,
    uniform_bank,
)
from wu2019.newmark import simulate_summary_batch, simulate_trajectory


TRAIN_BASE_SEED = 20260820
TRACE_BASE_SEED = 20261820
EVAL_BASE_SEED = 20270820
NUM_TRAIN_REALIZATIONS = 4
NUM_TRACE_REALIZATIONS = 4
NUM_EVAL_REALIZATIONS = 32
NUM_UPDATES = 200
LEARNING_RATE = 0.01
TRACE_INTERVAL = 10
DIRECT_AD_ZERO_ATOL = 1e-12

TRAIN_OMEGAS = np.arange(190.0, 220.0 + 1e-12, 1.0)
EVAL_OMEGAS = dense_frequency_grid()

OUTPUT_DIRECTORY = EXPERIMENT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIRECTORY / "markov_summary.json"
TRACE_FIGURE_PATH = OUTPUT_DIRECTORY / "markov_optimization_trace.png"
FRF_FIGURE_PATH = OUTPUT_DIRECTORY / "markov_final_frf.png"
HISTORY_FIGURE_PATH = OUTPUT_DIRECTORY / "markov_representative_history.png"

FRAME_COLOR = "#20242A"
NEUTRAL_COLOR = "#666B73"
INITIAL_COLOR = "#A8ADB3"
MARKOV_COLOR = "#27628D"
WU_COLOR = "#C47A32"


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _style_axis(axis) -> None:
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
    axis.grid(False)


def _reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def _peak(omegas, amplitudes) -> dict[str, float]:
    index = int(np.argmax(amplitudes))
    return {
        "omega": float(omegas[index]),
        "amplitude_m": float(amplitudes[index]),
        "amplitude_mm": float(1e3 * amplitudes[index]),
    }


def _gradient_diagnostics(coefficients, uniforms) -> dict:
    direct_value, direct_gradient = direct_ad_objective_and_gradient(
        coefficients, TRAIN_OMEGAS, uniforms
    )
    finite_difference = crn_centered_finite_difference(
        coefficients,
        TRAIN_OMEGAS,
        uniforms,
        epsilon=FD_EPSILON,
    )
    diagnostics = {
        "objective_m": direct_value,
        "direct_ad_gradient": direct_gradient.tolist(),
        "direct_ad_linf": float(np.max(np.abs(direct_gradient))),
        "epsilon": finite_difference.epsilon,
        "crn_fd_gradient": finite_difference.gradient.tolist(),
        "crn_fd_l2": float(np.linalg.norm(finite_difference.gradient)),
        "plus_objectives_m": finite_difference.plus_objectives.tolist(),
        "minus_objectives_m": finite_difference.minus_objectives.tolist(),
        "mode_difference_counts": (
            finite_difference.mode_difference_counts.tolist()
        ),
    }
    diagnostics["gates"] = {
        "direct_ad_zero": (
            diagnostics["direct_ad_linf"] <= DIRECT_AD_ZERO_ATOL
        ),
        "crn_fd_finite": bool(
            np.all(np.isfinite(finite_difference.gradient))
        ),
        "crn_fd_nonzero": diagnostics["crn_fd_l2"] > 0.0,
        "mode_history_changed": bool(
            np.any(finite_difference.mode_difference_counts > 0)
        ),
    }
    return diagnostics


def _train(trace_uniforms):
    coefficients = torch.nn.Parameter(
        torch.zeros(NUM_COEFFICIENTS, dtype=torch.float64)
    )
    optimizer = torch.optim.Adam([coefficients], lr=LEARNING_RATE)
    coefficient_history = np.empty(
        (NUM_UPDATES + 1, NUM_COEFFICIENTS), dtype=np.float64
    )
    sampled_objectives = np.empty(NUM_UPDATES, dtype=np.float64)
    gradient_norms = np.empty(NUM_UPDATES, dtype=np.float64)
    trace_iterations = np.arange(
        0, NUM_UPDATES + 1, TRACE_INTERVAL, dtype=np.int64
    )
    trace_objectives = np.empty(len(trace_iterations), dtype=np.float64)
    coefficient_history[0] = coefficients.detach().numpy()
    trace_objectives[0] = evaluate_markov(
        coefficient_history[0], TRAIN_OMEGAS, trace_uniforms
    ).objective

    trace_index = 1
    start = time.perf_counter()
    for iteration in range(NUM_UPDATES):
        coefficient_array = coefficients.detach().numpy().copy()
        train_uniforms = uniform_bank(
            NUM_TRAIN_REALIZATIONS,
            TRAIN_BASE_SEED + iteration * NUM_TRAIN_REALIZATIONS,
        )
        sampled_objectives[iteration] = evaluate_markov(
            coefficient_array, TRAIN_OMEGAS, train_uniforms
        ).objective
        finite_difference = crn_centered_finite_difference(
            coefficient_array,
            TRAIN_OMEGAS,
            train_uniforms,
            epsilon=FD_EPSILON,
        )
        gradient = finite_difference.gradient
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("CRN-FD gradient is non-finite")
        gradient_norms[iteration] = np.linalg.norm(gradient)
        optimizer.zero_grad(set_to_none=True)
        coefficients.grad = torch.from_numpy(gradient.copy())
        optimizer.step()
        coefficient_history[iteration + 1] = coefficients.detach().numpy()
        if (iteration + 1) % TRACE_INTERVAL == 0:
            trace_objectives[trace_index] = evaluate_markov(
                coefficient_history[iteration + 1],
                TRAIN_OMEGAS,
                trace_uniforms,
            ).objective
            print(
                f"update={iteration + 1:03d} "
                f"sampled={1e3 * sampled_objectives[iteration]:.9g} mm "
                f"trace={1e3 * trace_objectives[trace_index]:.9g} mm "
                f"gradient_l2={gradient_norms[iteration]:.9g}",
                flush=True,
            )
            trace_index += 1

    arrays = (
        coefficient_history,
        sampled_objectives,
        gradient_norms,
        trace_objectives,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("optimization history is non-finite")
    return {
        "coefficient_history": coefficient_history,
        "sampled_objectives": sampled_objectives,
        "gradient_norms": gradient_norms,
        "trace_iterations": trace_iterations,
        "trace_objectives": trace_objectives,
        "runtime_seconds": time.perf_counter() - start,
    }


def _plot_trace(history) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.6, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )
    iterations = np.arange(1, NUM_UPDATES + 1)
    axes[0].scatter(
        iterations,
        1e3 * history["sampled_objectives"],
        s=9,
        color=INITIAL_COLOR,
        alpha=0.65,
        linewidths=0.0,
        label="Sampled training bank",
    )
    axes[0].plot(
        history["trace_iterations"],
        1e3 * history["trace_objectives"],
        color=MARKOV_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Fixed trace bank",
    )
    axes[0].set_ylabel("Peak response (mm)")
    axes[0].legend(loc="best")
    axes[1].plot(
        iterations,
        history["gradient_norms"],
        color=WU_COLOR,
        linewidth=1.6,
    )
    axes[1].set_xlabel("Adam update")
    axes[1].set_ylabel("CRN-FD gradient norm")
    for axis in axes:
        _style_axis(axis)
    figure.savefig(TRACE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_frfs(constant, wu, initial, optimized) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    axis.plot(
        EVAL_OMEGAS,
        1e3 * constant,
        color=NEUTRAL_COLOR,
        linewidth=2.0,
        label="Constant N = 40 N",
    )
    axis.plot(
        EVAL_OMEGAS,
        1e3 * wu,
        color=WU_COLOR,
        linewidth=2.1,
        label="Wu continuous",
    )
    axis.plot(
        EVAL_OMEGAS,
        1e3 * initial,
        color=INITIAL_COLOR,
        linewidth=1.7,
        linestyle="--",
        label="Hard Markov initial",
    )
    axis.plot(
        EVAL_OMEGAS,
        1e3 * optimized,
        color=MARKOV_COLOR,
        linewidth=2.2,
        label="Hard Markov optimized",
    )
    axis.set_xlabel("Excitation frequency (rad s$^{-1}$)")
    axis.set_ylabel("Response amplitude (mm)")
    axis.legend(loc="best")
    _style_axis(axis)
    figure.savefig(FRF_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_representative_history(omega, preload, realization) -> dict:
    trajectory = simulate_trajectory(omega, preload, DEFAULT_SETTINGS)
    window_steps = 2 * DEFAULT_SETTINGS.steps_per_period
    time = trajectory.time[-window_steps:]
    time_ms = 1e3 * (time - time[0])
    displacement_mm = 1e3 * trajectory.displacement[-window_steps:]
    normal_force = trajectory.normal_force[-window_steps:]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.6, 4.8),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        time_ms,
        displacement_mm,
        color=MARKOV_COLOR,
        linewidth=1.8,
    )
    axes[0].set_ylabel("Displacement (mm)")
    axes[1].step(
        time_ms,
        normal_force,
        where="post",
        color=WU_COLOR,
        linewidth=1.7,
    )
    axes[1].set_xlabel("Time within window (ms)")
    axes[1].set_ylabel("Normal force (N)")
    axes[1].set_yticks([30.0, 50.0])
    axes[1].set_ylim(26.0, 54.0)
    for axis in axes:
        _style_axis(axis)
    figure.savefig(HISTORY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "omega": float(omega),
        "realization": int(realization),
        "window_periods": 2,
    }


def main() -> int:
    _configure_plotting()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    zero_coefficients = np.zeros(NUM_COEFFICIENTS, dtype=np.float64)
    initial_train_uniforms = uniform_bank(
        NUM_TRAIN_REALIZATIONS, TRAIN_BASE_SEED
    )
    gradient_diagnostics = _gradient_diagnostics(
        zero_coefficients, initial_train_uniforms
    )
    print(json.dumps(gradient_diagnostics, indent=2), flush=True)
    if not all(gradient_diagnostics["gates"].values()):
        raise SystemExit("initial hard-gradient gate failed")

    trace_uniforms = uniform_bank(
        NUM_TRACE_REALIZATIONS, TRACE_BASE_SEED
    )
    history = _train(trace_uniforms)
    optimized_coefficients = history["coefficient_history"][-1]

    evaluation_uniforms = uniform_bank(
        NUM_EVAL_REALIZATIONS, EVAL_BASE_SEED
    )
    initial = evaluate_markov(
        zero_coefficients, EVAL_OMEGAS, evaluation_uniforms
    )
    optimized = evaluate_markov(
        optimized_coefficients, EVAL_OMEGAS, evaluation_uniforms
    )
    constant = simulate_summary_batch(
        EVAL_OMEGAS,
        constant_normal_force(40.0, DEFAULT_SETTINGS),
        DEFAULT_SETTINGS,
    )
    wu = simulate_summary_batch(
        EVAL_OMEGAS,
        harmonic_normal_force(
            40.0, 10.0, 2, 4.4, DEFAULT_SETTINGS
        ),
        DEFAULT_SETTINGS,
    )
    constant_peak = _peak(EVAL_OMEGAS, constant.amplitude)
    wu_peak = _peak(EVAL_OMEGAS, wu.amplitude)
    initial_peak = _peak(EVAL_OMEGAS, initial.frequency_means)
    optimized_peak = _peak(EVAL_OMEGAS, optimized.frequency_means)
    reference = constant_peak["amplitude_m"]

    comparison = [
        {
            "controller": "Constant N=40",
            "normal_force": "40 N",
            "peak": constant_peak,
            "reduction_vs_constant_percent": 0.0,
        },
        {
            "controller": "Wu continuous",
            "normal_force": "continuous 30-50 N",
            "peak": wu_peak,
            "reduction_vs_constant_percent": _reduction(
                reference, wu_peak["amplitude_m"]
            ),
        },
        {
            "controller": "Hard Markov initial",
            "normal_force": "{30,50} N",
            "peak": initial_peak,
            "reduction_vs_constant_percent": _reduction(
                reference, initial_peak["amplitude_m"]
            ),
        },
        {
            "controller": "Hard Markov optimized",
            "normal_force": "{30,50} N",
            "peak": optimized_peak,
            "reduction_vs_constant_percent": _reduction(
                reference, optimized_peak["amplitude_m"]
            ),
        },
    ]

    optimized_peak_index = int(np.argmax(optimized.frequency_means))
    representative = int(
        np.argmin(
            np.abs(
                optimized.amplitudes[optimized_peak_index]
                - optimized.frequency_means[optimized_peak_index]
            )
        )
    )
    representative_metadata = _plot_representative_history(
        EVAL_OMEGAS[optimized_peak_index],
        optimized.preload[representative],
        representative,
    )
    _plot_trace(history)
    _plot_frfs(
        constant.amplitude,
        wu.amplitude,
        initial.frequency_means,
        optimized.frequency_means,
    )

    scientific_outcome = {
        "optimized_better_than_initial": (
            optimized.objective < initial.objective
        ),
        "optimized_reduction_vs_constant_positive": (
            optimized.objective < reference
        ),
        "initial_to_optimized_improvement_percent": _reduction(
            initial.objective, optimized.objective
        ),
        "optimized_reduction_vs_constant_percent": _reduction(
            reference, optimized.objective
        ),
        "gap_to_wu_reduction_percentage_points": (
            _reduction(reference, wu_peak["amplitude_m"])
            - _reduction(reference, optimized.objective)
        ),
        "meets_18_percent_phase3_science_gate": (
            _reduction(reference, optimized.objective) >= 18.0
        ),
    }
    summary = {
        "settings": {
            "steps_per_period": DEFAULT_SETTINGS.steps_per_period,
            "num_periods": DEFAULT_SETTINGS.num_periods,
            "measurement_periods": DEFAULT_SETTINGS.measurement_periods,
            "train_frequencies_rad_s": TRAIN_OMEGAS.tolist(),
            "evaluation_frequencies_rad_s": EVAL_OMEGAS.tolist(),
            "train_realizations": NUM_TRAIN_REALIZATIONS,
            "trace_realizations": NUM_TRACE_REALIZATIONS,
            "evaluation_realizations": NUM_EVAL_REALIZATIONS,
            "train_base_seed": TRAIN_BASE_SEED,
            "trace_base_seed": TRACE_BASE_SEED,
            "evaluation_base_seed": EVAL_BASE_SEED,
            "fd_epsilon": FD_EPSILON,
            "adam_learning_rate": LEARNING_RATE,
            "num_updates": NUM_UPDATES,
        },
        "gradient_diagnostics": gradient_diagnostics,
        "initial_coefficients": zero_coefficients.tolist(),
        "optimized_coefficients": optimized_coefficients.tolist(),
        "coefficient_history": history["coefficient_history"].tolist(),
        "sampled_training_objective_m": (
            history["sampled_objectives"].tolist()
        ),
        "gradient_l2": history["gradient_norms"].tolist(),
        "trace_iterations": history["trace_iterations"].tolist(),
        "trace_objective_m": history["trace_objectives"].tolist(),
        "training_runtime_seconds": history["runtime_seconds"],
        "evaluation": {
            "initial_frequency_mean_amplitude_m": (
                initial.frequency_means.tolist()
            ),
            "optimized_frequency_mean_amplitude_m": (
                optimized.frequency_means.tolist()
            ),
            "initial_transition_counts": (
                initial.transition_counts.astype(int).tolist()
            ),
            "optimized_transition_counts": (
                optimized.transition_counts.astype(int).tolist()
            ),
            "initial_high_mode_fraction": (
                initial.high_mode_fraction.tolist()
            ),
            "optimized_high_mode_fraction": (
                optimized.high_mode_fraction.tolist()
            ),
        },
        "comparison": comparison,
        "representative_history": representative_metadata,
        "implementation_pass": True,
        "scientific_outcome": scientific_outcome,
        "outputs": {
            "optimization_trace": str(TRACE_FIGURE_PATH),
            "final_frf": str(FRF_FIGURE_PATH),
            "representative_history": str(HISTORY_FIGURE_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\n## Independent dense evaluation", flush=True)
    for row in comparison:
        print(
            f"{row['controller']}: "
            f"peak={row['peak']['amplitude_mm']:.9g} mm "
            f"at {row['peak']['omega']:.6g} rad/s, "
            f"reduction={row['reduction_vs_constant_percent']:.6g}%",
            flush=True,
        )
    print(f"optimized_coefficients={optimized_coefficients.tolist()}")
    print(f"scientific_outcome={scientific_outcome}", flush=True)
    print(f"summary={SUMMARY_PATH}", flush=True)
    print(f"figure={TRACE_FIGURE_PATH}", flush=True)
    print(f"figure={FRF_FIGURE_PATH}", flush=True)
    print(f"figure={HISTORY_FIGURE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
