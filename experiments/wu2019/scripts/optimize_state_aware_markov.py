"""Optimize the causal state-aware hard Markov controller."""

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
from wu2019.markov import FD_EPSILON, evaluate_markov, uniform_bank
from wu2019.newmark import simulate_summary_batch
from wu2019.state_aware import (
    DISPLACEMENT_SCALE,
    INITIAL_STATE_AWARE_COEFFICIENTS,
    NUM_STATE_AWARE_COEFFICIENTS,
    PHASE2_COEFFICIENTS,
    VELOCITY_SCALE,
    crn_fd_state_aware,
    direct_ad_state_aware,
    evaluate_state_aware,
    replay_state_aware,
)


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
ZERO_GAIN_RTOL = 1e-12
ZERO_GAIN_ATOL = 1e-14
MATERIAL_IMPROVEMENT_PP = 2.0

TRAIN_OMEGAS = np.arange(190.0, 220.0 + 1e-12, 1.0)
EVAL_OMEGAS = dense_frequency_grid()

OUTPUT_DIRECTORY = EXPERIMENT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIRECTORY / "state_aware_summary.json"
TRACE_FIGURE_PATH = OUTPUT_DIRECTORY / "state_aware_optimization_trace.png"
FRF_FIGURE_PATH = OUTPUT_DIRECTORY / "state_aware_final_frf.png"
HISTORY_FIGURE_PATH = OUTPUT_DIRECTORY / "state_aware_representative_history.png"

FRAME_COLOR = "#20242A"
NEUTRAL_COLOR = "#62676E"
PERIODIC_COLOR = "#A5AAB0"
STATE_COLOR = "#245F8A"
WU_COLOR = "#BC7330"


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
        top=True,
        right=True,
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


def _initial_diagnostics(uniforms) -> dict:
    periodic = evaluate_markov(
        PHASE2_COEFFICIENTS, TRAIN_OMEGAS, uniforms
    )
    state_aware = evaluate_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS, TRAIN_OMEGAS, uniforms
    )
    direct_value, direct_gradient = direct_ad_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS, TRAIN_OMEGAS, uniforms
    )
    finite_difference = crn_fd_state_aware(
        INITIAL_STATE_AWARE_COEFFICIENTS,
        TRAIN_OMEGAS,
        uniforms,
        epsilon=FD_EPSILON,
    )
    objective_delta = abs(state_aware.objective - periodic.objective)
    amplitude_delta = float(
        np.max(np.abs(state_aware.amplitudes - periodic.amplitudes))
    )
    expected_transitions = np.broadcast_to(
        periodic.transition_counts,
        state_aware.transition_counts.shape,
    )
    diagnostics = {
        "objective_m": direct_value,
        "zero_gain_periodic_objective_m": periodic.objective,
        "zero_gain_objective_abs_delta_m": objective_delta,
        "zero_gain_amplitude_max_abs_delta_m": amplitude_delta,
        "zero_gain_transition_counts_equal": bool(
            np.array_equal(
                state_aware.transition_counts, expected_transitions
            )
        ),
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
        "mode_history_frequency_rad_s": (
            finite_difference.history_omega
        ),
    }
    zero_gain_aligned = bool(
        np.allclose(
            state_aware.amplitudes,
            periodic.amplitudes,
            rtol=ZERO_GAIN_RTOL,
            atol=ZERO_GAIN_ATOL,
        )
        and np.isclose(
            state_aware.objective,
            periodic.objective,
            rtol=ZERO_GAIN_RTOL,
            atol=ZERO_GAIN_ATOL,
        )
        and diagnostics["zero_gain_transition_counts_equal"]
    )
    diagnostics["gates"] = {
        "zero_gain_equivalence": zero_gain_aligned,
        "direct_ad_zero": (
            diagnostics["direct_ad_linf"] <= DIRECT_AD_ZERO_ATOL
        ),
        "crn_fd_finite": bool(
            np.all(np.isfinite(finite_difference.gradient))
        ),
        "crn_fd_nonzero": diagnostics["crn_fd_l2"] > 0.0,
        "state_gain_gradient_nonzero": bool(
            np.any(np.abs(finite_difference.gradient[5:]) > 0.0)
        ),
        "state_gain_changes_mode_history": bool(
            np.any(finite_difference.mode_difference_counts[5:] > 0)
        ),
    }
    return diagnostics


def _train(trace_uniforms):
    coefficients = torch.nn.Parameter(
        torch.tensor(
            INITIAL_STATE_AWARE_COEFFICIENTS, dtype=torch.float64
        )
    )
    optimizer = torch.optim.Adam([coefficients], lr=LEARNING_RATE)
    coefficient_history = np.empty(
        (NUM_UPDATES + 1, NUM_STATE_AWARE_COEFFICIENTS),
        dtype=np.float64,
    )
    sampled_objectives = np.empty(NUM_UPDATES, dtype=np.float64)
    gradient_norms = np.empty(NUM_UPDATES, dtype=np.float64)
    trace_iterations = np.arange(
        0, NUM_UPDATES + 1, TRACE_INTERVAL, dtype=np.int64
    )
    trace_objectives = np.empty(len(trace_iterations), dtype=np.float64)
    coefficient_history[0] = coefficients.detach().numpy()
    trace_objectives[0] = evaluate_state_aware(
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
        sampled_objectives[iteration] = evaluate_state_aware(
            coefficient_array, TRAIN_OMEGAS, train_uniforms
        ).objective
        finite_difference = crn_fd_state_aware(
            coefficient_array,
            TRAIN_OMEGAS,
            train_uniforms,
            epsilon=FD_EPSILON,
            check_mode_history=False,
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
            trace_objectives[trace_index] = evaluate_state_aware(
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
        color=PERIODIC_COLOR,
        alpha=0.65,
        linewidths=0.0,
        label="Sampled training bank",
    )
    axes[0].plot(
        history["trace_iterations"],
        1e3 * history["trace_objectives"],
        color=STATE_COLOR,
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


def _plot_frfs(constant, wu, periodic, state_aware) -> None:
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
        1e3 * periodic,
        color=PERIODIC_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="Periodic hard Markov",
    )
    axis.plot(
        EVAL_OMEGAS,
        1e3 * state_aware,
        color=STATE_COLOR,
        linewidth=2.2,
        label="State-aware hard Markov",
    )
    axis.set_xlabel("Excitation frequency (rad s$^{-1}$)")
    axis.set_ylabel("Response amplitude (mm)")
    axis.legend(loc="best")
    _style_axis(axis)
    figure.savefig(FRF_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_representative_history(
    coefficients, omega, tape, realization
) -> dict:
    replay = replay_state_aware(coefficients, omega, tape)
    window_steps = 2 * DEFAULT_SETTINGS.steps_per_period
    time = replay.time[-window_steps:]
    time_ms = 1e3 * (time - time[0])
    displacement_mm = 1e3 * replay.displacement[-window_steps:]
    normalized_velocity = replay.velocity[-window_steps:] / VELOCITY_SCALE
    normal_force = replay.preload[-window_steps:]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(6.6, 6.2),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        time_ms, displacement_mm, color=STATE_COLOR, linewidth=1.8
    )
    axes[0].set_ylabel("Displacement (mm)")
    axes[1].plot(
        time_ms, normalized_velocity, color=WU_COLOR, linewidth=1.6
    )
    axes[1].set_ylabel("Normalized velocity")
    axes[2].step(
        time_ms,
        normal_force,
        where="post",
        color=NEUTRAL_COLOR,
        linewidth=1.6,
    )
    axes[2].set_xlabel("Time within window (ms)")
    axes[2].set_ylabel("Normal force (N)")
    axes[2].set_yticks([30.0, 50.0])
    axes[2].set_ylim(26.0, 54.0)
    for axis in axes:
        _style_axis(axis)
    figure.savefig(HISTORY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "omega": float(omega),
        "realization": int(realization),
        "selection": "fixed first evaluation realization",
        "window_periods": 2,
        "transition_count_full_history": int(
            np.count_nonzero(np.diff(replay.modes.astype(np.int8)))
        ),
        "high_mode_fraction_full_history": float(np.mean(replay.modes)),
    }


def main() -> int:
    _configure_plotting()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    initial_uniforms = uniform_bank(
        NUM_TRAIN_REALIZATIONS, TRAIN_BASE_SEED
    )
    diagnostics = _initial_diagnostics(initial_uniforms)
    print(json.dumps(diagnostics, indent=2), flush=True)
    if not all(diagnostics["gates"].values()):
        raise SystemExit("state-aware implementation gate failed")

    trace_uniforms = uniform_bank(
        NUM_TRACE_REALIZATIONS, TRACE_BASE_SEED
    )
    history = _train(trace_uniforms)
    optimized_coefficients = history["coefficient_history"][-1]

    evaluation_uniforms = uniform_bank(
        NUM_EVAL_REALIZATIONS, EVAL_BASE_SEED
    )
    periodic = evaluate_markov(
        PHASE2_COEFFICIENTS, EVAL_OMEGAS, evaluation_uniforms
    )
    state_aware = evaluate_state_aware(
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
    periodic_peak = _peak(EVAL_OMEGAS, periodic.frequency_means)
    state_peak = _peak(EVAL_OMEGAS, state_aware.frequency_means)
    reference = constant_peak["amplitude_m"]
    periodic_reduction = _reduction(
        reference, periodic_peak["amplitude_m"]
    )
    state_reduction = _reduction(
        reference, state_peak["amplitude_m"]
    )
    delta_reduction_pp = state_reduction - periodic_reduction

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
            "controller": "Phase 2 periodic Markov",
            "normal_force": "{30,50} N",
            "peak": periodic_peak,
            "reduction_vs_constant_percent": periodic_reduction,
        },
        {
            "controller": "Phase 2B state-aware Markov",
            "normal_force": "{30,50} N",
            "peak": state_peak,
            "reduction_vs_constant_percent": state_reduction,
        },
    ]

    state_peak_index = int(np.argmax(state_aware.frequency_means))
    representative = 0
    representative_metadata = _plot_representative_history(
        optimized_coefficients,
        EVAL_OMEGAS[state_peak_index],
        evaluation_uniforms[representative],
        representative,
    )
    _plot_trace(history)
    _plot_frfs(
        constant.amplitude,
        wu.amplitude,
        periodic.frequency_means,
        state_aware.frequency_means,
    )

    scientific_outcome = {
        "state_aware_objective_lower_than_periodic": bool(
            state_aware.objective < periodic.objective
        ),
        "state_aware_improvement_vs_periodic_percent": _reduction(
            periodic.objective, state_aware.objective
        ),
        "periodic_reduction_vs_constant_percent": periodic_reduction,
        "state_aware_reduction_vs_constant_percent": state_reduction,
        "delta_reduction_percentage_points": delta_reduction_pp,
        "material_threshold_percentage_points": MATERIAL_IMPROVEMENT_PP,
        "state_feedback_materially_helpful": bool(
            state_aware.objective < periodic.objective
            and delta_reduction_pp >= MATERIAL_IMPROVEMENT_PP
        ),
        "meets_18_percent_future_science_gate": (
            state_reduction >= 18.0
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
            "training_seed_rule": (
                "TRAIN_BASE_SEED + update * train_realizations + realization"
            ),
            "fd_epsilon": FD_EPSILON,
            "adam_learning_rate": LEARNING_RATE,
            "num_updates": NUM_UPDATES,
            "displacement_scale_m": DISPLACEMENT_SCALE,
            "velocity_scale_m_s": VELOCITY_SCALE,
            "normalized_state_clipped": False,
            "causal_timing": (
                "previous state -> transition -> preload -> Newmark update"
            ),
            "parameter_order": [
                "a0", "a1", "b1", "a2", "b2", "c_v", "c_x"
            ],
        },
        "gradient_diagnostics": diagnostics,
        "phase2_periodic_coefficients": PHASE2_COEFFICIENTS.tolist(),
        "initial_coefficients": (
            INITIAL_STATE_AWARE_COEFFICIENTS.tolist()
        ),
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
            "periodic_frequency_mean_amplitude_m": (
                periodic.frequency_means.tolist()
            ),
            "state_aware_frequency_mean_amplitude_m": (
                state_aware.frequency_means.tolist()
            ),
            "periodic_transition_counts": (
                periodic.transition_counts.astype(int).tolist()
            ),
            "periodic_high_mode_fraction": (
                periodic.high_mode_fraction.tolist()
            ),
            "state_aware_transition_counts_by_frequency": (
                state_aware.transition_counts.astype(int).tolist()
            ),
            "state_aware_high_mode_fraction_by_frequency": (
                state_aware.high_mode_fraction.tolist()
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
