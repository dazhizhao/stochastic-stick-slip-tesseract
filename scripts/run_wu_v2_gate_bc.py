"""Run Wu-V2 independent-bank Gate B and 20-update Gate C."""

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
import torch

from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    SYSTEM,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    CONDITION_LABELS,
    FD_EPSILON,
    MARKOV_BASE_SEED,
    NUM_STEPS,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    crn_centered_fd,
    evaluate_markov_bank,
    markov_uniform_bank,
    policy_polar_coordinates,
)


GATE_0_PATH = ROOT / "outputs/wu_v2_gate0_final/results.json"
GATE_A_PATH = ROOT / "outputs/wu_v2_gate_a/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_gate_bc"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "gate_bc_summary.png"

BANK_A_STREAM = 0
BANK_B_STREAM = 1
TRAINING_STREAM = 2
EVALUATION_STREAM = 3
NUM_TRAINING_REALIZATIONS = 4
NUM_EVALUATION_REALIZATIONS = 8
NUM_UPDATES = 20
LEARNING_RATE = 0.01
COSINE_THRESHOLD = 0.5
NUMERICAL_ZERO_ATOL = 1e-12
A2_REPRODUCTION_RTOL = 1e-12
A2_REPRODUCTION_ATOL = 1e-14
Q0 = np.zeros(2, dtype=np.float64)
REPORTED_ITERATIONS = (0, 1, 2, 5, 10, 20)

FRAME_COLOR = "#20242A"
BANK_A_COLOR = "#376A8B"
BANK_B_COLOR = "#B36A4C"
OPTIMIZED_COLOR = "#315F55"
NEUTRAL_COLOR = "#737A83"
REFERENCE_COLOR = "#7D5687"


def _load_frozen_inputs() -> tuple[float, float, dict, np.ndarray]:
    gate_0 = json.loads(GATE_0_PATH.read_text())
    gate_a = json.loads(GATE_A_PATH.read_text())
    if gate_0["gate_0"]["result"] != "PASS":
        raise RuntimeError("Final Wu-style Gate 0 is not PASS")
    if gate_a["gate_a"]["result"] != "PASS":
        raise RuntimeError("Wu-V2 Gate A is not PASS")
    configuration = gate_a["configuration"]
    if not (
        np.isclose(configuration["preload_low"], PRELOAD_LOW)
        and np.isclose(configuration["preload_high"], PRELOAD_HIGH)
        and np.isclose(configuration["fd_epsilon"], FD_EPSILON)
        and configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
    ):
        raise RuntimeError("Gate A does not match the frozen A3 configuration")
    omega = float(configuration["omega_r"])
    omega_ratio = float(configuration["omega_r_ratio"])
    references = {
        "passive": float(gate_0["passive"]["steady_amplitude"]),
        "best_deterministic_1omega": float(
            gate_0["phase_sweep"]["one_omega"]["best_amplitude"]
        ),
        "best_deterministic_2omega": float(
            gate_0["phase_sweep"]["two_omega"]["best_amplitude"]
        ),
        "best_deterministic_2omega_phase": float(
            gate_0["phase_sweep"]["two_omega"]["best_phase"]
        ),
        "stochastic_gate_a_neutral": float(
            gate_a["neutral"]["mean_amplitude"]
        ),
    }
    gate_a_gradient = np.asarray(
        gate_a["crn_fd"]["gradient"], dtype=np.float64
    )
    return omega, omega_ratio, references, gate_a_gradient


def _mean_objective(evaluation: dict) -> float:
    objectives = np.asarray(evaluation["trajectory_objectives"])
    if not np.all(np.isfinite(objectives)):
        raise FloatingPointError("Markov objective is non-finite")
    return float(np.mean(objectives))


def _gradient_summary(fd: dict) -> dict:
    gradient = np.asarray(fd["gradient"], dtype=np.float64)
    finite = bool(np.all(np.isfinite(gradient)))
    l2_norm = float(np.linalg.norm(gradient))
    return {
        "gradient": gradient.tolist(),
        "l2_norm": l2_norm,
        "finite": finite,
        "nonzero": bool(finite and l2_norm > NUMERICAL_ZERO_ATOL),
        "plus_objectives": fd["plus_objectives"],
        "minus_objectives": fd["minus_objectives"],
        "mode_difference_counts": fd["mode_difference_counts"],
    }


def _bank_gradient(
    q: np.ndarray,
    stream_id: int,
    iteration: int,
    forcing: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
) -> dict:
    uniforms = markov_uniform_bank(
        NUM_TRAINING_REALIZATIONS, stream_id, iteration
    )
    return _gradient_summary(
        crn_centered_fd(q, forcing, uniforms, times, omega, time_step)
    )


def _gate_b(
    forcing: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
    gate_a_gradient: np.ndarray,
) -> dict:
    bank_a = _bank_gradient(
        Q0,
        BANK_A_STREAM,
        0,
        forcing,
        times,
        omega,
        time_step,
    )
    bank_b = _bank_gradient(
        Q0,
        BANK_B_STREAM,
        0,
        forcing,
        times,
        omega,
        time_step,
    )
    gradient_a = np.asarray(bank_a["gradient"])
    gradient_b = np.asarray(bank_b["gradient"])
    if not np.allclose(
        gradient_a,
        gate_a_gradient,
        rtol=A2_REPRODUCTION_RTOL,
        atol=A2_REPRODUCTION_ATOL,
    ):
        raise AssertionError("Bank A no longer reproduces the A2 gradient")
    cosine = None
    if bank_a["nonzero"] and bank_b["nonzero"]:
        cosine = float(
            np.dot(gradient_a, gradient_b)
            / (bank_a["l2_norm"] * bank_b["l2_norm"])
        )
    same_sign = [
        bool(
            (value_a > 0.0 and value_b > 0.0)
            or (value_a < 0.0 and value_b < 0.0)
        )
        for value_a, value_b in zip(gradient_a, gradient_b, strict=True)
    ]
    passed = bool(
        bank_a["finite"]
        and bank_b["finite"]
        and bank_a["nonzero"]
        and bank_b["nonzero"]
        and cosine is not None
        and cosine > COSINE_THRESHOLD
    )
    if passed:
        reason = (
            f"Independent-bank gradient cosine {cosine:.6g} exceeds "
            f"{COSINE_THRESHOLD}."
        )
    elif cosine is None:
        reason = "At least one independent-bank gradient is non-finite or zero."
    else:
        reason = (
            f"Independent-bank gradient cosine {cosine:.6g} does not exceed "
            f"{COSINE_THRESHOLD}."
        )
    return {
        "bank_a": bank_a,
        "bank_b": bank_b,
        "bank_a_matches_gate_a": True,
        "cosine": cosine,
        "component_same_sign": same_sign,
        "result": "PASS" if passed else "FAIL",
        "reason": reason,
    }


def _markov_summary(evaluation: dict) -> dict:
    occupancy = np.asarray(evaluation["high_mode_fraction"])
    transitions = np.asarray(evaluation["transition_counts"])
    return {
        "high_occupancy_per_contact": np.mean(
            occupancy, axis=(0, 1)
        ).tolist(),
        "mean_transitions_per_trajectory_contact": np.mean(
            transitions, axis=(0, 1)
        ).tolist(),
    }


def _history_entry(
    iteration: int,
    q: np.ndarray,
    sampled_train: float,
    fixed_evaluation: float,
    gradient_norm: float | None,
    training_bank_iteration: int,
) -> dict:
    return {
        "iteration": iteration,
        "q": np.asarray(q, dtype=np.float64).tolist(),
        "sampled_train_amplitude": sampled_train,
        "fixed_evaluation_amplitude": fixed_evaluation,
        "gradient_norm": gradient_norm,
        "training_bank_iteration": training_bank_iteration,
    }


def _optimize(
    forcing: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
    references: dict,
) -> tuple[dict, dict, dict]:
    evaluation_uniforms = markov_uniform_bank(
        NUM_EVALUATION_REALIZATIONS, EVALUATION_STREAM, 0
    )
    initial_training_uniforms = markov_uniform_bank(
        NUM_TRAINING_REALIZATIONS, TRAINING_STREAM, 0
    )
    initial_training = evaluate_markov_bank(
        Q0,
        forcing,
        initial_training_uniforms,
        times,
        omega,
        time_step,
    )
    initial_evaluation = evaluate_markov_bank(
        Q0,
        forcing,
        evaluation_uniforms,
        times,
        omega,
        time_step,
    )
    initial_eval_objective = _mean_objective(initial_evaluation)
    history = [
        _history_entry(
            0,
            Q0,
            _mean_objective(initial_training),
            initial_eval_objective,
            None,
            0,
        )
    ]

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)
    final_evaluation = initial_evaluation
    for update in range(1, NUM_UPDATES + 1):
        bank_iteration = update - 1
        training_uniforms = markov_uniform_bank(
            NUM_TRAINING_REALIZATIONS,
            TRAINING_STREAM,
            bank_iteration,
        )
        q_before = parameter.detach().numpy().copy()
        fd = crn_centered_fd(
            q_before,
            forcing,
            training_uniforms,
            times,
            omega,
            time_step,
        )
        gradient = np.asarray(fd["gradient"], dtype=np.float64)
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                f"Training gradient is non-finite at update {update}"
            )
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.from_numpy(gradient.copy())
        optimizer.step()
        q_after = parameter.detach().numpy().copy()
        if not np.all(np.isfinite(q_after)):
            raise FloatingPointError(
                f"Adam parameter is non-finite at update {update}"
            )
        sampled_training = evaluate_markov_bank(
            q_after,
            forcing,
            training_uniforms,
            times,
            omega,
            time_step,
        )
        final_evaluation = evaluate_markov_bank(
            q_after,
            forcing,
            evaluation_uniforms,
            times,
            omega,
            time_step,
        )
        history.append(
            _history_entry(
                update,
                q_after,
                _mean_objective(sampled_training),
                _mean_objective(final_evaluation),
                float(np.linalg.norm(gradient)),
                bank_iteration,
            )
        )

    final_q = parameter.detach().numpy().copy()
    final_objective = history[-1]["fixed_evaluation_amplitude"]
    relative_improvement = (
        100.0
        * (initial_eval_objective - final_objective)
        / initial_eval_objective
    )
    denominator = (
        initial_eval_objective - references["best_deterministic_2omega"]
    )
    recovered_fraction = None
    if denominator > 0.0:
        recovered_fraction = float(
            (initial_eval_objective - final_objective) / denominator
        )
    magnitude, phase = policy_polar_coordinates(final_q)
    optimization = {
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "num_updates": NUM_UPDATES,
            "selected_iteration": NUM_UPDATES,
        },
        "history": history,
        "initial_fixed_evaluation_amplitude": initial_eval_objective,
        "final_fixed_evaluation_amplitude": final_objective,
        "relative_improvement_percent": relative_improvement,
        "recovered_fraction": recovered_fraction,
    }
    learned_policy = {
        "q20": final_q.tolist(),
        "magnitude": magnitude,
        "coefficient_phase": phase,
        "phase_convention": "h=R*cos(2*omega*t-coefficient_phase)",
        "deterministic_best_2omega_phase": references[
            "best_deterministic_2omega_phase"
        ],
    }
    markov_diagnostics = {
        "iteration_0": _markov_summary(initial_evaluation),
        "iteration_20": _markov_summary(final_evaluation),
    }
    return optimization, learned_policy, markov_diagnostics


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
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


def _plot_gradient_axis(axis, gate_b: dict) -> None:
    gradient_a = np.asarray(gate_b["bank_a"]["gradient"])
    gradient_b = np.asarray(gate_b["bank_b"]["gradient"])
    for gradient, color, label in (
        (gradient_a, BANK_A_COLOR, "Bank A"),
        (gradient_b, BANK_B_COLOR, "Bank B"),
    ):
        norm = np.linalg.norm(gradient)
        if not np.isfinite(norm) or norm <= NUMERICAL_ZERO_ATOL:
            continue
        direction = -gradient / norm
        axis.quiver(
            0.0,
            0.0,
            direction[0],
            direction[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.018,
            label=label,
        )
    axis.scatter([0.0], [0.0], color=FRAME_COLOR, s=12, zorder=3)
    axis.set(
        xlabel=r"Descent direction $a_2$",
        ylabel=r"Descent direction $b_2$",
        xlim=(-1.08, 1.08),
        ylim=(-1.08, 1.08),
        xticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
        yticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    axis.set_aspect("equal", adjustable="box")
    cosine = gate_b["cosine"]
    cosine_label = "undefined" if cosine is None else f"{cosine:.3f}"
    axis.legend(
        loc="best", title=f"cosine = {cosine_label}", title_fontsize=8
    )


def _plot(results: dict) -> None:
    _configure_plotting()
    if results["optimization"] is None:
        figure, axis_a = plt.subplots(
            1, 1, figsize=(3.4, 3.2), constrained_layout=True
        )
        _plot_gradient_axis(axis_a, results["gate_b"])
        axes = (("a", axis_a),)
    else:
        figure = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.65))
        axis_a = figure.add_subplot(grid[0, 0])
        axis_b = figure.add_subplot(grid[0, 1])
        _plot_gradient_axis(axis_a, results["gate_b"])

        history = results["optimization"]["history"]
        iterations = [entry["iteration"] for entry in history]
        evaluations = [
            entry["fixed_evaluation_amplitude"] for entry in history
        ]
        neutral = evaluations[0]
        deterministic = results["wu_references"][
            "best_deterministic_2omega"
        ]
        axis_b.plot(
            iterations,
            evaluations,
            color=OPTIMIZED_COLOR,
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            label="CRN-FD",
        )
        axis_b.axhline(
            neutral,
            color=NEUTRAL_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="Neutral / Direct AD",
        )
        axis_b.axhline(
            deterministic,
            color=REFERENCE_COLOR,
            linestyle=":",
            linewidth=1.5,
            label=r"Deterministic $2\omega$",
        )
        axis_b.set(
            xlabel="Adam iteration",
            ylabel="Fixed-evaluation amplitude",
            xlim=(0, NUM_UPDATES),
            xticks=(0, 5, 10, 15, 20),
        )
        axis_b.legend(loc="best")
        axes = (("a", axis_a), ("b", axis_b))

    for label, axis in axes:
        _style_axis(axis)
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_results(results: dict) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    _plot(results)


def _print_results(results: dict) -> None:
    gate_b = results["gate_b"]
    print("## Gate B")
    print(f"g_A={gate_b['bank_a']['gradient']}")
    print(f"g_B={gate_b['bank_b']['gradient']}")
    print(f"norm_A={gate_b['bank_a']['l2_norm']:.16g}")
    print(f"norm_B={gate_b['bank_b']['l2_norm']:.16g}")
    print(f"cosine={gate_b['cosine']}")
    print(f"component_same_sign={gate_b['component_same_sign']}")
    print(f"Gate B: {gate_b['result']}")
    if results["optimization"] is not None:
        print("## Gate C")
        print("iter | sampled train | fixed eval | q | gradient norm")
        for entry in results["optimization"]["history"]:
            if entry["iteration"] in REPORTED_ITERATIONS:
                print(
                    f"{entry['iteration']:>4} | "
                    f"{entry['sampled_train_amplitude']:.10g} | "
                    f"{entry['fixed_evaluation_amplitude']:.10g} | "
                    f"{entry['q']} | {entry['gradient_norm']}"
                )
        print(
            "relative_improvement_percent="
            f"{results['optimization']['relative_improvement_percent']:.16g}"
        )
        print(f"q20={results['learned_policy']['q20']}")
        print(f"Gate C: {results['gate_c']['result']}")
    else:
        print("## Gate C")
        print("Gate C: SKIPPED")
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())


def main() -> int:
    omega, omega_ratio, references, gate_a_gradient = _load_frozen_inputs()
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
    )
    times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    gate_b = _gate_b(
        forcing,
        times,
        omega,
        time_step,
        gate_a_gradient,
    )
    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "omega_r": omega,
            "omega_r_ratio": omega_ratio,
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": NUM_STEPS // DIAGNOSTIC_NUM_PERIODS,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "fd_epsilon": FD_EPSILON,
            "markov_base_seed": MARKOV_BASE_SEED,
            "condition_labels": CONDITION_LABELS.tolist(),
            "all_realizations_share_nominal_forcing": True,
            "streams": {
                "bank_a": BANK_A_STREAM,
                "bank_b": BANK_B_STREAM,
                "training": TRAINING_STREAM,
                "fixed_evaluation": EVALUATION_STREAM,
            },
            "bank_a_realizations": 32,
            "bank_b_realizations": 32,
            "training_realizations_per_update": 32,
            "fixed_evaluation_realizations": 64,
            "cosine_threshold": COSINE_THRESHOLD,
        },
        "wu_references": references,
        "gate_b": gate_b,
        "optimization": None,
        "learned_policy": None,
        "markov_diagnostics": None,
        "gate_c": {
            "result": "SKIPPED",
            "reason": "Gate B did not pass.",
        },
    }
    if gate_b["result"] == "PASS":
        optimization, learned_policy, markov_diagnostics = _optimize(
            forcing, times, omega, time_step, references
        )
        results["optimization"] = optimization
        results["learned_policy"] = learned_policy
        results["markov_diagnostics"] = markov_diagnostics
        passed = bool(
            optimization["final_fixed_evaluation_amplitude"]
            < optimization["initial_fixed_evaluation_amplitude"]
        )
        results["gate_c"] = {
            "result": "PASS" if passed else "FAIL",
            "reason": (
                "Iteration 20 reduced the fixed-evaluation mean amplitude."
                if passed
                else "Iteration 20 did not reduce the fixed-evaluation mean amplitude."
            ),
        }
    _write_results(results)
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
