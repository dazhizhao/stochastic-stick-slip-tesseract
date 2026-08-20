"""Run the compact Markov-jump gradient and controller ablation study."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_markov_jump_long_training import (
    CONTROLLER_API,
    HELD_OUT_STREAM,
    LEARNING_RATE,
    MONITOR_STREAM,
    NUM_REALIZATIONS,
    NUM_UPDATES,
    PHYSICS_API,
    TRAINING_STREAM,
    BankEvaluation,
    _bank_objective,
    _evaluate_bank,
    _initial_theta,
    _seed_batches,
)
from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.engineering_markov import (
    BETA,
    DAMPING,
    FD_EPSILON,
    GATE_A_FORCING_SEEDS,
    LAMBDA_0,
    MARKOV_BASE_SEED,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    evaluate_markov_bank,
    markov_uniform_bank,
)
from stochastic_stick_slip.engineering_showcase import forcing_batch
from stochastic_stick_slip.model import (
    NUM_FOURIER_COEFFICIENTS,
    forcing_descriptor_batch,
)


OUTPUT_DIRECTORY = ROOT / "outputs/markov_jump_ablation"
OUTPUT_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "ablation_summary.png"
GATE_A_PATH = ROOT / "outputs/markov_jump_gate_a/results.json"
GATE_C_PATH = ROOT / "outputs/markov_jump_gate_c/results.json"
M4_HISTORY_PATH = (
    ROOT / "outputs/markov_jump_long_training/training_history.npz"
)

COUPLING_PLUS_STREAM = 10
COUPLING_MINUS_STREAM = 11
INDEPENDENT_TRAINING_MINUS_STREAM = 12
COUPLING_REPLICATES = 6
INDEPENDENT_UPDATES = 20
NUMERICAL_ZERO_ATOL = 1e-12
REPORTED_TRAINING_ITERATIONS = (0, 1, 5, 10, 20)

M4_HELD_OUT_NEUTRAL = 0.011426760350106483
M4_HELD_OUT_MLP = 0.010438629123075804
M4_HELD_OUT_MLP_WINS = 52

FRAME_COLOR = "#23272E"
DIRECT_COLOR = "#777D86"
CRN_COLOR = "#3977A8"
INDEPENDENT_COLOR = "#D4863B"
NEUTRAL_COLOR = "#A3A8AE"
SHARED_COLOR = "#77A7C8"
MLP_COLOR = "#245D8C"


def _evaluate_coefficients(coefficients, seeds, uniforms) -> BankEvaluation:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.int64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    collected = {
        "losses": [],
        "transition_counts": [],
        "high_mode_fraction": [],
    }
    for start, batch_seeds in zip(
        range(0, len(seeds), 8), _seed_batches(seeds), strict=True
    ):
        result = evaluate_markov_bank(
            coefficients[start : start + 8],
            forcing_batch(batch_seeds),
            uniforms[start : start + 8],
        )
        collected["losses"].append(np.asarray(result.losses))
        collected["transition_counts"].append(
            np.asarray(result.transition_counts)
        )
        collected["high_mode_fraction"].append(
            np.asarray(result.high_mode_fraction)
        )
    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in collected.items()
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise FloatingPointError("coefficient bank evaluation is non-finite")
    return BankEvaluation(
        coefficients=coefficients.copy(),
        losses=arrays["losses"],
        seed_losses=np.mean(arrays["losses"], axis=1),
        transition_counts=arrays["transition_counts"],
        high_mode_fraction=arrays["high_mode_fraction"],
    )


def _per_seed_coordinate_fd(
    coefficients,
    seeds,
    plus_uniforms,
    minus_uniforms,
) -> np.ndarray:
    """Return per-seed coordinate FD using explicit plus/minus tapes."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.int64)
    plus_uniforms = np.asarray(plus_uniforms, dtype=np.float64)
    minus_uniforms = np.asarray(minus_uniforms, dtype=np.float64)
    forcing = forcing_batch(seeds)
    columns = []
    for column in range(NUM_FOURIER_COEFFICIENTS):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[:, column] += FD_EPSILON
        minus[:, column] -= FD_EPSILON
        plus_losses = np.mean(
            np.asarray(
                evaluate_markov_bank(plus, forcing, plus_uniforms).losses
            ),
            axis=1,
        )
        minus_losses = np.mean(
            np.asarray(
                evaluate_markov_bank(minus, forcing, minus_uniforms).losses
            ),
            axis=1,
        )
        columns.append(
            (plus_losses - minus_losses) / (2.0 * FD_EPSILON)
        )
    jacobian = np.stack(columns, axis=1)
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("coordinate FD Jacobian is non-finite")
    return jacobian


def _shared_coordinate_gradient(
    shared_coefficients,
    seeds,
    plus_uniforms,
    minus_uniforms,
) -> np.ndarray:
    coefficients = np.broadcast_to(
        np.asarray(shared_coefficients, dtype=np.float64),
        (len(seeds), NUM_FOURIER_COEFFICIENTS),
    ).copy()
    return np.mean(
        _per_seed_coordinate_fd(
            coefficients, seeds, plus_uniforms, minus_uniforms
        ),
        axis=0,
    )


def _cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        np.max(np.abs(left)) <= NUMERICAL_ZERO_ATOL
        or np.max(np.abs(right)) <= NUMERICAL_ZERO_ATOL
    ):
        return None
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _direction_statistics(gradients):
    gradients = np.asarray(gradients, dtype=np.float64)
    if not np.all(np.isfinite(gradients)):
        raise FloatingPointError("coupling gradients are non-finite")
    mean_gradient = np.mean(gradients, axis=0)
    to_mean = [_cosine(gradient, mean_gradient) for gradient in gradients]
    pairwise = [
        _cosine(gradients[left], gradients[right])
        for left, right in itertools.combinations(range(len(gradients)), 2)
    ]
    defined_to_mean = [value for value in to_mean if value is not None]
    defined_pairwise = [value for value in pairwise if value is not None]
    return {
        "mean_gradient": mean_gradient.tolist(),
        "mean_gradient_l2": float(np.linalg.norm(mean_gradient)),
        "gradient_l2": np.linalg.norm(gradients, axis=1).tolist(),
        "cosine_to_mean": to_mean,
        "cosine_to_mean_defined_count": len(defined_to_mean),
        "cosine_to_mean_mean": (
            float(np.mean(defined_to_mean)) if defined_to_mean else None
        ),
        "cosine_to_mean_min": (
            float(np.min(defined_to_mean)) if defined_to_mean else None
        ),
        "pairwise_cosines": pairwise,
        "pairwise_defined_count": len(defined_pairwise),
        "pairwise_cosine_mean": (
            float(np.mean(defined_pairwise)) if defined_pairwise else None
        ),
    }


def _run_coupling_replicates():
    shared = np.zeros(NUM_FOURIER_COEFFICIENTS, dtype=np.float64)
    crn_gradients = []
    independent_gradients = []
    start = time.perf_counter()
    for replicate in range(COUPLING_REPLICATES):
        plus_uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=COUPLING_PLUS_STREAM,
            forcing_seeds=GATE_A_FORCING_SEEDS,
            iteration=replicate,
        )
        minus_uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=COUPLING_MINUS_STREAM,
            forcing_seeds=GATE_A_FORCING_SEEDS,
            iteration=replicate,
        )
        crn = _shared_coordinate_gradient(
            shared,
            GATE_A_FORCING_SEEDS,
            plus_uniforms,
            plus_uniforms,
        )
        independent = _shared_coordinate_gradient(
            shared,
            GATE_A_FORCING_SEEDS,
            plus_uniforms,
            minus_uniforms,
        )
        crn_gradients.append(crn)
        independent_gradients.append(independent)
        print(
            f"coupling_replicate={replicate + 1}/{COUPLING_REPLICATES} "
            f"crn_l2={np.linalg.norm(crn):.9g} "
            f"independent_l2={np.linalg.norm(independent):.9g}",
            flush=True,
        )
    return {
        "crn_gradients": np.asarray(crn_gradients).tolist(),
        "independent_gradients": np.asarray(independent_gradients).tolist(),
        "crn": _direction_statistics(crn_gradients),
        "independent": _direction_statistics(independent_gradients),
    }, time.perf_counter() - start


def _read_m4_history():
    if not M4_HISTORY_PATH.exists():
        raise FileNotFoundError(
            "M4 training_history.npz is required; M5 does not retrain the MLP"
        )
    with np.load(M4_HISTORY_PATH) as archive:
        required = {
            "theta_history": (NUM_UPDATES + 1, 469),
            "train_objective": (NUM_UPDATES + 1,),
            "gradient_norm": (NUM_UPDATES,),
            "monitor_iterations": (NUM_UPDATES // 10 + 1,),
            "monitor_objective": (NUM_UPDATES // 10 + 1,),
        }
        if set(archive.files) != set(required):
            raise ValueError("M4 history keys do not match the frozen schema")
        history = {name: np.asarray(archive[name]).copy() for name in required}
    for name, shape in required.items():
        if history[name].shape != shape:
            raise ValueError(
                f"M4 {name} has shape {history[name].shape}, "
                f"expected {shape}"
            )
        if not np.all(np.isfinite(history[name])):
            raise FloatingPointError(f"M4 {name} is non-finite")
    if not np.array_equal(history["theta_history"][0], _initial_theta()):
        raise ValueError("M4 theta0 does not match the fixed controller")
    expected_monitor = np.arange(0, NUM_UPDATES + 1, 10, dtype=np.int64)
    if not np.array_equal(history["monitor_iterations"], expected_monitor):
        raise ValueError("M4 monitor iterations do not match the frozen schedule")
    if (
        NUM_UPDATES != 200
        or NUM_REALIZATIONS != 4
        or LEARNING_RATE != 0.01
        or TRAINING_STREAM != 7
        or MONITOR_STREAM != 8
        or HELD_OUT_STREAM != 9
    ):
        raise ValueError("M4 imported configuration is not the frozen M5 baseline")
    return history


def _m4_crn_reference(history):
    monitor_lookup = dict(
        zip(
            history["monitor_iterations"].astype(int),
            history["monitor_objective"],
            strict=True,
        )
    )
    return {
        "sampled_objective": [
            {
                "iteration": iteration,
                "objective": float(history["train_objective"][iteration]),
            }
            for iteration in REPORTED_TRAINING_ITERATIONS
        ],
        "fixed_monitor": [
            {"iteration": iteration, "objective": float(monitor_lookup[iteration])}
            for iteration in (0, 10, 20)
        ],
        "gradient_norm": history["gradient_norm"][:INDEPENDENT_UPDATES].tolist(),
        "source": "M4 training_history.npz; no retraining",
    }


def _train_independent(controller, theta0, monitor_uniforms):
    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    sampled = np.empty(INDEPENDENT_UPDATES + 1, dtype=np.float64)
    gradient_norm = np.empty(INDEPENDENT_UPDATES, dtype=np.float64)
    monitor_iterations = np.asarray([0, 10, 20], dtype=np.int64)
    monitor = np.empty(3, dtype=np.float64)

    initial_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=TRAINING_STREAM,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=0,
    )
    sampled[0] = _bank_objective(
        _evaluate_bank(controller, theta0, H4_TRAINING_SEEDS, initial_uniforms)
    )
    monitor[0] = _bank_objective(
        _evaluate_bank(controller, theta0, H4_TRAINING_SEEDS, monitor_uniforms)
    )

    start = time.perf_counter()
    monitor_index = 1
    for iteration in range(1, INDEPENDENT_UPDATES + 1):
        plus_uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=TRAINING_STREAM,
            forcing_seeds=H4_TRAINING_SEEDS,
            iteration=iteration,
        )
        minus_uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=INDEPENDENT_TRAINING_MINUS_STREAM,
            forcing_seeds=H4_TRAINING_SEEDS,
            iteration=iteration,
        )
        optimizer.zero_grad(set_to_none=True)
        coefficient_tensors = []
        coefficient_jacobians = []
        for start_index, batch_seeds in zip(
            range(0, len(H4_TRAINING_SEEDS), 8),
            _seed_batches(H4_TRAINING_SEEDS),
            strict=True,
        ):
            coefficients = apply_tesseract(
                controller,
                {
                    "theta": theta,
                    "descriptors": forcing_descriptor_batch(batch_seeds),
                },
            )["coeffs"]
            coefficient_tensors.append(coefficients)
            coefficient_jacobians.append(
                _per_seed_coordinate_fd(
                    coefficients.detach().numpy(),
                    batch_seeds,
                    plus_uniforms[start_index : start_index + 8],
                    minus_uniforms[start_index : start_index + 8],
                )
                / len(H4_TRAINING_SEEDS)
            )
        torch.autograd.backward(
            coefficient_tensors,
            grad_tensors=[
                torch.from_numpy(jacobian) for jacobian in coefficient_jacobians
            ],
        )
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("independent-tape theta gradient is non-finite")
        gradient_norm[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        theta_array = theta.detach().numpy().copy()
        sampled[iteration] = _bank_objective(
            _evaluate_bank(
                controller,
                theta_array,
                H4_TRAINING_SEEDS,
                plus_uniforms,
            )
        )
        if iteration % 10 == 0:
            monitor[monitor_index] = _bank_objective(
                _evaluate_bank(
                    controller,
                    theta_array,
                    H4_TRAINING_SEEDS,
                    monitor_uniforms,
                )
            )
            monitor_index += 1
        print(
            f"independent_iteration={iteration:03d} "
            f"sampled={sampled[iteration]:.16g} "
            f"gradient_norm={gradient_norm[iteration - 1]:.9g}"
            + (
                f" monitor={monitor[monitor_index - 1]:.16g}"
                if iteration % 10 == 0
                else ""
            ),
            flush=True,
        )
    if monitor_index != len(monitor_iterations):
        raise AssertionError("independent fixed-monitor history is incomplete")
    return {
        "sampled_objective": sampled.tolist(),
        "gradient_norm": gradient_norm.tolist(),
        "fixed_monitor_iterations": monitor_iterations.tolist(),
        "fixed_monitor_objective": monitor.tolist(),
        "parameter_displacement": float(
            np.linalg.norm(theta.detach().numpy() - theta0)
        ),
    }, time.perf_counter() - start


def _shared_bank(shared, seeds, uniforms):
    coefficients = np.broadcast_to(
        np.asarray(shared, dtype=np.float64),
        (len(seeds), NUM_FOURIER_COEFFICIENTS),
    ).copy()
    return _evaluate_coefficients(coefficients, seeds, uniforms)


def _shared_training_loss(physics, shared, seeds, uniforms):
    losses = []
    for start, batch_seeds in zip(
        range(0, len(seeds), 8), _seed_batches(seeds), strict=True
    ):
        coefficients = shared.expand(len(batch_seeds), -1)
        response = apply_tesseract(
            physics,
            {
                "coeffs": coefficients,
                "forcing_seeds": batch_seeds,
                "markov_uniforms": uniforms[start : start + 8],
            },
        )
        losses.append(response["seed_losses"])
    return torch.cat(losses).mean()


def _train_shared(physics, monitor_uniforms):
    shared = torch.nn.Parameter(
        torch.zeros(NUM_FOURIER_COEFFICIENTS, dtype=torch.float64)
    )
    optimizer = torch.optim.Adam([shared], lr=LEARNING_RATE)
    sampled = np.empty(NUM_UPDATES + 1, dtype=np.float64)
    gradient_norm = np.empty(NUM_UPDATES, dtype=np.float64)
    monitor_iterations = np.arange(0, NUM_UPDATES + 1, 10, dtype=np.int64)
    monitor = np.empty(len(monitor_iterations), dtype=np.float64)

    initial_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=TRAINING_STREAM,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=0,
    )
    zero = shared.detach().numpy().copy()
    sampled[0] = _bank_objective(
        _shared_bank(zero, H4_TRAINING_SEEDS, initial_uniforms)
    )
    monitor[0] = _bank_objective(
        _shared_bank(zero, H4_TRAINING_SEEDS, monitor_uniforms)
    )

    start = time.perf_counter()
    monitor_index = 1
    for iteration in range(1, NUM_UPDATES + 1):
        uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=TRAINING_STREAM,
            forcing_seeds=H4_TRAINING_SEEDS,
            iteration=iteration,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _shared_training_loss(
            physics, shared, H4_TRAINING_SEEDS, uniforms
        )
        loss.backward()
        gradient = shared.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("shared coefficient gradient is non-finite")
        gradient_norm[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        shared_array = shared.detach().numpy().copy()
        sampled[iteration] = _bank_objective(
            _shared_bank(shared_array, H4_TRAINING_SEEDS, uniforms)
        )
        monitor_value = None
        if iteration % 10 == 0:
            monitor_value = _bank_objective(
                _shared_bank(
                    shared_array, H4_TRAINING_SEEDS, monitor_uniforms
                )
            )
            monitor[monitor_index] = monitor_value
            monitor_index += 1
        if iteration % 10 == 0 or iteration == 1:
            print(
                f"shared_iteration={iteration:03d} "
                f"sampled={sampled[iteration]:.16g} "
                f"gradient_norm={gradient_norm[iteration - 1]:.9g}"
                + (
                    f" monitor={monitor_value:.16g}"
                    if monitor_value is not None
                    else ""
                ),
                flush=True,
            )
    if monitor_index != len(monitor_iterations):
        raise AssertionError("shared fixed-monitor history is incomplete")
    return {
        "final_coefficients": shared.detach().numpy().tolist(),
        "sampled_objective": sampled.tolist(),
        "gradient_norm": gradient_norm.tolist(),
        "fixed_monitor_iterations": monitor_iterations.tolist(),
        "fixed_monitor_objective": monitor.tolist(),
    }, time.perf_counter() - start


def _controller_statistics(name, evaluation, neutral, monitor_final):
    relative = 100.0 * (
        neutral.seed_losses - evaluation.seed_losses
    ) / neutral.seed_losses
    mean = _bank_objective(evaluation)
    neutral_mean = _bank_objective(neutral)
    return {
        "name": name,
        "monitor_final": float(monitor_final),
        "held_out_mean": mean,
        "relative_to_neutral": float(
            100.0 * (neutral_mean - mean) / neutral_mean
        ),
        "median_improvement": float(np.median(relative)),
        "wins_vs_neutral": int(
            np.count_nonzero(evaluation.seed_losses < neutral.seed_losses)
        ),
        "num_forcing_conditions": len(evaluation.seed_losses),
        "forcing_losses": evaluation.seed_losses.tolist(),
        "forcing_improvement": relative.tolist(),
    }


def _evaluate_controller_ablation(
    controller, mlp_theta, shared, seeds, uniforms
):
    neutral = _shared_bank(
        np.zeros(NUM_FOURIER_COEFFICIENTS), seeds, uniforms
    )
    shared_evaluation = _shared_bank(shared, seeds, uniforms)
    mlp = _evaluate_bank(controller, mlp_theta, seeds, uniforms)
    return neutral, shared_evaluation, mlp


def _load_gradient_rule_evidence():
    gate_a = json.loads(GATE_A_PATH.read_text())
    gate_c = json.loads(GATE_C_PATH.read_text())
    if gate_a["gate_a"]["result"] != "PASS":
        raise ValueError("tracked Gate A result is not PASS")
    if gate_c["gate_c"]["result"] != "PASS":
        raise ValueError("tracked Gate C result is not PASS")
    if not gate_a["direct_ad"]["numerical_zero"]:
        raise ValueError("tracked Gate A Direct AD is not numerical-zero")
    return {
        "gate_a_source": str(GATE_A_PATH.relative_to(ROOT)),
        "gate_c_source": str(GATE_C_PATH.relative_to(ROOT)),
        "direct_ad": gate_a["direct_ad"],
        "crn_fd": gate_a["crn_fd"],
        "gate_c": {
            "direct_parameter_displacement": gate_c["direct_ad"][
                "parameter_displacement"
            ],
            "direct_final_theta_gradient_norm": gate_c["direct_ad"][
                "gradient_history"
            ][-1]["theta_gradient_norm"],
            "crn_parameter_displacement": gate_c["crn_fd"][
                "parameter_displacement"
            ],
            "crn_evaluation_initial": gate_c["paired_evaluation"][
                "initial_mean_objective"
            ],
            "crn_evaluation_final": gate_c["paired_evaluation"][
                "final_mean_objective"
            ],
        },
    }


def _configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 1.0,
            "axes.edgecolor": FRAME_COLOR,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.color": FRAME_COLOR,
            "ytick.color": FRAME_COLOR,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _style_axis(axis):
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)
    axis.tick_params(
        top=False,
        right=False,
        direction="in",
        width=0.9,
        colors=FRAME_COLOR,
    )


def _panel_label(axis, label):
    axis.text(
        -0.16,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        color=FRAME_COLOR,
    )


def _plot_summary(gradient_rule, coupling, controller_results):
    _configure_style()
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(0.78, 1.42),
        height_ratios=(1.34, 0.92),
    )
    gradient_axis = fig.add_subplot(grid[0, 0])
    coupling_grid = grid[0, 1].subgridspec(
        2, 1, height_ratios=(1.05, 0.95), hspace=0.08
    )
    cosine_axis = fig.add_subplot(coupling_grid[0, 0])
    monitor_axis = fig.add_subplot(coupling_grid[1, 0])
    controller_axis = fig.add_subplot(grid[1, :])

    direct_norm = gradient_rule["direct_ad"]["l2_norm"]
    crn_norm = gradient_rule["crn_fd"]["l2_norm"]
    gradient_axis.hlines(
        [0, 1],
        0.0,
        [direct_norm, crn_norm],
        color=[DIRECT_COLOR, CRN_COLOR],
        lw=2.2,
    )
    gradient_axis.scatter(
        [direct_norm, crn_norm],
        [0, 1],
        s=46,
        color=[DIRECT_COLOR, CRN_COLOR],
        edgecolor=FRAME_COLOR,
        linewidth=0.55,
        zorder=3,
    )
    gradient_axis.set(
        yticks=[0, 1],
        yticklabels=["Direct AD", "CRN-FD"],
        xlabel="Physics coefficient gradient norm",
        xlim=(-0.045 * crn_norm, 1.18 * crn_norm),
        ylim=(-0.55, 1.55),
    )
    gradient_axis.text(
        direct_norm + 0.035 * crn_norm,
        0,
        "0",
        va="center",
        color=DIRECT_COLOR,
    )
    gradient_axis.text(
        crn_norm,
        1.17,
        r"$1.10\times10^{-2}$",
        ha="center",
        color=CRN_COLOR,
    )
    gradient_axis.text(
        0.51,
        0.17,
        "sample-path gradient\nis locally constant",
        transform=gradient_axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=FRAME_COLOR,
    )

    for row, (method, color) in enumerate(
        (
            ("independent", INDEPENDENT_COLOR),
            ("crn", CRN_COLOR),
        )
    ):
        values = coupling[method]["cosine_to_mean"]
        defined = [
            (index, value)
            for index, value in enumerate(values)
            if value is not None
        ]
        if defined:
            x = [value for _, value in defined]
            offsets = (
                np.linspace(-0.08, 0.08, len(x))
                if len(x) > 1
                else np.asarray([0.0])
            )
            cosine_axis.scatter(
                x,
                row + offsets,
                s=31,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            mean_value = coupling[method]["cosine_to_mean_mean"]
            cosine_axis.vlines(
                mean_value,
                row - 0.19,
                row + 0.19,
                color=color,
                linewidth=2.1,
                zorder=4,
            )
            cosine_axis.text(
                mean_value,
                row + 0.24,
                f"mean = {mean_value:.3f}",
                ha="center",
                va="center",
                fontsize=6.9,
                color=color,
            )
    cosine_axis.axvline(0.0, color=FRAME_COLOR, lw=0.8, alpha=0.55)
    cosine_axis.set(
        yticks=[0, 1],
        yticklabels=["Independent", "CRN"],
        xlabel="Cosine to method mean",
        xlim=(-1.05, 1.05),
        ylim=(-0.38, 1.38),
    )

    crn_monitor = coupling["crn_20_reference"]["fixed_monitor"]
    independent_iterations = coupling["independent_20"][
        "fixed_monitor_iterations"
    ]
    independent_monitor = coupling["independent_20"][
        "fixed_monitor_objective"
    ]
    crn_iterations = [point["iteration"] for point in crn_monitor]
    crn_objective = [point["objective"] for point in crn_monitor]
    monitor_axis.plot(
        independent_iterations,
        independent_monitor,
        color=INDEPENDENT_COLOR,
        marker="o",
        ms=3.6,
        lw=1.5,
    )
    monitor_axis.plot(
        crn_iterations,
        crn_objective,
        color=CRN_COLOR,
        marker="o",
        ms=3.6,
        lw=1.8,
    )
    monitor_axis.text(
        20.45,
        crn_objective[-1],
        "CRN",
        ha="left",
        va="center",
        fontsize=6.8,
        color=CRN_COLOR,
    )
    monitor_axis.text(
        20.45,
        independent_monitor[-1],
        "Independent",
        ha="left",
        va="center",
        fontsize=6.8,
        color=INDEPENDENT_COLOR,
    )
    monitor_axis.set(
        xlabel="Adam iteration",
        ylabel="Fixed monitor",
        xticks=[0, 10, 20],
        xlim=(-0.8, 25.8),
        ylim=(0.0107, 0.01262),
    )

    names = ["Neutral", "Shared", "MLP"]
    changes = np.asarray(
        [controller_results[name.lower()]["relative_to_neutral"] for name in names]
    )
    positions = np.arange(3)[::-1]
    colors = [NEUTRAL_COLOR, SHARED_COLOR, MLP_COLOR]
    controller_axis.hlines(
        positions,
        0.0,
        changes,
        color=colors,
        linewidth=6.2,
        alpha=0.88,
    )
    controller_axis.scatter(
        changes,
        positions,
        s=49,
        color=colors,
        edgecolor=FRAME_COLOR,
        linewidth=0.65,
        zorder=3,
    )
    controller_axis.set(
        yticks=positions,
        yticklabels=["Neutral", "Shared\n53/64 improved", "MLP\n52/64 improved"],
        xlabel="Held-out improvement vs Neutral (%)",
        xlim=(-0.35, 10.15),
        ylim=(-0.55, 2.55),
    )
    for index, change in enumerate(changes):
        position = positions[index]
        label = "0%" if index == 0 else f"{change:.2f}%"
        controller_axis.text(
            change + 0.16,
            position,
            label,
            ha="left",
            va="center",
            fontsize=7.1,
            color=FRAME_COLOR,
        )
    controller_axis.annotate(
        "",
        xy=(changes[2], 0.5),
        xytext=(changes[1], 0.5),
        arrowprops={"arrowstyle": "<->", "color": FRAME_COLOR, "lw": 0.75},
    )
    controller_axis.text(
        4.2,
        0.5,
        "MLP vs Shared: +0.65%; MLP better on 33/64",
        fontsize=6.8,
        color=FRAME_COLOR,
        ha="left",
        va="center",
    )

    for axis in (gradient_axis, cosine_axis, monitor_axis, controller_axis):
        _style_axis(axis)
    _panel_label(gradient_axis, "a")
    _panel_label(cosine_axis, "b")
    _panel_label(controller_axis, "c")
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with Image.open(FIGURE_PATH) as image:
        image.verify()
    return FIGURE_PATH


def _plot_existing_results() -> Path:
    """Regenerate the tracked figure without rerunning any ablation."""
    results = json.loads(OUTPUT_PATH.read_text())
    return _plot_summary(
        results["gradient_rule"],
        results["coupling"],
        results["controller"],
    )


def _write_results(results):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")


def _print_results(results):
    print("## Gradient rule")
    print(
        f"Direct AD L2={results['gradient_rule']['direct_ad']['l2_norm']:.9g} "
        f"CRN-FD L2={results['gradient_rule']['crn_fd']['l2_norm']:.9g}"
    )
    print("## Coupling stability")
    for method in ("crn", "independent"):
        stats = results["coupling"][method]
        print(
            f"{method}: cosine-to-mean mean={stats['cosine_to_mean_mean']} "
            f"min={stats['cosine_to_mean_min']} "
            f"pairwise_mean={stats['pairwise_cosine_mean']}"
        )
    print("## 20-step fixed monitor")
    print(
        "CRN="
        f"{results['coupling']['crn_20_reference']['fixed_monitor']}"
    )
    independent_monitor = results["coupling"]["independent_20"]
    independent_pairs = list(
        zip(
            independent_monitor["fixed_monitor_iterations"],
            independent_monitor["fixed_monitor_objective"],
            strict=True,
        )
    )
    print(f"Independent={independent_pairs}")
    print("## Controller held-out")
    for name in ("neutral", "shared", "mlp"):
        row = results["controller"][name]
        print(
            f"{name}: monitor={row['monitor_final']:.16g} "
            f"held_out={row['held_out_mean']:.16g} "
            f"vs_neutral={row['relative_to_neutral']:+.6g}% "
            f"wins={row['wins_vs_neutral']}/64"
        )
    print(
        "MLP vs Shared: "
        f"{results['controller']['mlp_vs_shared']['relative_improvement']:+.6g}% "
        f"wins={results['controller']['mlp_vs_shared']['wins']}/64"
    )
    print("## Runtime")
    print(json.dumps(results["runtime"], indent=2))
    print(FIGURE_PATH.resolve())
    print(OUTPUT_PATH.resolve())
    print("## COMPLETE")


def main() -> int:
    torch.set_default_dtype(torch.float64)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()
    gradient_rule = _load_gradient_rule_evidence()
    m4_history = _read_m4_history()

    coupling, coupling_seconds = _run_coupling_replicates()
    coupling["crn_20_reference"] = _m4_crn_reference(m4_history)

    controller = Tesseract.from_tesseract_api(CONTROLLER_API)
    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    theta0 = _initial_theta()
    monitor_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=MONITOR_STREAM,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=0,
    )
    independent_20, independent_seconds = _train_independent(
        controller, theta0, monitor_uniforms
    )
    coupling["independent_20"] = independent_20

    shared_training, shared_seconds = _train_shared(physics, monitor_uniforms)
    final_shared = np.asarray(
        shared_training["final_coefficients"], dtype=np.float64
    )

    held_out_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=HELD_OUT_STREAM,
        forcing_seeds=H4_TEST_SEEDS,
        iteration=0,
    )
    held_out_start = time.perf_counter()
    (
        neutral_evaluation,
        shared_evaluation,
        mlp_evaluation,
    ) = _evaluate_controller_ablation(
        controller,
        m4_history["theta_history"][-1],
        final_shared,
        H4_TEST_SEEDS,
        held_out_uniforms,
    )
    held_out_seconds = time.perf_counter() - held_out_start

    if not np.isclose(
        _bank_objective(neutral_evaluation),
        M4_HELD_OUT_NEUTRAL,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("recomputed M4 neutral held-out objective differs")
    mlp_wins = int(
        np.count_nonzero(
            mlp_evaluation.seed_losses < neutral_evaluation.seed_losses
        )
    )
    if not np.isclose(
        _bank_objective(mlp_evaluation),
        M4_HELD_OUT_MLP,
        rtol=1e-12,
        atol=1e-14,
    ) or mlp_wins != M4_HELD_OUT_MLP_WINS:
        raise ValueError("recomputed M4 MLP held-out result differs")

    controller_results = {
        "neutral": _controller_statistics(
            "Neutral",
            neutral_evaluation,
            neutral_evaluation,
            m4_history["monitor_objective"][0],
        ),
        "shared": _controller_statistics(
            "Shared",
            shared_evaluation,
            neutral_evaluation,
            shared_training["fixed_monitor_objective"][-1],
        ),
        "mlp": _controller_statistics(
            "MLP",
            mlp_evaluation,
            neutral_evaluation,
            m4_history["monitor_objective"][-1],
        ),
    }
    controller_results["shared_training"] = shared_training
    controller_results["mlp_source"] = {
        "history": str(M4_HISTORY_PATH.relative_to(ROOT)),
        "retrained": False,
    }
    controller_results["mlp_vs_shared"] = {
        "relative_improvement": float(
            100.0
            * (
                _bank_objective(shared_evaluation)
                - _bank_objective(mlp_evaluation)
            )
            / _bank_objective(shared_evaluation)
        ),
        "wins": int(
            np.count_nonzero(
                mlp_evaluation.seed_losses < shared_evaluation.seed_losses
            )
        ),
        "num_forcing_conditions": len(H4_TEST_SEEDS),
    }

    figure_start = time.perf_counter()
    _plot_summary(gradient_rule, coupling, controller_results)
    figure_seconds = time.perf_counter() - figure_start
    runtime = {
        "coupling_replicates_seconds": coupling_seconds,
        "independent_20_steps_seconds": independent_seconds,
        "shared_200_steps_seconds": shared_seconds,
        "held_out_evaluation_seconds": held_out_seconds,
        "figure_seconds": figure_seconds,
        "total_seconds": time.perf_counter() - overall_start,
    }
    results = {
        "configuration": {
            "markov_base_seed": MARKOV_BASE_SEED,
            "coupling_plus_stream": COUPLING_PLUS_STREAM,
            "coupling_minus_stream": COUPLING_MINUS_STREAM,
            "independent_training_plus_stream": TRAINING_STREAM,
            "independent_training_minus_stream": INDEPENDENT_TRAINING_MINUS_STREAM,
            "monitor_stream": MONITOR_STREAM,
            "held_out_stream": HELD_OUT_STREAM,
            "num_coupling_replicates": COUPLING_REPLICATES,
            "num_realizations": NUM_REALIZATIONS,
            "fd_epsilon": FD_EPSILON,
            "learning_rate": LEARNING_RATE,
            "independent_updates": INDEPENDENT_UPDATES,
            "shared_updates": NUM_UPDATES,
            "damping": DAMPING,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "lambda_0": LAMBDA_0,
            "beta": BETA,
            "coupling_forcing_seeds": GATE_A_FORCING_SEEDS.tolist(),
            "training_seeds": H4_TRAINING_SEEDS.tolist(),
            "held_out_seeds": H4_TEST_SEEDS.tolist(),
            "numerical_zero_atol": NUMERICAL_ZERO_ATOL,
        },
        "gradient_rule": gradient_rule,
        "coupling": coupling,
        "controller": controller_results,
        "runtime": runtime,
    }
    _write_results(results)
    _print_results(results)
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--plot-only"]:
        print(_plot_existing_results().resolve())
        raise SystemExit(0)
    if arguments:
        raise SystemExit("usage: run_markov_jump_ablation.py [--plot-only]")
    raise SystemExit(main())
