"""Run the fixed 200-step stochastic Markov-jump optimization showcase."""

from __future__ import annotations

from dataclasses import dataclass
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
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from PIL import Image
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import (
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    flatten_controller_parameters,
)
from stochastic_stick_slip.engineering_markov import (
    BETA,
    DAMPING,
    FULL_FIELD_MECHANICS_SIMULATOR,
    LAMBDA_0,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    evaluate_markov_bank,
    markov_uniform_bank,
)
from stochastic_stick_slip.engineering_showcase import (
    FOURIER_BASIS,
    SYSTEM,
    forcing_batch,
)
from stochastic_stick_slip.markov_jump import (
    markov_transition_probabilities,
)
from stochastic_stick_slip.model import (
    NUM_STEPS,
    forcing_descriptor_batch,
)
from stochastic_stick_slip.showcase import full_nodal_field


CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/markov_jump_fem/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/markov_jump_long_training"
TRAINING_STREAM = 7
MONITOR_STREAM = 8
HELD_OUT_STREAM = 9
NUM_REALIZATIONS = 4
LEARNING_RATE = 0.01
NUM_UPDATES = 200
MONITOR_ITERATIONS = np.arange(0, NUM_UPDATES + 1, 10, dtype=np.int64)
REPORTED_ITERATIONS = (0, 1, 10, 20, 50, 100, 150, 200)
NUM_PHYSICAL_FRAMES = 80

INITIAL_COLOR = "#5A6069"
OPTIMIZED_COLOR = "#2F6DA4"
SAMPLED_COLOR = "#A8C4DD"
LOW_COLOR = "#7A8FAD"
HIGH_COLOR = "#D9782D"
LH_COLOR = HIGH_COLOR
HL_COLOR = OPTIMIZED_COLOR
FRAME_COLOR = "#20242A"
SLIP_EDGE_COLOR = "#B43C2F"


@dataclass(frozen=True)
class BankEvaluation:
    coefficients: np.ndarray
    losses: np.ndarray
    seed_losses: np.ndarray
    transition_counts: np.ndarray
    high_mode_fraction: np.ndarray


@dataclass(frozen=True)
class TrainingHistory:
    theta: np.ndarray
    train_objective: np.ndarray
    gradient_norm: np.ndarray
    monitor_iterations: np.ndarray
    monitor_objective: np.ndarray


@dataclass(frozen=True)
class Replay:
    representative_seed: int
    displacement: np.ndarray
    velocity: np.ndarray
    modes: np.ndarray
    slip: np.ndarray
    probability_low_to_high: np.ndarray
    probability_high_to_low: np.ndarray
    initial_full_displacement: np.ndarray
    optimized_full_displacement: np.ndarray


def _seed_batches(seeds):
    seeds = np.asarray(seeds, dtype=np.int64)
    for start in range(0, len(seeds), 8):
        yield seeds[start : start + 8]


def _initial_theta() -> np.ndarray:
    return np.asarray(
        flatten_controller_parameters(build_controller()).detach(),
        dtype=np.float64,
    )


def _create_tesseracts():
    return (
        Tesseract.from_tesseract_api(CONTROLLER_API),
        Tesseract.from_tesseract_api(PHYSICS_API),
    )


def _controller_coefficients(controller, theta, seeds):
    return np.asarray(
        controller.apply(
            {
                "theta": np.asarray(theta, dtype=np.float64),
                "descriptors": forcing_descriptor_batch(seeds),
            }
        )["coeffs"],
        dtype=np.float64,
    )


def _differentiable_training_loss(
    controller,
    physics,
    theta,
    seeds,
    uniforms,
):
    losses = []
    for start, batch_seeds in zip(
        range(0, len(seeds), 8),
        _seed_batches(seeds),
        strict=True,
    ):
        coefficients = apply_tesseract(
            controller,
            {
                "theta": theta,
                "descriptors": forcing_descriptor_batch(batch_seeds),
            },
        )["coeffs"]
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


def _evaluate_bank(controller, theta, seeds, uniforms) -> BankEvaluation:
    collected = {
        "coefficients": [],
        "losses": [],
        "transition_counts": [],
        "high_mode_fraction": [],
    }
    for start, batch_seeds in zip(
        range(0, len(seeds), 8),
        _seed_batches(seeds),
        strict=True,
    ):
        coefficients = _controller_coefficients(
            controller, theta, batch_seeds
        )
        result = evaluate_markov_bank(
            coefficients,
            forcing_batch(batch_seeds),
            uniforms[start : start + 8],
        )
        collected["coefficients"].append(coefficients)
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
        raise FloatingPointError("hard Markov bank evaluation is non-finite")
    return BankEvaluation(
        coefficients=arrays["coefficients"],
        losses=arrays["losses"],
        seed_losses=np.mean(arrays["losses"], axis=1),
        transition_counts=arrays["transition_counts"],
        high_mode_fraction=arrays["high_mode_fraction"],
    )


def _bank_objective(evaluation: BankEvaluation) -> float:
    return float(np.mean(evaluation.seed_losses))


def _save_training_history(history: TrainingHistory) -> Path:
    path = OUTPUT_DIRECTORY / "training_history.npz"
    np.savez(
        path,
        theta_history=history.theta,
        train_objective=history.train_objective,
        gradient_norm=history.gradient_norm,
        monitor_iterations=history.monitor_iterations,
        monitor_objective=history.monitor_objective,
    )
    return path


def _train(controller, physics, theta0, monitor_uniforms):
    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    theta_history = np.empty(
        (NUM_UPDATES + 1, NUM_CONTROLLER_PARAMETERS), dtype=np.float64
    )
    train_objective = np.empty(NUM_UPDATES + 1, dtype=np.float64)
    gradient_norm = np.empty(NUM_UPDATES, dtype=np.float64)
    monitor_objective = np.empty(len(MONITOR_ITERATIONS), dtype=np.float64)
    theta_history[0] = theta0

    initial_training_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=TRAINING_STREAM,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=0,
    )
    train_objective[0] = _bank_objective(
        _evaluate_bank(
            controller,
            theta0,
            H4_TRAINING_SEEDS,
            initial_training_uniforms,
        )
    )
    monitor_seconds = 0.0
    monitor_start = time.perf_counter()
    monitor_objective[0] = _bank_objective(
        _evaluate_bank(
            controller,
            theta0,
            H4_TRAINING_SEEDS,
            monitor_uniforms,
        )
    )
    monitor_seconds += time.perf_counter() - monitor_start

    first_backward_seconds = None
    monitor_index = 1
    training_start = time.perf_counter()
    for iteration in range(1, NUM_UPDATES + 1):
        uniforms = markov_uniform_bank(
            NUM_REALIZATIONS,
            stream_id=TRAINING_STREAM,
            forcing_seeds=H4_TRAINING_SEEDS,
            iteration=iteration,
        )
        optimizer.zero_grad(set_to_none=True)
        backward_start = time.perf_counter()
        loss = _differentiable_training_loss(
            controller,
            physics,
            theta,
            H4_TRAINING_SEEDS,
            uniforms,
        )
        loss.backward()
        backward_seconds = time.perf_counter() - backward_start
        if first_backward_seconds is None:
            first_backward_seconds = backward_seconds
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("theta gradient is non-finite")
        gradient_norm[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        theta_array = theta.detach().numpy().copy()
        theta_history[iteration] = theta_array
        post_step = _evaluate_bank(
            controller,
            theta_array,
            H4_TRAINING_SEEDS,
            uniforms,
        )
        train_objective[iteration] = _bank_objective(post_step)

        monitor_value = None
        if iteration % 10 == 0:
            monitor_start = time.perf_counter()
            monitor_value = _bank_objective(
                _evaluate_bank(
                    controller,
                    theta_array,
                    H4_TRAINING_SEEDS,
                    monitor_uniforms,
                )
            )
            monitor_seconds += time.perf_counter() - monitor_start
            monitor_objective[monitor_index] = monitor_value
            monitor_index += 1
        message = (
            f"iteration={iteration:03d} "
            f"sampled={train_objective[iteration]:.16g} "
            f"gradient_norm={gradient_norm[iteration - 1]:.9g}"
        )
        if monitor_value is not None:
            message += f" monitor={monitor_value:.16g}"
        print(message, flush=True)

    training_seconds = time.perf_counter() - training_start
    if monitor_index != len(MONITOR_ITERATIONS):
        raise AssertionError("monitor history is incomplete")
    history = TrainingHistory(
        theta=theta_history,
        train_objective=train_objective,
        gradient_norm=gradient_norm,
        monitor_iterations=MONITOR_ITERATIONS.copy(),
        monitor_objective=monitor_objective,
    )
    for value in (
        history.theta,
        history.train_objective,
        history.gradient_norm,
        history.monitor_objective,
    ):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("training history is non-finite")
    return history, {
        "first_backward_seconds": float(first_backward_seconds),
        "training_seconds": training_seconds,
        "monitor_seconds": monitor_seconds,
    }


def _held_out_statistics(initial, optimized):
    initial_objective = _bank_objective(initial)
    optimized_objective = _bank_objective(optimized)
    relative_improvement = 100.0 * (
        initial.seed_losses - optimized.seed_losses
    ) / initial.seed_losses
    return {
        "initial_mean": initial_objective,
        "optimized_mean": optimized_objective,
        "relative_improvement": 100.0
        * (initial_objective - optimized_objective)
        / initial_objective,
        "median_improvement": float(np.median(relative_improvement)),
        "improved_count": int(
            np.count_nonzero(optimized.seed_losses < initial.seed_losses)
        ),
        "per_forcing_improvement": relative_improvement,
    }


def _markov_statistics(evaluation):
    return {
        "mean_transitions": np.mean(
            evaluation.transition_counts, axis=(0, 1)
        ).tolist(),
        "mean_high_occupancy": np.mean(
            evaluation.high_mode_fraction, axis=(0, 1)
        ).tolist(),
    }


def _representative_index(improvement):
    distance = np.abs(improvement - np.median(improvement))
    order = np.lexsort((H4_TEST_SEEDS, distance))
    return int(order[0])


def _representative_coefficients(controller, theta, index):
    batch_start = (index // 8) * 8
    batch = H4_TEST_SEEDS[batch_start : batch_start + 8]
    coefficients = _controller_coefficients(controller, theta, batch)
    return coefficients[index - batch_start]


def _transition_probabilities(coefficients):
    low_to_high, high_to_low = markov_transition_probabilities(
        coefficients[None, :],
        FOURIER_BASIS,
        SYSTEM.time_step,
        LAMBDA_0,
        BETA,
    )
    return np.asarray(low_to_high[0]), np.asarray(high_to_low[0])


def _full_field(forcing, preload):
    outputs = FULL_FIELD_MECHANICS_SIMULATOR(
        DAMPING,
        forcing,
        preload,
    )
    return tuple(np.asarray(output) for output in outputs)


def _replay(controller, history, held_out_uniforms, improvement):
    index = _representative_index(improvement)
    seed = int(H4_TEST_SEEDS[index])
    forcing = forcing_batch(np.asarray([seed], dtype=np.int64))
    uniforms = held_out_uniforms[index : index + 1, :1]
    displacement = np.empty((NUM_UPDATES + 1, NUM_STEPS))
    velocity = np.empty_like(displacement)
    modes = np.empty(
        (NUM_UPDATES + 1, NUM_STEPS, 2), dtype=np.bool_
    )
    slip = np.empty_like(modes)
    probability_low_to_high = np.empty_like(displacement)
    probability_high_to_low = np.empty_like(displacement)
    endpoints = {}

    start = time.perf_counter()
    for iteration, theta in enumerate(history.theta):
        coefficients = _representative_coefficients(
            controller, theta, index
        )
        result = evaluate_markov_bank(
            coefficients[None, :], forcing, uniforms
        )
        displacement[iteration] = np.asarray(result.displacement[0, 0])
        velocity[iteration] = np.asarray(result.velocity[0, 0])
        modes[iteration] = np.asarray(result.modes[0, 0])
        slip[iteration] = np.asarray(result.slip[0, 0])
        (
            probability_low_to_high[iteration],
            probability_high_to_low[iteration],
        ) = _transition_probabilities(coefficients)
        if iteration in (0, NUM_UPDATES):
            endpoints[iteration] = _full_field(
                forcing,
                np.asarray(result.preload[0]),
            )
            if not np.allclose(
                endpoints[iteration][0][0],
                displacement[iteration],
                rtol=1e-12,
                atol=1e-14,
            ) or not np.array_equal(
                endpoints[iteration][2][0], slip[iteration]
            ):
                raise AssertionError(
                    "full-field replay and Markov batch forward differ"
                )
        if iteration % 25 == 0 or iteration == NUM_UPDATES:
            print(
                f"representative_replay={iteration:03d}/{NUM_UPDATES}",
                flush=True,
            )
    replay_seconds = time.perf_counter() - start
    replay = Replay(
        representative_seed=seed,
        displacement=displacement,
        velocity=velocity,
        modes=modes,
        slip=slip,
        probability_low_to_high=probability_low_to_high,
        probability_high_to_low=probability_high_to_low,
        initial_full_displacement=full_nodal_field(endpoints[0][5][0]),
        optimized_full_displacement=full_nodal_field(
            endpoints[NUM_UPDATES][5][0]
        ),
    )
    return replay, replay_seconds


def _configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 1.1,
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
        width=1.0,
        colors=FRAME_COLOR,
    )


def _expanded_limits(values, fraction=0.06):
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    margin = fraction * (span if span > 0.0 else max(abs(maximum), 1.0))
    return minimum - margin, maximum + margin


def _panel_labels(axes):
    for label, axis in zip("abc", axes, strict=True):
        axis.text(
            -0.11,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )


def _plot_optimization_history(history):
    iterations = np.arange(NUM_UPDATES + 1)
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.plot(
        iterations,
        history.train_objective,
        color=SAMPLED_COLOR,
        lw=1.0,
        alpha=0.75,
        label="Sampled training",
    )
    axis.plot(
        history.monitor_iterations,
        history.monitor_objective,
        color=OPTIMIZED_COLOR,
        lw=2.0,
        marker="o",
        ms=4.2,
        label="Fixed monitor",
    )
    axis.set(
        xlim=(0, NUM_UPDATES),
        xlabel="Adam iteration",
        ylabel="Mean-square displacement",
    )
    axis.legend(loc="best")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "optimization_history.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_held_out_distribution(initial, optimized, statistics):
    improvement = statistics["per_forcing_improvement"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )
    violin = axes[0].violinplot(
        [initial.seed_losses, optimized.seed_losses],
        positions=[0, 1],
        widths=0.68,
        showextrema=False,
    )
    for body, color in zip(
        violin["bodies"], [INITIAL_COLOR, OPTIMIZED_COLOR], strict=True
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.2)
    box = axes[0].boxplot(
        [initial.seed_losses, optimized.seed_losses],
        positions=[0, 1],
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": FRAME_COLOR, "linewidth": 1.1},
        whiskerprops={"color": FRAME_COLOR, "linewidth": 1.0},
        capprops={"color": FRAME_COLOR, "linewidth": 1.0},
        medianprops={"color": FRAME_COLOR, "linewidth": 1.4},
    )
    for patch, color in zip(
        box["boxes"], [INITIAL_COLOR, OPTIMIZED_COLOR], strict=True
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    jitter = 0.08 * np.sin(np.arange(64) * 2.399963229728653)
    axes[0].scatter(
        jitter,
        initial.seed_losses,
        s=12,
        color=INITIAL_COLOR,
        alpha=0.62,
        edgecolor="none",
    )
    axes[0].scatter(
        1.0 + jitter,
        optimized.seed_losses,
        s=12,
        color=OPTIMIZED_COLOR,
        alpha=0.62,
        edgecolor="none",
    )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Initial", "Iteration 200"],
        ylabel="Mean-square displacement",
    )
    axes[0].text(
        0.04,
        0.97,
        f"Mean: {statistics['relative_improvement']:+.2f}%\n"
        f"{statistics['improved_count']}/64 improved",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color=FRAME_COLOR,
    )

    lower = min(float(np.min(improvement)), 0.0)
    upper = max(float(np.max(improvement)), 0.0)
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    edges = np.linspace(lower, upper, 13)
    axes[1].hist(
        improvement,
        bins=edges,
        color=OPTIMIZED_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].axvline(0.0, color=FRAME_COLOR, lw=1.3)
    axes[1].set(
        xlabel="Relative improvement (%)",
        ylabel="Forcing conditions",
    )
    axes[1].text(
        0.97,
        0.97,
        f"Median: {statistics['median_improvement']:+.2f}%",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=FRAME_COLOR,
    )
    for axis in axes:
        _style_axis(axis)
    for label, axis in zip("ab", axes, strict=True):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    path = OUTPUT_DIRECTORY / "held_out_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_representative_response(replay):
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 0.85]},
    )
    axes[0].plot(
        periods,
        replay.displacement[0],
        color=INITIAL_COLOR,
        lw=1.4,
        label="Initial",
    )
    axes[0].plot(
        periods,
        replay.displacement[-1],
        color=OPTIMIZED_COLOR,
        lw=1.6,
        label="Iteration 200",
    )
    axes[0].set_ylabel("Displacement")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(
        periods,
        replay.probability_low_to_high[-1],
        color=LH_COLOR,
        lw=1.6,
        label="LOW to HIGH",
    )
    axes[1].plot(
        periods,
        replay.probability_high_to_low[-1],
        color=HL_COLOR,
        lw=1.6,
        label="HIGH to LOW",
    )
    axes[1].set_ylabel("Probability")
    axes[1].legend(loc="upper right", ncol=2)

    mode_image = np.stack(
        (
            replay.modes[0, :, 0],
            replay.modes[-1, :, 0],
            replay.modes[0, :, 1],
            replay.modes[-1, :, 1],
        )
    )
    axes[2].imshow(
        mode_image,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([LOW_COLOR, HIGH_COLOR]),
        vmin=0,
        vmax=1,
        extent=(periods[0], periods[-1], 3.5, -0.5),
    )
    axes[2].set(
        yticks=[0, 1, 2, 3],
        yticklabels=["A initial", "A iter 200", "B initial", "B iter 200"],
        xlabel="Time (periods)",
        ylabel="Mode",
    )
    for axis in axes:
        _style_axis(axis)
    _panel_labels(axes)
    path = OUTPUT_DIRECTORY / "representative_markov_response.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _quantized_frame(fig, colors=112):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3]).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT
    )


def _save_gif(frames, path, duration):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return path


def _create_optimizer_gif(history, replay):
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    iterations = np.arange(NUM_UPDATES + 1)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 7.0),
        dpi=76,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.1]},
    )
    sampled_line, = axes[0].plot(
        [], [], color=SAMPLED_COLOR, lw=0.9, alpha=0.75,
        label="Sampled training",
    )
    monitor_line, = axes[0].plot(
        [], [], color=OPTIMIZED_COLOR, lw=2.0, marker="o", ms=3.8,
        label="Fixed monitor",
    )
    axes[0].set(
        xlim=(0, NUM_UPDATES),
        ylim=_expanded_limits(
            np.concatenate(
                (history.train_objective, history.monitor_objective)
            )
        ),
        ylabel="Objective",
    )
    axes[0].legend(loc="best", ncol=2)

    low_to_high_line, = axes[1].plot(
        [], [], color=LH_COLOR, lw=1.6, label="LOW to HIGH"
    )
    high_to_low_line, = axes[1].plot(
        [], [], color=HL_COLOR, lw=1.6, label="HIGH to LOW"
    )
    all_probability = np.concatenate(
        (
            replay.probability_low_to_high.ravel(),
            replay.probability_high_to_low.ravel(),
        )
    )
    axes[1].set(
        xlim=(periods[0], periods[-1]),
        ylim=_expanded_limits(all_probability),
        ylabel="Probability",
    )
    axes[1].legend(loc="upper right", ncol=2)

    axes[2].plot(
        periods,
        replay.displacement[0],
        color=INITIAL_COLOR,
        lw=1.3,
        label="Initial",
    )
    current_line, = axes[2].plot(
        [], [], color=OPTIMIZED_COLOR, lw=1.6, label="Current"
    )
    axes[2].set(
        xlim=(periods[0], periods[-1]),
        ylim=_expanded_limits(replay.displacement),
        xlabel="Time (periods)",
        ylabel="Displacement",
    )
    axes[2].legend(loc="upper right", ncol=2)
    for axis in axes:
        _style_axis(axis)
    annotation = fig.suptitle("", fontsize=10)
    frames = []
    for iteration in iterations:
        sampled_line.set_data(
            iterations[: iteration + 1],
            history.train_objective[: iteration + 1],
        )
        visible_monitor = history.monitor_iterations <= iteration
        monitor_line.set_data(
            history.monitor_iterations[visible_monitor],
            history.monitor_objective[visible_monitor],
        )
        low_to_high_line.set_data(
            periods, replay.probability_low_to_high[iteration]
        )
        high_to_low_line.set_data(
            periods, replay.probability_high_to_low[iteration]
        )
        current_line.set_data(periods, replay.displacement[iteration])
        latest_monitor = history.monitor_objective[
            np.flatnonzero(visible_monitor)[-1]
        ]
        annotation.set_text(
            f"Iteration {iteration:03d}   "
            f"sampled = {history.train_objective[iteration]:.7f}   "
            f"monitor = {latest_monitor:.7f}"
        )
        frames.append(_quantized_frame(fig))
        if iteration % 25 == 0 or iteration == NUM_UPDATES:
            print(
                f"optimizer_gif_frame={iteration:03d}/{NUM_UPDATES}",
                flush=True,
            )
    plt.close(fig)
    return _save_gif(
        frames,
        OUTPUT_DIRECTORY / "optimization_all_iterations.gif",
        duration=55,
    )


def _create_deformation_gif(replay):
    initial = replay.initial_full_displacement
    optimized = replay.optimized_full_displacement
    magnitude = np.concatenate(
        (
            np.linalg.norm(initial, axis=2).ravel(),
            np.linalg.norm(optimized, axis=2).ravel(),
        )
    )
    magnitude_max = float(np.max(magnitude))
    deformation_scale = 0.12 / magnitude_max
    norm = Normalize(0.0, magnitude_max)
    frame_indices = np.linspace(
        0, NUM_STEPS - 1, NUM_PHYSICAL_FRAMES, dtype=np.int64
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 2.8), dpi=95, constrained_layout=True
    )
    collections = []
    contact_artists = []
    for axis, label in zip(
        axes, ("Initial", "Iteration 200"), strict=True
    ):
        collection = PolyCollection(
            SYSTEM.points[SYSTEM.cells],
            array=np.zeros(len(SYSTEM.cells)),
            cmap="viridis",
            norm=norm,
            edgecolors="#4C5560",
            linewidths=0.35,
        )
        axis.add_collection(collection)
        artists = [
            axis.scatter([], [], s=40, linewidth=1.35, zorder=5)
            for _ in range(2)
        ]
        axis.text(
            0.03,
            0.94,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axis.set(xlim=(-0.03, 1.04), ylim=(-0.18, 0.28), xlabel="x")
        axis.set_aspect("equal")
        _style_axis(axis)
        collections.append(collection)
        contact_artists.append(artists)
    axes[0].set_ylabel("y")
    colorbar = fig.colorbar(
        collections[-1], ax=axes, fraction=0.028, pad=0.02
    )
    colorbar.set_label("Displacement magnitude")
    annotation = fig.text(
        0.5, 0.995, "", ha="center", va="top", fontsize=9
    )
    frames = []
    for step in frame_indices:
        for panel, (field, mode, slip) in enumerate(
            (
                (initial, replay.modes[0], replay.slip[0]),
                (optimized, replay.modes[-1], replay.slip[-1]),
            )
        ):
            deformed = SYSTEM.points + deformation_scale * field[step]
            collections[panel].set_verts(deformed[SYSTEM.cells])
            collections[panel].set_array(
                np.mean(
                    np.linalg.norm(field[step], axis=1)[SYSTEM.cells],
                    axis=1,
                )
            )
            for contact, artist in enumerate(contact_artists[panel]):
                artist.set_offsets(
                    deformed[SYSTEM.contact_nodes[contact]][None, :]
                )
                artist.set_facecolor(
                    HIGH_COLOR if mode[step, contact] else LOW_COLOR
                )
                artist.set_edgecolor(
                    SLIP_EDGE_COLOR if slip[step, contact] else FRAME_COLOR
                )
        periods = float(
            SYSTEM.times[step] * SYSTEM.omega_1 / (2.0 * np.pi)
        )
        annotation.set_text(
            f"Held-out seed {replay.representative_seed}   "
            f"time = {periods:.2f} periods   "
            "fill: LOW/HIGH, edge: STICK/SLIP"
        )
        frames.append(_quantized_frame(fig, colors=128))
    plt.close(fig)
    path = _save_gif(
        frames,
        OUTPUT_DIRECTORY / "initial_vs_optimized_deformation.gif",
        duration=85,
    )
    return path, deformation_scale, magnitude_max


def _validate_media(paths, expected_gif_frames):
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    for path, expected_frames in expected_gif_frames.items():
        with Image.open(path) as image:
            if image.n_frames != expected_frames:
                raise RuntimeError(
                    f"{path} has {image.n_frames} frames, "
                    f"expected {expected_frames}"
                )


def _print_milestones(history):
    print("iter | sampled train | fixed monitor")
    for iteration in REPORTED_ITERATIONS:
        monitor_matches = np.flatnonzero(
            history.monitor_iterations == iteration
        )
        monitor_text = (
            f"{history.monitor_objective[monitor_matches[0]]:.16g}"
            if monitor_matches.size
            else "-"
        )
        print(
            f"{iteration:03d} | "
            f"{history.train_objective[iteration]:.16g} | "
            f"{monitor_text}"
        )


def main() -> int:
    torch.set_default_dtype(torch.float64)
    _configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    controller, physics = _create_tesseracts()
    theta0 = _initial_theta()
    if theta0.shape != (NUM_CONTROLLER_PARAMETERS,):
        raise ValueError("unexpected initial controller shape")
    monitor_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=MONITOR_STREAM,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=0,
    )

    history, runtime = _train(
        controller, physics, theta0, monitor_uniforms
    )
    history_path = _save_training_history(history)

    held_out_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS,
        stream_id=HELD_OUT_STREAM,
        forcing_seeds=H4_TEST_SEEDS,
        iteration=0,
    )
    held_out_start = time.perf_counter()
    held_out_initial = _evaluate_bank(
        controller, theta0, H4_TEST_SEEDS, held_out_uniforms
    )
    held_out_optimized = _evaluate_bank(
        controller, history.theta[-1], H4_TEST_SEEDS, held_out_uniforms
    )
    runtime["held_out_seconds"] = time.perf_counter() - held_out_start
    held_out = _held_out_statistics(
        held_out_initial, held_out_optimized
    )
    initial_markov = _markov_statistics(held_out_initial)
    optimized_markov = _markov_statistics(held_out_optimized)

    replay, runtime["replay_seconds"] = _replay(
        controller,
        history,
        held_out_uniforms,
        held_out["per_forcing_improvement"],
    )
    figure_paths = [
        _plot_optimization_history(history),
        _plot_held_out_distribution(
            held_out_initial, held_out_optimized, held_out
        ),
        _plot_representative_response(replay),
    ]
    gif_start = time.perf_counter()
    optimizer_gif = _create_optimizer_gif(history, replay)
    (
        deformation_gif,
        deformation_scale,
        displacement_color_max,
    ) = _create_deformation_gif(replay)
    runtime["gif_seconds"] = time.perf_counter() - gif_start
    media_paths = figure_paths + [optimizer_gif, deformation_gif]
    _validate_media(
        media_paths,
        {
            optimizer_gif: NUM_UPDATES + 1,
            deformation_gif: NUM_PHYSICAL_FRAMES,
        },
    )

    summary = {
        "training": {
            "sampled_initial": float(history.train_objective[0]),
            "sampled_final": float(history.train_objective[-1]),
            "monitor_initial": float(history.monitor_objective[0]),
            "monitor_final": float(history.monitor_objective[-1]),
        },
        "held_out": {
            key: value
            for key, value in held_out.items()
            if key != "per_forcing_improvement"
        },
        "markov": {
            "initial": initial_markov,
            "optimized": optimized_markov,
        },
        "representative_seed": replay.representative_seed,
        "runtime": runtime,
        "deformation_scale": deformation_scale,
        "displacement_color_max": displacement_color_max,
    }
    print("## Training")
    _print_milestones(history)
    print("## Held-out")
    print(json.dumps(summary["held_out"], indent=2))
    print("## Markov response")
    print(json.dumps(summary["markov"], indent=2))
    print(f"representative_seed={replay.representative_seed}")
    print("## Runtime")
    print(json.dumps(runtime, indent=2))
    print("## Visualization")
    for path in media_paths:
        print(path.resolve())
    print(history_path.resolve())
    print("## COMPLETE")
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit(
            "usage: run_markov_jump_long_training.py"
        )
    raise SystemExit(main())
