"""Run the locked 32x4, 500-iteration final Hackathon showcase."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import build_controller, flatten_controller_parameters
from stochastic_stick_slip.model import forcing_descriptor_batch
from stochastic_stick_slip.showcase import (
    NUM_ELEMENTS_X,
    NUM_ELEMENTS_Y,
    NUM_STEPS,
    SYSTEM,
    evaluate_full_trajectory,
    full_nodal_field,
    preload_history,
)

CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/showcase"
BASE_Q = np.array([0.2, 0.04], dtype=np.float64)
LEARNING_RATE = 0.01
MAX_ITERATIONS = 500
MILESTONE_ITERATIONS = (0, 20, 100, 200, 300, 400, 500)
NUM_PHYSICAL_FRAMES = 80

FIXED_COLOR = "#5A6069"
MLP_COLOR = "#2F6DA4"
ACCENT_COLOR = "#D9782D"
FRAME_COLOR = "#20242A"
STICK_COLOR = "#14345F"
SLIP_COLOR = "#EB6A2A"


@dataclass(frozen=True)
class Evaluation:
    losses: np.ndarray
    coefficients: np.ndarray
    stick_to_slip: np.ndarray
    slip_to_stick: np.ndarray
    displacement_min: np.ndarray
    displacement_max: np.ndarray
    velocity_min: np.ndarray
    velocity_max: np.ndarray


@dataclass(frozen=True)
class TrainingHistory:
    theta: np.ndarray
    objective: np.ndarray
    gradient: np.ndarray
    n_min: np.ndarray
    n_max: np.ndarray


def _seed_batches(seeds):
    for start in range(0, len(seeds), 8):
        yield seeds[start : start + 8]


def _initial_theta() -> torch.Tensor:
    return flatten_controller_parameters(build_controller()).detach().clone()


def create_tesseracts():
    controller = Tesseract.from_tesseract_api(CONTROLLER_API)
    previous_variant = os.environ.get("STICK_SLIP_FEM_VARIANT")
    os.environ["STICK_SLIP_FEM_VARIANT"] = "showcase"
    try:
        physics = Tesseract.from_tesseract_api(PHYSICS_API)
    finally:
        if previous_variant is None:
            os.environ.pop("STICK_SLIP_FEM_VARIANT", None)
        else:
            os.environ["STICK_SLIP_FEM_VARIANT"] = previous_variant
    return controller, physics


def _differentiable_batch_losses(controller, physics, theta, seeds):
    coefficients = apply_tesseract(
        controller,
        {"theta": theta, "descriptors": forcing_descriptor_batch(seeds)},
    )["coeffs"]
    return apply_tesseract(
        physics,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds},
    )["seed_losses"]


def full_training_loss(controller, physics, theta):
    losses = [
        _differentiable_batch_losses(controller, physics, theta, seeds)
        for seeds in _seed_batches(H4_TRAINING_SEEDS)
    ]
    return torch.cat(losses).mean()


def _controller_coefficients(controller, theta, seeds):
    return np.asarray(
        controller.apply(
            {
                "theta": np.asarray(theta, dtype=np.float64),
                "descriptors": forcing_descriptor_batch(seeds),
            }
        )["coeffs"]
    )


def evaluate(controller, physics, theta, seeds, fixed=False) -> Evaluation:
    collected = {name: [] for name in (
        "losses", "coefficients", "stick_to_slip", "slip_to_stick",
        "displacement_min", "displacement_max", "velocity_min", "velocity_max",
    )}
    for batch_seeds in _seed_batches(seeds):
        coefficients = (
            np.zeros((8, 5), dtype=np.float64)
            if fixed
            else _controller_coefficients(controller, theta, batch_seeds)
        )
        response = physics.apply(
            {"q": BASE_Q, "coeffs": coefficients, "seeds": batch_seeds}
        )
        collected["losses"].append(np.asarray(response["seed_losses"]))
        collected["coefficients"].append(coefficients)
        for name in (
            "stick_to_slip", "slip_to_stick", "displacement_min",
            "displacement_max", "velocity_min", "velocity_max",
        ):
            collected[name].append(np.asarray(response[name]))
    return Evaluation(**{name: np.concatenate(values) for name, values in collected.items()})


def _switching_gate(evaluation: Evaluation) -> bool:
    return bool(
        np.all(np.sum(evaluation.stick_to_slip, axis=0) > 0)
        and np.all(np.sum(evaluation.slip_to_stick, axis=0) > 0)
    )


def _control_range(coefficients):
    histories = np.asarray(preload_history(BASE_Q[1], coefficients))
    return float(np.min(histories)), float(np.max(histories))


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def preflight(controller, physics):
    fixed_start = time.perf_counter()
    fixed = evaluate(
        controller, physics, np.asarray(_initial_theta()), H4_TRAINING_SEEDS, True
    )
    fixed_seconds = time.perf_counter() - fixed_start
    theta = torch.nn.Parameter(_initial_theta())
    backward_start = time.perf_counter()
    loss = full_training_loss(controller, physics, theta)
    loss.backward()
    backward_seconds = time.perf_counter() - backward_start
    gradient = theta.grad.detach().cpu().numpy()
    gate = bool(
        np.all(np.isfinite(fixed.losses))
        and np.isfinite(float(loss.detach()))
        and gradient.shape == (469,)
        and np.all(np.isfinite(gradient))
        and np.linalg.norm(gradient) > 0.0
        and _switching_gate(fixed)
    )
    return {
        "fixed": fixed,
        "fixed_seconds": fixed_seconds,
        "backward_loss": float(loss.detach()),
        "backward_gradient_norm": float(np.linalg.norm(gradient)),
        "backward_seconds": backward_seconds,
        "peak_rss_gib": _peak_rss_gib(),
        "gate": gate,
    }


def _save_training_history(history: TrainingHistory):
    path = OUTPUT_DIRECTORY / "training_history.npz"
    np.savez(
        path,
        theta_history=history.theta,
        objective_history=history.objective,
        gradient_history=history.gradient,
        n_min_history=history.n_min,
        n_max_history=history.n_max,
    )
    return path


def train(controller, physics, initial_objective):
    theta = torch.nn.Parameter(_initial_theta())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    theta_history = np.empty((MAX_ITERATIONS + 1, 469), dtype=np.float64)
    objective_history = np.empty(MAX_ITERATIONS + 1, dtype=np.float64)
    gradient_history = np.empty(MAX_ITERATIONS, dtype=np.float64)
    n_min_history = np.empty(MAX_ITERATIONS + 1, dtype=np.float64)
    n_max_history = np.empty(MAX_ITERATIONS + 1, dtype=np.float64)
    theta_history[0] = theta.detach().cpu().numpy()
    objective_history[0] = initial_objective
    n_min_history[0] = BASE_Q[1]
    n_max_history[0] = BASE_Q[1]

    start = time.perf_counter()
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = full_training_loss(controller, physics, theta)
        loss.backward()
        gradient_norm = float(torch.linalg.vector_norm(theta.grad))
        optimizer.step()
        theta_array = theta.detach().cpu().numpy()
        hard = evaluate(controller, physics, theta_array, H4_TRAINING_SEEDS)
        objective = float(np.mean(hard.losses))
        preload_min, preload_max = _control_range(hard.coefficients)
        theta_history[iteration] = theta_array
        objective_history[iteration] = objective
        gradient_history[iteration - 1] = gradient_norm
        n_min_history[iteration] = preload_min
        n_max_history[iteration] = preload_max
        print(
            f"iteration={iteration:03d} objective={objective:.16g} "
            f"gradient_norm={gradient_norm:.9g} "
            f"N=[{preload_min:.7f},{preload_max:.7f}]",
            flush=True,
        )
    history = TrainingHistory(
        theta=theta_history,
        objective=objective_history,
        gradient=gradient_history,
        n_min=n_min_history,
        n_max=n_max_history,
    )
    return history, time.perf_counter() - start, _save_training_history(history)


def _trajectory_arrays(result):
    free_displacement = np.asarray(result.displacement)
    free_velocity = np.asarray(result.velocity)
    return {
        "free_displacement": free_displacement,
        "free_velocity": free_velocity,
        "displacement": full_nodal_field(free_displacement),
        "velocity": full_nodal_field(free_velocity),
        "slip": np.asarray(result.slip, dtype=np.int64),
        "stick_to_slip": np.asarray(result.stick_to_slip, dtype=np.int64),
        "slip_to_stick": np.asarray(result.slip_to_stick, dtype=np.int64),
    }


def _observation_history(free_displacement):
    return np.asarray(free_displacement) @ np.asarray(SYSTEM.observation)


def _representative_coefficients(controller, theta, seed):
    index = int(np.flatnonzero(H4_TEST_SEEDS == seed)[0])
    batch_start = (index // 8) * 8
    batch = H4_TEST_SEEDS[batch_start : batch_start + 8]
    coefficients = _controller_coefficients(controller, theta, batch)
    return coefficients[index - batch_start]


def replay_representative(controller, history, fixed_test, trained_test):
    relative_improvement = (fixed_test.losses - trained_test.losses) / fixed_test.losses
    median_improvement = float(np.median(relative_improvement))
    distance = np.abs(relative_improvement - median_improvement)
    representative_index = int(np.flatnonzero(distance == np.min(distance))[0])
    representative_seed = int(H4_TEST_SEEDS[representative_index])
    start = time.perf_counter()
    fixed_result = evaluate_full_trajectory(
        BASE_Q, np.zeros(5, dtype=np.float64), representative_seed
    )
    fixed = _trajectory_arrays(fixed_result)
    fixed_observation = _observation_history(fixed["free_displacement"])
    fixed_velocity = _observation_history(fixed["free_velocity"])
    displacement = np.empty((MAX_ITERATIONS + 1, NUM_STEPS), dtype=np.float64)
    velocity = np.empty_like(displacement)
    preload = np.empty_like(displacement)
    slip = np.empty((MAX_ITERATIONS + 1, NUM_STEPS, 2), dtype=np.int8)
    final = final_result = final_coefficients = None
    for iteration, theta in enumerate(history.theta):
        coefficients = _representative_coefficients(controller, theta, representative_seed)
        result = evaluate_full_trajectory(BASE_Q, coefficients, representative_seed)
        arrays = _trajectory_arrays(result)
        displacement[iteration] = _observation_history(arrays["free_displacement"])
        velocity[iteration] = _observation_history(arrays["free_velocity"])
        preload[iteration] = np.asarray(
            preload_history(BASE_Q[1], coefficients[None, :])
        )[0]
        slip[iteration] = arrays["slip"]
        if iteration == MAX_ITERATIONS:
            final, final_result, final_coefficients = arrays, result, coefficients
        if iteration % 50 == 0 or iteration == MAX_ITERATIONS:
            print(f"representative_replay={iteration:03d}/{MAX_ITERATIONS}", flush=True)
    replay_path = OUTPUT_DIRECTORY / "representative_replay.npz"
    np.savez(
        replay_path,
        representative_seed=representative_seed,
        times=np.asarray(SYSTEM.times),
        fixed_displacement=fixed_observation,
        fixed_velocity=fixed_velocity,
        fixed_slip=fixed["slip"],
        displacement_history=displacement,
        velocity_history=velocity,
        preload_history=preload,
        slip_history=slip,
    )
    return {
        "representative_seed": representative_seed,
        "representative_index": representative_index,
        "median_improvement": median_improvement,
        "relative_improvement": relative_improvement,
        "fixed": fixed,
        "fixed_result": fixed_result,
        "fixed_observation": fixed_observation,
        "fixed_velocity": fixed_velocity,
        "final": final,
        "final_result": final_result,
        "final_coefficients": final_coefficients,
        "displacement": displacement,
        "velocity": velocity,
        "preload": preload,
        "slip": slip,
        "seconds": time.perf_counter() - start,
        "path": replay_path,
    }


def _configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
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
    axis.tick_params(top=True, right=True, direction="in", width=1.0)


def _expanded_limits(values, fraction=0.06):
    minimum, maximum = float(np.min(values)), float(np.max(values))
    span = maximum - minimum
    margin = fraction * (span if span > 0.0 else max(abs(maximum), 1.0))
    return minimum - margin, maximum + margin


def plot_large_fem_setup():
    fig, axis = plt.subplots(figsize=(9.0, 2.5), constrained_layout=True)
    segments = []
    for cell in SYSTEM.cells:
        polygon = SYSTEM.points[np.append(cell, cell[0])]
        segments.extend(zip(polygon[:-1], polygon[1:]))
    axis.add_collection(LineCollection(segments, colors="#7A828C", linewidths=0.55))
    axis.plot([0, 0], [0, 0.1], color=FRAME_COLOR, linewidth=4.0)
    axis.scatter(
        SYSTEM.contact_coordinates[:, 0], SYSTEM.contact_coordinates[:, 1],
        s=70, color=[STICK_COLOR, SLIP_COLOR], edgecolor="white", linewidth=0.9, zorder=4,
    )
    axis.text(0.6875, -0.015, "A", ha="center", va="top", weight="bold")
    axis.text(0.9375, -0.015, "B", ha="center", va="top", weight="bold")
    axis.annotate(
        r"$F(t,\xi)$", xy=(1.0, 0.05), xytext=(0.91, 0.18),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT_COLOR, "lw": 1.8},
        color=ACCENT_COLOR, ha="center",
    )
    axis.scatter([1.0], [0.05], marker="D", s=45, color=MLP_COLOR, zorder=5)
    axis.text(0.985, 0.035, "obs.", ha="right", va="top", color=MLP_COLOR)
    axis.text(0.012, 0.087, "fixed", ha="left", va="top")
    axis.set(xlim=(-0.02, 1.04), ylim=(-0.035, 0.205), xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "large_fem_setup.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_optimization_history(history):
    iterations = np.arange(MAX_ITERATIONS + 1)
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.axhline(history.objective[0], color=FIXED_COLOR, lw=1.4, ls="--", label="Fixed")
    axis.plot(iterations, history.objective, color=MLP_COLOR, lw=2.0, label="MLP")
    milestones = np.asarray(MILESTONE_ITERATIONS)
    axis.scatter(milestones, history.objective[milestones], color=ACCENT_COLOR, s=27, zorder=3)
    axis.set(xlim=(0, MAX_ITERATIONS), xlabel="Iteration", ylabel="Training objective")
    axis.set_xticks(MILESTONE_ITERATIONS)
    axis.legend(loc="best")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "optimization_history_500.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_held_out_improvement(relative_improvement):
    percent = 100.0 * relative_improvement
    colors = np.where(percent >= 0.0, MLP_COLOR, ACCENT_COLOR)
    fig, axis = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    axis.axhline(0.0, color=FRAME_COLOR, lw=1.0)
    axis.vlines(H4_TEST_SEEDS, 0.0, percent, color=colors, lw=1.2)
    axis.scatter(H4_TEST_SEEDS, percent, color=colors, s=19, zorder=3)
    axis.set(xlabel="Held-out seed", ylabel="Improvement (%)")
    axis.set_xticks(H4_TEST_SEEDS[::8])
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "held_out_improvement.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_representative_response(replay):
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(periods, replay["fixed_observation"], color=FIXED_COLOR, lw=1.5, label="Fixed")
    axes[0].plot(periods, replay["displacement"][-1], color=MLP_COLOR, lw=1.7, label="Iteration 500")
    axes[0].set_ylabel("Displacement")
    axes[0].legend(loc="best")
    axes[1].axhline(BASE_Q[1], color=FIXED_COLOR, lw=1.3, ls="--", label="Fixed")
    axes[1].plot(periods, replay["preload"][-1], color=MLP_COLOR, lw=1.7, label="Iteration 500")
    axes[1].set(xlabel="Time (periods)", ylabel="Preload")
    axes[1].legend(loc="best")
    for axis in axes:
        _style_axis(axis)
    path = OUTPUT_DIRECTORY / "representative_response.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _quantized_frame(fig, colors=96):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3]).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def _save_gif(frames, path, duration):
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=duration,
        loop=0, optimize=False, disposal=2,
    )
    return path


def create_all_iterations_gif(history, replay):
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    fig, axes = plt.subplots(
        3, 1, figsize=(6.8, 6.8), dpi=76,
        gridspec_kw={"height_ratios": [0.85, 1.0, 1.25]}, constrained_layout=True,
    )
    objective_line, = axes[0].plot([], [], color=MLP_COLOR, lw=1.8)
    objective_marker, = axes[0].plot([], [], marker="o", color=ACCENT_COLOR, ms=4.5)
    axes[0].axhline(history.objective[0], color=FIXED_COLOR, lw=1.1, ls="--")
    axes[0].set(xlim=(0, MAX_ITERATIONS), ylim=_expanded_limits(history.objective), ylabel="Train objective")
    axes[1].axhline(BASE_Q[1], color=FIXED_COLOR, lw=1.2, ls="--", label="Fixed")
    preload_line, = axes[1].plot([], [], color=MLP_COLOR, lw=1.6, label="Current")
    axes[1].set(
        xlim=(periods[0], periods[-1]),
        ylim=_expanded_limits(np.concatenate(([BASE_Q[1]], replay["preload"].ravel()))),
        ylabel="Preload",
    )
    axes[1].legend(loc="upper right", ncol=2)
    axes[2].plot(periods, replay["fixed_observation"], color=FIXED_COLOR, lw=1.3, label="Fixed")
    displacement_line, = axes[2].plot([], [], color=MLP_COLOR, lw=1.6, label="Current")
    all_displacement = np.concatenate((replay["fixed_observation"], replay["displacement"].ravel()))
    axes[2].set(
        xlim=(periods[0], periods[-1]), ylim=_expanded_limits(all_displacement),
        xlabel="Time (periods)", ylabel="Displacement",
    )
    axes[2].legend(loc="upper right", ncol=2)
    for axis in axes:
        _style_axis(axis)
    annotation = fig.suptitle("", fontsize=10)
    frames = []
    iterations = np.arange(MAX_ITERATIONS + 1)
    for iteration in iterations:
        objective_line.set_data(iterations[: iteration + 1], history.objective[: iteration + 1])
        objective_marker.set_data([iteration], [history.objective[iteration]])
        preload_line.set_data(periods, replay["preload"][iteration])
        displacement_line.set_data(periods, replay["displacement"][iteration])
        annotation.set_text(
            f"Iteration {iteration:03d}   Train objective = {history.objective[iteration]:.7f}   "
            f"Held-out seed = {replay['representative_seed']}"
        )
        frames.append(_quantized_frame(fig))
        if iteration % 100 == 0 or iteration == MAX_ITERATIONS:
            print(f"optimizer_gif_frame={iteration:03d}/{MAX_ITERATIONS}", flush=True)
    plt.close(fig)
    return _save_gif(frames, OUTPUT_DIRECTORY / "optimization_all_iterations.gif", duration=45)


def create_deformation_gif(replay):
    fixed, trained = replay["fixed"]["displacement"], replay["final"]["displacement"]
    fixed_slip, trained_slip = replay["fixed"]["slip"], replay["final"]["slip"]
    magnitude = np.concatenate((np.linalg.norm(fixed, axis=2).ravel(), np.linalg.norm(trained, axis=2).ravel()))
    magnitude_max = float(np.max(magnitude))
    deformation_scale = 0.12 / magnitude_max
    norm = Normalize(0.0, magnitude_max)
    frame_indices = np.linspace(0, NUM_STEPS - 1, NUM_PHYSICAL_FRAMES, dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), dpi=95, constrained_layout=True)
    collections, contact_artists = [], []
    for axis, label in zip(axes, ("Fixed", "Iteration 500 MLP")):
        collection = PolyCollection(
            SYSTEM.points[SYSTEM.cells], array=np.zeros(len(SYSTEM.cells)), cmap="viridis", norm=norm,
            edgecolors="#4C5560", linewidths=0.35,
        )
        axis.add_collection(collection)
        artists = [axis.scatter([], [], s=32, edgecolor="white", linewidth=0.6, zorder=5) for _ in range(2)]
        axis.text(0.02, 0.94, label, transform=axis.transAxes, ha="left", va="top", weight="bold")
        axis.set(xlim=(-0.03, 1.04), ylim=(-0.18, 0.28), xlabel="x")
        axis.set_aspect("equal")
        _style_axis(axis)
        collections.append(collection)
        contact_artists.append(artists)
    axes[0].set_ylabel("y")
    colorbar = fig.colorbar(collections[-1], ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("Displacement magnitude")
    annotation = fig.text(0.5, 0.995, "", ha="center", va="top", fontsize=10)
    frames = []
    for step in frame_indices:
        for panel, (field, states) in enumerate(((fixed, fixed_slip), (trained, trained_slip))):
            deformed = SYSTEM.points + deformation_scale * field[step]
            collections[panel].set_verts(deformed[SYSTEM.cells])
            collections[panel].set_array(np.mean(np.linalg.norm(field[step], axis=1)[SYSTEM.cells], axis=1))
            for contact, artist in enumerate(contact_artists[panel]):
                artist.set_offsets(deformed[SYSTEM.contact_nodes[contact]][None, :])
                artist.set_color(SLIP_COLOR if states[step, contact] else STICK_COLOR)
        periods = float(SYSTEM.times[step] * SYSTEM.omega_1 / (2.0 * np.pi))
        annotation.set_text(
            f"Held-out seed {replay['representative_seed']}   Time = {periods:.2f} periods   "
            "blue: STICK, orange: SLIP"
        )
        frames.append(_quantized_frame(fig, colors=128))
    plt.close(fig)
    path = _save_gif(frames, OUTPUT_DIRECTORY / "fixed_vs_final_deformation.gif", duration=85)
    return path, deformation_scale, magnitude_max


def _validate_media(paths, expected_gif_frames):
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    for path, expected in expected_gif_frames.items():
        with Image.open(path) as image:
            if image.n_frames != expected:
                raise RuntimeError(f"{path} has {image.n_frames} frames, expected {expected}")


def _transition_summary(result):
    return {
        "stick_to_slip": np.asarray(result.stick_to_slip, dtype=np.int64).tolist(),
        "slip_to_stick": np.asarray(result.slip_to_stick, dtype=np.int64).tolist(),
    }


def main() -> int:
    torch.set_default_dtype(torch.float64)
    _configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    controller, physics = create_tesseracts()
    preflight_result = preflight(controller, physics)
    print("## Preflight")
    print(f"mesh: {NUM_ELEMENTS_X}x{NUM_ELEMENTS_Y} QUAD4")
    print(f"elements: {len(SYSTEM.cells)}")
    print(f"nodes: {len(SYSTEM.points)}")
    print(f"total_dofs: {SYSTEM.num_total_dofs}")
    print(f"free_dofs: {SYSTEM.num_free_dofs}")
    print(f"omega_1: {SYSTEM.omega_1:.16g}")
    print(f"fixed_forward_seconds: {preflight_result['fixed_seconds']:.9g}")
    print(f"backward_seconds: {preflight_result['backward_seconds']:.9g}")
    print(f"backward_gradient_norm: {preflight_result['backward_gradient_norm']:.16g}")
    print(f"peak_rss_gib: {preflight_result['peak_rss_gib']:.9g}")
    print(f"preflight_gate: {preflight_result['gate']}")
    if not preflight_result["gate"]:
        return 1

    fixed_train = preflight_result["fixed"]
    fixed_train_objective = float(np.mean(fixed_train.losses))
    history, training_seconds, history_path = train(controller, physics, fixed_train_objective)
    final_theta = history.theta[-1]
    trained_train = evaluate(controller, physics, final_theta, H4_TRAINING_SEEDS)
    evaluation_start = time.perf_counter()
    fixed_test = evaluate(controller, physics, final_theta, H4_TEST_SEEDS, fixed=True)
    trained_test = evaluate(controller, physics, final_theta, H4_TEST_SEEDS)
    evaluation_seconds = time.perf_counter() - evaluation_start

    trained_train_objective = float(np.mean(trained_train.losses))
    fixed_test_objective = float(np.mean(fixed_test.losses))
    trained_test_objective = float(np.mean(trained_test.losses))
    test_improvement = (fixed_test_objective - trained_test_objective) / fixed_test_objective
    win_count = int(np.count_nonzero(trained_test.losses < fixed_test.losses))
    replay = replay_representative(controller, history, fixed_test, trained_test)
    figure_paths = [
        plot_large_fem_setup(), plot_optimization_history(history),
        plot_held_out_improvement(replay["relative_improvement"]), plot_representative_response(replay),
    ]
    gif_start = time.perf_counter()
    optimizer_gif = create_all_iterations_gif(history, replay)
    deformation_gif, deformation_scale, displacement_color_max = create_deformation_gif(replay)
    gif_seconds = time.perf_counter() - gif_start
    media_paths = figure_paths + [optimizer_gif, deformation_gif]
    _validate_media(media_paths, {optimizer_gif: 501, deformation_gif: NUM_PHYSICAL_FRAMES})

    objective_min_iteration = int(np.argmin(history.objective))
    objective_min = float(history.objective[objective_min_iteration])
    improvement_100 = (history.objective[0] - history.objective[100]) / history.objective[0]
    improvement_500 = (history.objective[0] - history.objective[500]) / history.objective[0]
    representative_fixed_transitions = _transition_summary(replay["fixed_result"])
    representative_final_transitions = _transition_summary(replay["final_result"])
    passed = bool(
        np.all(np.isfinite(history.theta)) and np.all(np.isfinite(history.objective))
        and np.all(np.isfinite(history.gradient)) and np.all(np.isfinite(history.n_min))
        and np.all(np.isfinite(history.n_max)) and trained_train_objective < fixed_train_objective
        and trained_test_objective < fixed_test_objective and _switching_gate(trained_test)
    )
    summary = {
        "mesh": "32x4 QUAD4", "elements": 128, "nodes": 165,
        "total_dofs": 330, "free_dofs": 320,
        "contact_coordinates": SYSTEM.contact_coordinates.tolist(), "omega_1": float(SYSTEM.omega_1),
        "milestone_objectives": {str(i): float(history.objective[i]) for i in MILESTONE_ITERATIONS},
        "minimum_train_objective": objective_min, "minimum_train_iteration": objective_min_iteration,
        "improvement_0_to_100": improvement_100, "improvement_0_to_500": improvement_500,
        "fixed_test_objective": fixed_test_objective,
        "iteration_500_test_objective": trained_test_objective,
        "test_relative_improvement": test_improvement, "test_win_count": win_count,
        "test_fixed_losses": fixed_test.losses.tolist(),
        "test_iteration_500_losses": trained_test.losses.tolist(),
        "test_seed_relative_improvement": replay["relative_improvement"].tolist(),
        "gradient_norm_start": float(history.gradient[0]), "gradient_norm_end": float(history.gradient[-1]),
        "N_min_final": float(history.n_min[-1]), "N_max_final": float(history.n_max[-1]),
        "fixed_train_stick_to_slip": np.sum(fixed_train.stick_to_slip, axis=0).tolist(),
        "fixed_train_slip_to_stick": np.sum(fixed_train.slip_to_stick, axis=0).tolist(),
        "trained_test_stick_to_slip": np.sum(trained_test.stick_to_slip, axis=0).tolist(),
        "trained_test_slip_to_stick": np.sum(trained_test.slip_to_stick, axis=0).tolist(),
        "representative_seed": replay["representative_seed"],
        "representative_median_improvement": replay["median_improvement"],
        "representative_fixed_transitions": representative_fixed_transitions,
        "representative_iteration_500_transitions": representative_final_transitions,
        "preflight_backward_seconds": preflight_result["backward_seconds"],
        "training_seconds": training_seconds, "held_out_evaluation_seconds": evaluation_seconds,
        "representative_replay_seconds": replay["seconds"], "gif_generation_seconds": gif_seconds,
        "deformation_scale": deformation_scale, "displacement_color_max": displacement_color_max,
        "peak_rss_gib": _peak_rss_gib(), "pass": passed,
    }
    summary_path = OUTPUT_DIRECTORY / "showcase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("## 32x4 FEM")
    print(f"contact_coordinates: {SYSTEM.contact_coordinates.tolist()}")
    print(
        "fixed_train_transitions: "
        f"stick_to_slip={summary['fixed_train_stick_to_slip']} "
        f"slip_to_stick={summary['fixed_train_slip_to_stick']}"
    )
    print("## 500-step optimization")
    for iteration in MILESTONE_ITERATIONS:
        print(f"J_iter{iteration}: {history.objective[iteration]:.16g}")
    print(f"J_minimum: {objective_min:.16g} at iteration {objective_min_iteration}")
    print(f"improvement_0_to_100: {improvement_100:.16g}")
    print(f"improvement_0_to_500: {improvement_500:.16g}")
    print(f"gradient_norm_start: {history.gradient[0]:.16g}")
    print(f"gradient_norm_end: {history.gradient[-1]:.16g}")
    print(f"N_final: [{history.n_min[-1]:.16g}, {history.n_max[-1]:.16g}]")
    print("## Held-out evaluation")
    print(f"J_fixed_test: {fixed_test_objective:.16g}")
    print(f"J_iter500_test: {trained_test_objective:.16g}")
    print(f"test_relative_improvement: {test_improvement:.16g}")
    print(f"test_win_count: {win_count}/64")
    print(f"representative_seed: {replay['representative_seed']}")
    print(f"representative_fixed_transitions: {representative_fixed_transitions}")
    print(f"representative_iteration_500_transitions: {representative_final_transitions}")
    print("## Runtime")
    print(f"preflight_backward_seconds: {preflight_result['backward_seconds']:.9g}")
    print(f"training_seconds: {training_seconds:.9g}")
    print(f"held_out_evaluation_seconds: {evaluation_seconds:.9g}")
    print(f"representative_replay_seconds: {replay['seconds']:.9g}")
    print(f"gif_generation_seconds: {gif_seconds:.9g}")
    print(f"peak_rss_gib: {_peak_rss_gib():.9g}")
    print("## Visualization")
    for path in media_paths:
        print(path.resolve())
    print(history_path.resolve())
    print(replay["path"].resolve())
    print(summary_path.resolve())
    print("## PASS" if passed else "## FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
