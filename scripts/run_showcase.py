"""Run the locked 32x4, 100-iteration final Hackathon showcase."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import meshio
import numpy as np
from PIL import Image
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import (
    build_controller,
    flatten_controller_parameters,
)
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
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
PARAVIEW_DIRECTORY = OUTPUT_DIRECTORY / "paraview"
RENDER_DIRECTORY = OUTPUT_DIRECTORY / "rendered_frames"
BASE_Q = np.array([0.2, 0.04], dtype=np.float64)
LEARNING_RATE = 0.01
MAX_ITERATIONS = 100
CHECKPOINT_ITERATIONS = (0, 20, 40, 60, 80, 100)
NUM_VISUALIZATION_FRAMES = 80

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
        {
            "theta": theta,
            "descriptors": forcing_descriptor_batch(seeds),
        },
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
    collected = {
        "losses": [],
        "coefficients": [],
        "stick_to_slip": [],
        "slip_to_stick": [],
        "displacement_min": [],
        "displacement_max": [],
        "velocity_min": [],
        "velocity_max": [],
    }
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
            "stick_to_slip",
            "slip_to_stick",
            "displacement_min",
            "displacement_max",
            "velocity_min",
            "velocity_max",
        ):
            collected[name].append(np.asarray(response[name]))
    return Evaluation(
        losses=np.concatenate(collected["losses"]),
        coefficients=np.concatenate(collected["coefficients"]),
        stick_to_slip=np.concatenate(collected["stick_to_slip"]),
        slip_to_stick=np.concatenate(collected["slip_to_stick"]),
        displacement_min=np.concatenate(collected["displacement_min"]),
        displacement_max=np.concatenate(collected["displacement_max"]),
        velocity_min=np.concatenate(collected["velocity_min"]),
        velocity_max=np.concatenate(collected["velocity_max"]),
    )


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


def _save_checkpoint(theta, iteration):
    np.save(
        CHECKPOINT_DIRECTORY / f"theta_iter_{iteration:03d}.npy",
        np.asarray(theta, dtype=np.float64),
        allow_pickle=False,
    )


def train(controller, physics, initial_objective):
    theta = torch.nn.Parameter(_initial_theta())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    history = [
        {
            "iteration": 0,
            "objective": initial_objective,
            "gradient_norm": np.nan,
            "N_min": BASE_Q[1],
            "N_max": BASE_Q[1],
        }
    ]
    _save_checkpoint(theta.detach().cpu().numpy(), 0)

    start = time.perf_counter()
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = full_training_loss(controller, physics, theta)
        loss.backward()
        gradient_norm = float(torch.linalg.vector_norm(theta.grad))
        optimizer.step()

        hard = evaluate(
            controller,
            physics,
            theta.detach().cpu().numpy(),
            H4_TRAINING_SEEDS,
        )
        objective = float(np.mean(hard.losses))
        preload_min, preload_max = _control_range(hard.coefficients)
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "gradient_norm": gradient_norm,
                "N_min": preload_min,
                "N_max": preload_max,
            }
        )
        if iteration in CHECKPOINT_ITERATIONS:
            _save_checkpoint(theta.detach().cpu().numpy(), iteration)
        print(
            f"iteration={iteration:03d} objective={objective:.16g} "
            f"gradient_norm={gradient_norm:.9g} "
            f"N=[{preload_min:.7f},{preload_max:.7f}]",
            flush=True,
        )
    return (
        theta.detach().cpu().numpy(),
        history,
        time.perf_counter() - start,
    )


def write_history(history):
    path = OUTPUT_DIRECTORY / "training_history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return path


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


def _write_pvd(path, frame_paths, times):
    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for frame_path, time_value in zip(frame_paths, times):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time_value:.16g}",
            group="",
            part="0",
            file=frame_path.name,
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_series(label, trajectory, frame_indices):
    directory = PARAVIEW_DIRECTORY / label
    directory.mkdir(parents=True, exist_ok=True)
    points = np.column_stack((SYSTEM.points, np.zeros(len(SYSTEM.points))))
    times = np.asarray(SYSTEM.times)[frame_indices]
    frame_paths = []
    for frame, step in enumerate(frame_indices):
        displacement = np.column_stack(
            (trajectory["displacement"][step], np.zeros(len(SYSTEM.points)))
        )
        velocity = np.column_stack(
            (trajectory["velocity"][step], np.zeros(len(SYSTEM.points)))
        )
        contact_state = np.full(len(SYSTEM.points), -1, dtype=np.int64)
        contact_state[SYSTEM.contact_nodes] = trajectory["slip"][step]
        is_contact = np.zeros(len(SYSTEM.points), dtype=np.int64)
        is_contact[SYSTEM.contact_nodes] = 1
        mesh = meshio.Mesh(
            points=points,
            cells=[("quad", SYSTEM.cells)],
            point_data={
                "displacement": displacement,
                "displacement_magnitude": np.linalg.norm(
                    displacement, axis=1
                ),
                "velocity_magnitude": np.linalg.norm(velocity, axis=1),
                "contact_state": contact_state,
                "is_contact": is_contact,
            },
        )
        frame_path = directory / f"{label}_{frame:04d}.vtu"
        meshio.write(frame_path, mesh, binary=True)
        frame_paths.append(frame_path)
    pvd_path = directory / f"{label}.pvd"
    _write_pvd(pvd_path, frame_paths, times)
    return pvd_path, frame_paths, times


def validate_vtk_series(pvd_path, expected_frames):
    datasets = ET.parse(pvd_path).getroot().findall("./Collection/DataSet")
    if len(datasets) != expected_frames:
        raise RuntimeError(f"{pvd_path} has {len(datasets)} frames")
    required = {
        "displacement",
        "displacement_magnitude",
        "velocity_magnitude",
        "contact_state",
        "is_contact",
    }
    for dataset in datasets:
        mesh = meshio.read(pvd_path.parent / dataset.attrib["file"])
        if required != set(mesh.point_data):
            raise RuntimeError("VTU point fields do not match the showcase contract")
        if len(mesh.points) != 165 or len(mesh.cells_dict["quad"]) != 128:
            raise RuntimeError("VTU mesh does not match the 32x4 showcase")
        for values in mesh.point_data.values():
            if not np.all(np.isfinite(values)):
                raise RuntimeError("VTU contains non-finite point data")


def _observation_history(free_displacement):
    return np.asarray(free_displacement) @ np.asarray(SYSTEM.observation)


def _representative_coefficients(controller, theta, seed):
    index = int(np.flatnonzero(H4_TEST_SEEDS == seed)[0])
    batch_start = (index // 8) * 8
    batch = H4_TEST_SEEDS[batch_start : batch_start + 8]
    coefficients = _controller_coefficients(controller, theta, batch)
    return coefficients[index - batch_start]


def prepare_visualization_data(controller, trained_theta, history, fixed_test, trained_test):
    relative_improvement = (fixed_test.losses - trained_test.losses) / fixed_test.losses
    median_improvement = float(np.median(relative_improvement))
    representative_index = int(
        np.argmin(np.abs(relative_improvement - median_improvement))
    )
    representative_seed = int(H4_TEST_SEEDS[representative_index])
    trained_coefficients = trained_test.coefficients[representative_index]

    trajectory_start = time.perf_counter()
    fixed_result = evaluate_full_trajectory(
        BASE_Q, np.zeros(5, dtype=np.float64), representative_seed
    )
    trained_result = evaluate_full_trajectory(
        BASE_Q, trained_coefficients, representative_seed
    )
    fixed_trajectory = _trajectory_arrays(fixed_result)
    trained_trajectory = _trajectory_arrays(trained_result)

    checkpoint_trajectories = []
    for iteration in CHECKPOINT_ITERATIONS:
        theta = np.load(
            CHECKPOINT_DIRECTORY / f"theta_iter_{iteration:03d}.npy",
            allow_pickle=False,
        )
        coefficients = _representative_coefficients(
            controller, theta, representative_seed
        )
        result = _trajectory_arrays(
            evaluate_full_trajectory(BASE_Q, coefficients, representative_seed)
        )
        checkpoint_trajectories.append(
            {
                "iteration": iteration,
                "objective": history[iteration]["objective"],
                "coefficients": coefficients,
                "displacement": _observation_history(
                    result["free_displacement"]
                ),
                "preload": np.asarray(
                    preload_history(BASE_Q[1], coefficients[None, :])
                )[0],
            }
        )
    trajectory_seconds = time.perf_counter() - trajectory_start

    full_path = OUTPUT_DIRECTORY / "representative_full_trajectories.npz"
    np.savez(
        full_path,
        times=np.asarray(SYSTEM.times),
        fixed_displacement=fixed_trajectory["displacement"],
        fixed_velocity=fixed_trajectory["velocity"],
        fixed_slip=fixed_trajectory["slip"],
        trained_displacement=trained_trajectory["displacement"],
        trained_velocity=trained_trajectory["velocity"],
        trained_slip=trained_trajectory["slip"],
    )
    return {
        "representative_seed": representative_seed,
        "representative_index": representative_index,
        "median_improvement": median_improvement,
        "relative_improvement": relative_improvement,
        "trained_coefficients": trained_coefficients,
        "fixed": fixed_trajectory,
        "trained": trained_trajectory,
        "fixed_result": fixed_result,
        "trained_result": trained_result,
        "checkpoint_trajectories": checkpoint_trajectories,
        "trajectory_seconds": trajectory_seconds,
        "full_path": full_path,
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
    axis.tick_params(top=False, right=False, direction="in", width=1.0)


def plot_large_fem_setup():
    fig, axis = plt.subplots(figsize=(9.0, 2.5), constrained_layout=True)
    segments = []
    for cell in SYSTEM.cells:
        polygon = SYSTEM.points[np.append(cell, cell[0])]
        segments.extend(zip(polygon[:-1], polygon[1:]))
    axis.add_collection(
        LineCollection(segments, colors="#7A828C", linewidths=0.55)
    )
    axis.plot([0, 0], [0, 0.1], color=FRAME_COLOR, linewidth=4.0)
    axis.scatter(
        SYSTEM.contact_coordinates[:, 0],
        SYSTEM.contact_coordinates[:, 1],
        s=70,
        color=[STICK_COLOR, SLIP_COLOR],
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    axis.text(0.6875, -0.015, "A", ha="center", va="top", weight="bold")
    axis.text(0.9375, -0.015, "B", ha="center", va="top", weight="bold")
    axis.annotate(
        r"$F(t,\xi)$",
        xy=(1.0, 0.05),
        xytext=(0.91, 0.18),
        arrowprops={"arrowstyle": "-|>", "color": ACCENT_COLOR, "lw": 1.8},
        color=ACCENT_COLOR,
        ha="center",
    )
    axis.scatter(
        [1.0], [0.05], marker="D", s=45, color=MLP_COLOR, zorder=5
    )
    axis.text(0.985, 0.035, "obs.", ha="right", va="top", color=MLP_COLOR)
    axis.text(0.012, 0.087, "fixed", ha="left", va="top")
    axis.set_xlim(-0.02, 1.04)
    axis.set_ylim(-0.035, 0.205)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "large_fem_setup.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_optimization_history(history):
    iterations = np.array([row["iteration"] for row in history])
    objectives = np.array([row["objective"] for row in history])
    fig, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    axis.axhline(
        objectives[0], color=FIXED_COLOR, linewidth=1.5, linestyle="--", label="Fixed"
    )
    axis.plot(
        iterations,
        objectives,
        color=MLP_COLOR,
        linewidth=2.2,
        label="MLP",
    )
    axis.scatter(
        [0, 100], [objectives[0], objectives[-1]],
        color=[FIXED_COLOR, MLP_COLOR], s=35, zorder=3
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    axis.legend(loc="best")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "optimization_history.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_held_out_improvement(relative_improvement):
    percent = 100.0 * relative_improvement
    colors = np.where(percent >= 0.0, MLP_COLOR, ACCENT_COLOR)
    fig, axis = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    axis.axhline(0.0, color=FRAME_COLOR, linewidth=1.0)
    axis.vlines(H4_TEST_SEEDS, 0.0, percent, color=colors, linewidth=1.2)
    axis.scatter(H4_TEST_SEEDS, percent, color=colors, s=19, zorder=3)
    axis.set_xlabel("Held-out seed")
    axis.set_ylabel("Improvement (%)")
    axis.set_xticks(H4_TEST_SEEDS[::8])
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "held_out_improvement.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_representative_response(visualization):
    times = np.asarray(SYSTEM.times)
    periods = times * SYSTEM.omega_1 / (2.0 * np.pi)
    fixed_displacement = _observation_history(
        visualization["fixed"]["free_displacement"]
    )
    trained_displacement = _observation_history(
        visualization["trained"]["free_displacement"]
    )
    trained_preload = np.asarray(
        preload_history(
            BASE_Q[1], visualization["trained_coefficients"][None, :]
        )
    )[0]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.8, 6.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(periods, fixed_displacement, color=FIXED_COLOR, lw=1.5, label="Fixed")
    axes[0].plot(periods, trained_displacement, color=MLP_COLOR, lw=1.7, label="MLP")
    axes[0].set_ylabel("Displacement")
    axes[0].legend(loc="best")
    axes[1].axhline(BASE_Q[1], color=FIXED_COLOR, lw=1.3, ls="--", label="Fixed")
    axes[1].plot(periods, trained_preload, color=MLP_COLOR, lw=1.7, label="MLP")
    axes[1].set_xlabel("Time (periods)")
    axes[1].set_ylabel("Preload")
    axes[1].legend(loc="best")
    for axis in axes:
        _style_axis(axis)
    path = OUTPUT_DIRECTORY / "representative_response.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def create_progress_gif(checkpoint_trajectories):
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    all_displacements = np.concatenate(
        [entry["displacement"] for entry in checkpoint_trajectories]
    )
    all_preloads = np.concatenate(
        [entry["preload"] for entry in checkpoint_trajectories]
    )
    displacement_margin = 0.05 * np.ptp(all_displacements)
    preload_margin = 0.05 * np.ptp(all_preloads)
    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 5.8), sharex=True, constrained_layout=True
    )
    displacement_line, = axes[0].plot([], [], color=MLP_COLOR, lw=1.8)
    preload_line, = axes[1].plot([], [], color=MLP_COLOR, lw=1.8)
    axes[1].axhline(BASE_Q[1], color=FIXED_COLOR, lw=1.2, ls="--")
    annotation = axes[0].text(
        0.02, 0.95, "", transform=axes[0].transAxes, ha="left", va="top"
    )
    axes[0].set_xlim(periods[0], periods[-1])
    axes[0].set_ylim(
        np.min(all_displacements) - displacement_margin,
        np.max(all_displacements) + displacement_margin,
    )
    axes[1].set_ylim(
        np.min(all_preloads) - preload_margin,
        np.max(all_preloads) + preload_margin,
    )
    axes[0].set_ylabel("Displacement")
    axes[1].set_ylabel("Preload")
    axes[1].set_xlabel("Time (periods)")
    for axis in axes:
        _style_axis(axis)

    def update(frame):
        entry = checkpoint_trajectories[frame]
        displacement_line.set_data(periods, entry["displacement"])
        preload_line.set_data(periods, entry["preload"])
        annotation.set_text(
            f"Iteration {entry['iteration']}\nJ = {entry['objective']:.6g}"
        )
        return displacement_line, preload_line, annotation

    animation = FuncAnimation(
        fig, update, frames=len(checkpoint_trajectories), blit=False
    )
    path = OUTPUT_DIRECTORY / "optimization_progress.gif"
    animation.save(path, writer=PillowWriter(fps=1), dpi=120)
    plt.close(fig)
    return path


def write_render_metadata(fixed_pvd, trained_pvd, sampled_times, frame_indices, visualization):
    fixed_magnitude = np.linalg.norm(visualization["fixed"]["displacement"], axis=2)
    trained_magnitude = np.linalg.norm(
        visualization["trained"]["displacement"], axis=2
    )
    displacement_max = float(max(np.max(fixed_magnitude), np.max(trained_magnitude)))
    deformation_scale = 0.12 / displacement_max
    points_3d = np.column_stack((SYSTEM.points, np.zeros(len(SYSTEM.points))))

    def series_metadata(label, pvd_path, trajectory):
        displacement_3d = np.concatenate(
            (
                trajectory["displacement"][frame_indices],
                np.zeros((len(frame_indices), len(SYSTEM.points), 1)),
            ),
            axis=2,
        )
        centers = (
            points_3d[SYSTEM.contact_nodes][None, :, :]
            + deformation_scale * displacement_3d[:, SYSTEM.contact_nodes, :]
        )
        return {
            "label": label,
            "pvd": str(pvd_path.resolve()),
            "contact_centers": centers.tolist(),
            "contact_state": trajectory["slip"][frame_indices].tolist(),
        }

    metadata = {
        "times": sampled_times.tolist(),
        "displacement_max": displacement_max,
        "deformation_scale": deformation_scale,
        "camera_parallel_scale": 0.30,
        "fixed": series_metadata("Fixed", fixed_pvd, visualization["fixed"]),
        "trained": series_metadata(
            "MLP after 100 iterations", trained_pvd, visualization["trained"]
        ),
    }
    path = PARAVIEW_DIRECTORY / "render_metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path


def _find_pvpython():
    candidates = [
        Path("/Applications/ParaView-6.1.0.app/Contents/bin/pvpython"),
        Path("/Applications/ParaView.app/Contents/bin/pvpython"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    result = subprocess.run(
        ["which", "pvpython"], capture_output=True, text=True, check=False
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def _combine_deformation_gif():
    fixed_paths = sorted((RENDER_DIRECTORY / "fixed").glob("fixed_*.png"))
    trained_paths = sorted((RENDER_DIRECTORY / "trained").glob("trained_*.png"))
    if len(fixed_paths) != NUM_VISUALIZATION_FRAMES or len(trained_paths) != NUM_VISUALIZATION_FRAMES:
        raise RuntimeError("ParaView did not render the expected 80+80 frames")
    frames = []
    for fixed_path, trained_path in zip(fixed_paths, trained_paths):
        with Image.open(fixed_path) as fixed_image, Image.open(trained_path) as trained_image:
            combined = Image.new("RGB", (1800, 360), "white")
            combined.paste(fixed_image.convert("RGB"), (0, 0))
            combined.paste(trained_image.convert("RGB"), (900, 0))
            combined = combined.resize((1400, 280), Image.Resampling.LANCZOS)
            frames.append(
                combined.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            )
    path = OUTPUT_DIRECTORY / "fixed_vs_mlp_deformation.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


def render_deformation_gif(metadata_path):
    pvpython = _find_pvpython()
    if pvpython is None:
        raise RuntimeError("pvpython is not available")
    render_script = ROOT / "scripts/render_showcase_paraview.py"
    for series in ("fixed", "trained"):
        output = RENDER_DIRECTORY / series
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(pvpython),
                str(render_script),
                "--metadata",
                str(metadata_path),
                "--series",
                series,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
    return _combine_deformation_gif()


def _write_summary(path, summary):
    path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    torch.set_default_dtype(torch.float64)
    _configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PARAVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RENDER_DIRECTORY.mkdir(parents=True, exist_ok=True)

    controller, physics = create_tesseracts()
    preflight_result = preflight(controller, physics)
    print("## Preflight")
    print(f"mesh: {NUM_ELEMENTS_X}x{NUM_ELEMENTS_Y} QUAD4")
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
    trained_theta, history, training_seconds = train(
        controller, physics, fixed_train_objective
    )
    history_path = write_history(history)
    trained_train = evaluate(
        controller, physics, trained_theta, H4_TRAINING_SEEDS
    )

    evaluation_start = time.perf_counter()
    fixed_test = evaluate(
        controller, physics, trained_theta, H4_TEST_SEEDS, fixed=True
    )
    trained_test = evaluate(
        controller, physics, trained_theta, H4_TEST_SEEDS
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    trained_train_objective = float(np.mean(trained_train.losses))
    fixed_test_objective = float(np.mean(fixed_test.losses))
    trained_test_objective = float(np.mean(trained_test.losses))
    train_improvement = (
        fixed_train_objective - trained_train_objective
    ) / fixed_train_objective
    test_improvement = (
        fixed_test_objective - trained_test_objective
    ) / fixed_test_objective
    win_count = int(np.count_nonzero(trained_test.losses < fixed_test.losses))

    visualization = prepare_visualization_data(
        controller, trained_theta, history, fixed_test, trained_test
    )
    frame_indices = np.linspace(
        0, NUM_STEPS - 1, NUM_VISUALIZATION_FRAMES, dtype=np.int64
    )
    export_start = time.perf_counter()
    fixed_pvd, _, sampled_times = export_series(
        "fixed", visualization["fixed"], frame_indices
    )
    trained_pvd, _, _ = export_series(
        "trained", visualization["trained"], frame_indices
    )
    validate_vtk_series(fixed_pvd, NUM_VISUALIZATION_FRAMES)
    validate_vtk_series(trained_pvd, NUM_VISUALIZATION_FRAMES)
    export_seconds = time.perf_counter() - export_start

    metadata_path = write_render_metadata(
        fixed_pvd, trained_pvd, sampled_times, frame_indices, visualization
    )
    figure_paths = [
        plot_large_fem_setup(),
        plot_optimization_history(history),
        plot_held_out_improvement(visualization["relative_improvement"]),
        plot_representative_response(visualization),
    ]
    progress_gif = create_progress_gif(
        visualization["checkpoint_trajectories"]
    )
    deformation_gif = render_deformation_gif(metadata_path)

    representative_fixed_transitions = {
        "stick_to_slip": np.asarray(
            visualization["fixed_result"].stick_to_slip
        ).tolist(),
        "slip_to_stick": np.asarray(
            visualization["fixed_result"].slip_to_stick
        ).tolist(),
    }
    representative_trained_transitions = {
        "stick_to_slip": np.asarray(
            visualization["trained_result"].stick_to_slip
        ).tolist(),
        "slip_to_stick": np.asarray(
            visualization["trained_result"].slip_to_stick
        ).tolist(),
    }
    passed = bool(
        np.all(np.isfinite([row["objective"] for row in history]))
        and np.all(np.isfinite([row["gradient_norm"] for row in history[1:]]))
        and trained_train_objective < fixed_train_objective
        and trained_test_objective < fixed_test_objective
        and _switching_gate(trained_test)
    )
    summary = {
        "mesh": "32x4 QUAD4",
        "elements": 128,
        "nodes": 165,
        "total_dofs": 330,
        "free_dofs": 320,
        "contact_coordinates": SYSTEM.contact_coordinates.tolist(),
        "omega_1": SYSTEM.omega_1,
        "initial_train_objective": fixed_train_objective,
        "final_train_objective": trained_train_objective,
        "train_relative_improvement": train_improvement,
        "fixed_test_objective": fixed_test_objective,
        "trained_test_objective": trained_test_objective,
        "test_relative_improvement": test_improvement,
        "test_win_count": win_count,
        "gradient_norm_start": history[1]["gradient_norm"],
        "gradient_norm_end": history[-1]["gradient_norm"],
        "N_min_final": history[-1]["N_min"],
        "N_max_final": history[-1]["N_max"],
        "fixed_train_stick_to_slip": np.sum(
            fixed_train.stick_to_slip, axis=0
        ).tolist(),
        "fixed_train_slip_to_stick": np.sum(
            fixed_train.slip_to_stick, axis=0
        ).tolist(),
        "trained_test_stick_to_slip": np.sum(
            trained_test.stick_to_slip, axis=0
        ).tolist(),
        "trained_test_slip_to_stick": np.sum(
            trained_test.slip_to_stick, axis=0
        ).tolist(),
        "representative_seed": visualization["representative_seed"],
        "representative_median_improvement": visualization["median_improvement"],
        "representative_fixed_transitions": representative_fixed_transitions,
        "representative_trained_transitions": representative_trained_transitions,
        "preflight_backward_seconds": preflight_result["backward_seconds"],
        "training_seconds": training_seconds,
        "held_out_evaluation_seconds": evaluation_seconds,
        "trajectory_seconds": visualization["trajectory_seconds"],
        "vtk_export_seconds": export_seconds,
        "peak_rss_gib": _peak_rss_gib(),
        "pass": passed,
    }
    summary_path = OUTPUT_DIRECTORY / "showcase_summary.json"
    _write_summary(summary_path, summary)

    print("## Larger FEM")
    print(f"contact_coordinates: {SYSTEM.contact_coordinates.tolist()}")
    print(
        "fixed_train_transitions: "
        f"stick_to_slip={summary['fixed_train_stick_to_slip']} "
        f"slip_to_stick={summary['fixed_train_slip_to_stick']}"
    )
    print("## 100-iteration optimization")
    print(f"J_initial: {fixed_train_objective:.16g}")
    print(f"J_final: {trained_train_objective:.16g}")
    print(f"train_relative_improvement: {train_improvement:.16g}")
    print(f"gradient_norm_start: {history[1]['gradient_norm']:.16g}")
    print(f"gradient_norm_end: {history[-1]['gradient_norm']:.16g}")
    print(f"N_final: [{history[-1]['N_min']:.16g}, {history[-1]['N_max']:.16g}]")
    print("## Held-out evaluation")
    print(f"J_fixed_test: {fixed_test_objective:.16g}")
    print(f"J_mlp_test: {trained_test_objective:.16g}")
    print(f"test_relative_improvement: {test_improvement:.16g}")
    print(f"test_win_count: {win_count}/64")
    print(f"representative_seed: {visualization['representative_seed']}")
    print(f"representative_fixed_transitions: {representative_fixed_transitions}")
    print(f"representative_trained_transitions: {representative_trained_transitions}")
    print("## Runtime")
    print(f"preflight_backward_seconds: {preflight_result['backward_seconds']:.9g}")
    print(f"training_seconds: {training_seconds:.9g}")
    print(f"held_out_evaluation_seconds: {evaluation_seconds:.9g}")
    print(f"trajectory_seconds: {visualization['trajectory_seconds']:.9g}")
    print(f"vtk_export_seconds: {export_seconds:.9g}")
    print(f"peak_rss_gib: {_peak_rss_gib():.9g}")
    print("## Visualization")
    for path in figure_paths:
        print(path.resolve())
    print(deformation_gif.resolve())
    print(progress_gif.resolve())
    print(fixed_pvd.resolve())
    print(trained_pvd.resolve())
    print(history_path.resolve())
    print(summary_path.resolve())
    print("## PASS" if passed else "## FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
