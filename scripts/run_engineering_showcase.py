"""Train the locked low-damping resonant engineering showcase for 500 steps."""

from __future__ import annotations

import json
import os
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
from matplotlib.colors import Normalize
import numpy as np
import torch
from tesseract_core import Tesseract

import scripts.run_showcase as shared
from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.engineering_showcase import (
    FIRST_FREQUENCY_RATIO,
    NUM_ELEMENTS_X,
    NUM_ELEMENTS_Y,
    NUM_STEPS,
    SECOND_FREQUENCY_RATIO,
    SYSTEM,
    evaluate_full_trajectory,
    full_nodal_field,
    preload_history,
)


CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/engineering_showcase"
BASE_Q = np.array([0.10, 0.04], dtype=np.float64)
LEARNING_RATE = 0.1
PROBE_BASELINE_OBJECTIVE = 0.010061224717546585
BASELINE_RTOL = 1e-10
BASELINE_ATOL = 1e-12
NUM_PHYSICAL_FRAMES = 80


def _configure_shared_runner() -> None:
    """Bind the mature showcase utilities to the frozen engineering case."""
    shared.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    shared.BASE_Q = BASE_Q
    shared.SYSTEM = SYSTEM
    shared.NUM_ELEMENTS_X = NUM_ELEMENTS_X
    shared.NUM_ELEMENTS_Y = NUM_ELEMENTS_Y
    shared.NUM_STEPS = NUM_STEPS
    shared.LEARNING_RATE = LEARNING_RATE
    shared.preload_history = preload_history
    shared.evaluate_full_trajectory = evaluate_full_trajectory
    shared.full_nodal_field = full_nodal_field
    shared._style_axis = _style_axis


def _configure_style() -> None:
    shared._configure_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.grid": False,
        }
    )


def _style_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(shared.FRAME_COLOR)
        spine.set_linewidth(1.1)
    axis.tick_params(
        top=False,
        right=False,
        direction="in",
        width=1.0,
        color=shared.FRAME_COLOR,
    )


def create_tesseracts():
    controller = Tesseract.from_tesseract_api(CONTROLLER_API)
    previous_variant = os.environ.get("STICK_SLIP_FEM_VARIANT")
    os.environ["STICK_SLIP_FEM_VARIANT"] = "engineering_showcase"
    try:
        physics = Tesseract.from_tesseract_api(PHYSICS_API)
    finally:
        if previous_variant is None:
            os.environ.pop("STICK_SLIP_FEM_VARIANT", None)
        else:
            os.environ["STICK_SLIP_FEM_VARIANT"] = previous_variant
    return controller, physics


def plot_cumulative_vibration(replay) -> Path:
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    divisor = np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    initial = np.cumsum(replay["fixed_observation"] ** 2) / divisor
    optimized = np.cumsum(replay["displacement"][-1] ** 2) / divisor
    figure, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    axis.plot(
        periods,
        initial,
        color=shared.FIXED_COLOR,
        linewidth=1.5,
        label="Initial",
    )
    axis.plot(
        periods,
        optimized,
        color=shared.MLP_COLOR,
        linewidth=1.7,
        label="Iteration 500",
    )
    axis.set(xlabel="Time (periods)", ylabel="Running mean-square displacement")
    axis.legend(loc="best")
    _style_axis(axis)
    path = OUTPUT_DIRECTORY / "cumulative_vibration.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_representative_response(replay) -> Path:
    periods = np.asarray(SYSTEM.times) * SYSTEM.omega_1 / (2.0 * np.pi)
    figure, axes = plt.subplots(
        2, 1, figsize=(7.8, 6.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        periods,
        replay["fixed_observation"],
        color=shared.FIXED_COLOR,
        linewidth=1.5,
        label="Initial",
    )
    axes[0].plot(
        periods,
        replay["displacement"][-1],
        color=shared.MLP_COLOR,
        linewidth=1.7,
        label="Iteration 500",
    )
    axes[0].set_ylabel("Displacement")
    axes[0].legend(loc="best")
    axes[1].axhline(
        BASE_Q[1],
        color=shared.FIXED_COLOR,
        linewidth=1.3,
        linestyle="--",
        label="Initial",
    )
    axes[1].plot(
        periods,
        replay["preload"][-1],
        color=shared.MLP_COLOR,
        linewidth=1.7,
        label="Iteration 500",
    )
    axes[1].set(xlabel="Time (periods)", ylabel="Preload")
    axes[1].legend(loc="best")
    for axis in axes:
        _style_axis(axis)
    path = OUTPUT_DIRECTORY / "representative_response.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def create_deformation_gif(replay):
    initial = replay["fixed"]["displacement"]
    optimized = replay["final"]["displacement"]
    initial_slip = replay["fixed"]["slip"]
    optimized_slip = replay["final"]["slip"]
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
    figure, axes = plt.subplots(
        1, 2, figsize=(8.0, 2.7), dpi=95, constrained_layout=True
    )
    collections = []
    contact_artists = []
    for axis, label in zip(axes, ("Initial", "Iteration 500 MLP")):
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
            axis.scatter(
                [], [], s=32, edgecolor="white", linewidth=0.6, zorder=5
            )
            for _ in range(2)
        ]
        axis.text(
            0.02,
            0.94,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            weight="bold",
        )
        axis.set(xlim=(-0.03, 1.04), ylim=(-0.18, 0.28), xlabel="x")
        axis.set_aspect("equal")
        _style_axis(axis)
        collections.append(collection)
        contact_artists.append(artists)
    axes[0].set_ylabel("y")
    colorbar = figure.colorbar(collections[-1], ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("Displacement magnitude")
    annotation = figure.text(0.5, 0.995, "", ha="center", va="top", fontsize=10)
    frames = []
    for step in frame_indices:
        for panel, (field, states) in enumerate(
            ((initial, initial_slip), (optimized, optimized_slip))
        ):
            deformed = SYSTEM.points + deformation_scale * field[step]
            collections[panel].set_verts(deformed[SYSTEM.cells])
            collections[panel].set_array(
                np.mean(
                    np.linalg.norm(field[step], axis=1)[SYSTEM.cells], axis=1
                )
            )
            for contact, artist in enumerate(contact_artists[panel]):
                artist.set_offsets(deformed[SYSTEM.contact_nodes[contact]][None, :])
                artist.set_color(
                    shared.SLIP_COLOR if states[step, contact] else shared.STICK_COLOR
                )
        periods = float(SYSTEM.times[step] * SYSTEM.omega_1 / (2.0 * np.pi))
        annotation.set_text(
            f"Held-out seed {replay['representative_seed']}   "
            f"Time = {periods:.2f} periods   blue: STICK, orange: SLIP"
        )
        frames.append(shared._quantized_frame(figure, colors=128))
    plt.close(figure)
    path = OUTPUT_DIRECTORY / "initial_vs_optimized_deformation.gif"
    shared._save_gif(frames, path, duration=85)
    return path, deformation_scale, magnitude_max


def _validate_history(history) -> None:
    expected_shapes = {
        "theta": (501, 469),
        "objective": (501,),
        "gradient": (500,),
        "n_min": (501,),
        "n_max": (501,),
    }
    for name, shape in expected_shapes.items():
        values = getattr(history, name)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid {name} history: {values.shape}")


def _classification(relative_improvement: float) -> str:
    if relative_improvement >= 0.20:
        return "Strong"
    if relative_improvement >= 0.10:
        return "Moderate"
    return "Weak"


def main() -> int:
    torch.set_default_dtype(torch.float64)
    _configure_shared_runner()
    _configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    controller, physics = create_tesseracts()

    preflight = shared.preflight(controller, physics)
    fixed_train_objective = float(np.mean(preflight["fixed"].losses))
    baseline_matches_probe = bool(
        np.isclose(
            fixed_train_objective,
            PROBE_BASELINE_OBJECTIVE,
            rtol=BASELINE_RTOL,
            atol=BASELINE_ATOL,
        )
    )
    print("## Preflight")
    print(f"mesh: {NUM_ELEMENTS_X}x{NUM_ELEMENTS_Y} QUAD4")
    print(f"free_dofs: {SYSTEM.num_free_dofs}")
    print(f"damping: {BASE_Q[0]:.16g}")
    print(f"learning_rate: {LEARNING_RATE:.16g}")
    print(f"frequency_ratios: [{FIRST_FREQUENCY_RATIO}, {SECOND_FREQUENCY_RATIO}]")
    print(f"fixed_train_objective: {fixed_train_objective:.16g}")
    print(f"probe_baseline_match: {baseline_matches_probe}")
    print(f"backward_seconds: {preflight['backward_seconds']:.9g}")
    print(f"backward_gradient_norm: {preflight['backward_gradient_norm']:.16g}")
    print(f"switching_gate: {shared._switching_gate(preflight['fixed'])}")
    if not preflight["gate"] or not baseline_matches_probe:
        print("## FAIL")
        return 1

    history, training_seconds, history_path = shared.train(
        controller, physics, fixed_train_objective
    )
    _validate_history(history)
    final_theta = history.theta[-1]
    trained_train = shared.evaluate(
        controller, physics, final_theta, H4_TRAINING_SEEDS
    )

    evaluation_start = time.perf_counter()
    initial_test = shared.evaluate(
        controller, physics, final_theta, H4_TEST_SEEDS, fixed=True
    )
    optimized_test = shared.evaluate(
        controller, physics, final_theta, H4_TEST_SEEDS
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    initial_test_objective = float(np.mean(initial_test.losses))
    optimized_test_objective = float(np.mean(optimized_test.losses))
    test_improvement = (
        initial_test_objective - optimized_test_objective
    ) / initial_test_objective
    per_seed_improvement = (
        initial_test.losses - optimized_test.losses
    ) / initial_test.losses
    median_improvement = float(np.median(per_seed_improvement))
    win_count = int(np.count_nonzero(optimized_test.losses < initial_test.losses))
    result_classification = _classification(test_improvement)

    replay = shared.replay_representative(
        controller, history, initial_test, optimized_test
    )
    visualization_start = time.perf_counter()
    held_out_path, held_out_statistics = shared.plot_held_out_distribution(
        initial_test.losses, optimized_test.losses
    )
    representative_path = plot_representative_response(replay)
    cumulative_path = plot_cumulative_vibration(replay)
    optimizer_gif = shared.create_all_iterations_gif(history, replay)
    deformation_gif, deformation_scale, displacement_color_max = (
        create_deformation_gif(replay)
    )
    visualization_seconds = time.perf_counter() - visualization_start
    media_paths = [
        held_out_path,
        representative_path,
        cumulative_path,
        optimizer_gif,
        deformation_gif,
    ]
    shared._validate_media(
        media_paths,
        {optimizer_gif: 501, deformation_gif: NUM_PHYSICAL_FRAMES},
    )

    minimum_iteration = int(np.argmin(history.objective))
    minimum_objective = float(history.objective[minimum_iteration])
    representative_initial_transitions = shared._transition_summary(
        replay["fixed_result"]
    )
    representative_final_transitions = shared._transition_summary(
        replay["final_result"]
    )
    passed = bool(
        preflight["gate"]
        and baseline_matches_probe
        and shared._switching_gate(trained_train)
        and shared._switching_gate(optimized_test)
        and np.all(np.isfinite(initial_test.losses))
        and np.all(np.isfinite(optimized_test.losses))
        and len(media_paths) == 5
    )
    summary = {
        "operating_condition": {
            "damping": float(BASE_Q[0]),
            "base_preload": float(BASE_Q[1]),
            "first_frequency_ratio": FIRST_FREQUENCY_RATIO,
            "second_frequency_ratio": SECOND_FREQUENCY_RATIO,
            "learning_rate": LEARNING_RATE,
        },
        "mesh": "32x4 QUAD4",
        "free_dofs": SYSTEM.num_free_dofs,
        "milestone_objectives": {
            str(iteration): float(history.objective[iteration])
            for iteration in shared.MILESTONE_ITERATIONS
        },
        "minimum_train_objective": minimum_objective,
        "minimum_train_iteration": minimum_iteration,
        "train_improvement_0_to_100": float(
            (history.objective[0] - history.objective[100]) / history.objective[0]
        ),
        "train_improvement_0_to_500": float(
            (history.objective[0] - history.objective[500]) / history.objective[0]
        ),
        "gradient_norm_start": float(history.gradient[0]),
        "gradient_norm_end": float(history.gradient[-1]),
        "N_min_final": float(history.n_min[-1]),
        "N_max_final": float(history.n_max[-1]),
        "initial_test_objective": initial_test_objective,
        "iteration_500_test_objective": optimized_test_objective,
        "test_relative_improvement": float(test_improvement),
        "test_median_seed_improvement": median_improvement,
        "test_win_count": win_count,
        "classification": result_classification,
        "test_initial_losses": initial_test.losses.tolist(),
        "test_iteration_500_losses": optimized_test.losses.tolist(),
        "test_seed_relative_improvement": per_seed_improvement.tolist(),
        "held_out_distribution_statistics": {
            key: value
            for key, value in held_out_statistics.items()
            if key != "improvement"
        },
        "representative_seed": replay["representative_seed"],
        "representative_median_improvement": replay["median_improvement"],
        "representative_initial_transitions": representative_initial_transitions,
        "representative_iteration_500_transitions": representative_final_transitions,
        "preflight_backward_seconds": preflight["backward_seconds"],
        "training_seconds": training_seconds,
        "held_out_evaluation_seconds": evaluation_seconds,
        "representative_replay_seconds": replay["seconds"],
        "visualization_seconds": visualization_seconds,
        "deformation_scale": deformation_scale,
        "displacement_color_max": displacement_color_max,
        "pass": passed,
    }
    summary_path = OUTPUT_DIRECTORY / "engineering_showcase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("## 500-step optimization")
    for iteration in shared.MILESTONE_ITERATIONS:
        print(f"J_iter{iteration}: {history.objective[iteration]:.16g}")
    print(f"J_minimum: {minimum_objective:.16g} at iteration {minimum_iteration}")
    print(f"gradient_norm_end: {history.gradient[-1]:.16g}")
    print(f"N_final: [{history.n_min[-1]:.16g}, {history.n_max[-1]:.16g}]")
    print("## Held-out")
    print(f"J_initial_test: {initial_test_objective:.16g}")
    print(f"J_iter500_test: {optimized_test_objective:.16g}")
    print(f"aggregate_improvement: {test_improvement:.16g}")
    print(f"median_seed_improvement: {median_improvement:.16g}")
    print(f"wins: {win_count}/64")
    print(f"classification: {result_classification}")
    print(f"representative_seed: {replay['representative_seed']}")
    print(f"representative_initial_transitions: {representative_initial_transitions}")
    print(f"representative_iteration_500_transitions: {representative_final_transitions}")
    print("## Runtime")
    print(f"preflight_backward_seconds: {preflight['backward_seconds']:.9g}")
    print(f"training_seconds: {training_seconds:.9g}")
    print(f"held_out_evaluation_seconds: {evaluation_seconds:.9g}")
    print(f"representative_replay_seconds: {replay['seconds']:.9g}")
    print(f"visualization_seconds: {visualization_seconds:.9g}")
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
