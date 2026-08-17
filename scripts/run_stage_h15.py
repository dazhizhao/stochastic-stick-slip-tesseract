"""Run the complete local Stage H1.5 validation and bounded descent."""

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from stochastic_stick_slip.model import (
    FD_RELATIVE_EPSILON,
    NUM_ELEMENTS_X,
    NUM_ELEMENTS_Y,
    SYSTEM,
    TRAINING_SEEDS,
    crn_fd_jacobian,
    evaluate_batch,
    select_baseline,
)


PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OBJECTIVE_API = ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/stage_h15"
STEP_SIZES = (0.00125, 0.0025, 0.005, 0.01)
MAX_DESCENT_STEPS = 5
OBJECTIVE_TIME_LIMIT = 10.0
GRADIENT_TIME_LIMIT = 60.0


def _style_axis(axis):
    axis.set_facecolor("white")
    axis.grid(False)
    axis.tick_params(
        direction="in",
        top=False,
        right=False,
        width=0.9,
        colors="#1A1A1A",
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#1A1A1A")
        spine.set_linewidth(0.99)


def _configure_figure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "legend.frameon": False,
        }
    )


def _plot_results(baseline, history):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _configure_figure_style()
    times = np.asarray(SYSTEM.times)

    response_path = OUTPUT_DIRECTORY / "mesh_and_two_contact_response.png"
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 8.0),
        gridspec_kw={"height_ratios": [0.8, 2.0, 1.0, 1.0]},
    )
    mesh_axis = axes[0]
    for cell in SYSTEM.cells:
        polygon = SYSTEM.points[np.append(cell, cell[0])]
        mesh_axis.plot(polygon[:, 0], polygon[:, 1], color="#606060", linewidth=0.7)
    mesh_axis.scatter(
        SYSTEM.contact_coordinates[:, 0],
        SYSTEM.contact_coordinates[:, 1],
        marker="^",
        s=52,
        color=["#145DA0", "#B23A48"],
        zorder=3,
    )
    mesh_axis.text(
        SYSTEM.contact_coordinates[0, 0],
        -0.025,
        "A",
        color="#145DA0",
        ha="center",
        va="top",
        fontweight="bold",
    )
    mesh_axis.text(
        SYSTEM.contact_coordinates[1, 0],
        -0.025,
        "B",
        color="#B23A48",
        ha="center",
        va="top",
        fontweight="bold",
    )
    mesh_axis.set_xlim(-0.02, 1.02)
    mesh_axis.set_ylim(-0.06, 0.13)
    mesh_axis.set_ylabel("y")
    mesh_axis.tick_params(labelbottom=False)

    axes[1].plot(
        times,
        np.asarray(baseline.displacement[0]),
        color="#303030",
        linewidth=1.6,
    )
    axes[1].set_ylabel("Displacement")
    axes[1].tick_params(labelbottom=False)

    contact_colors = ("#145DA0", "#B23A48")
    for contact_index, axis in enumerate(axes[2:]):
        axis.step(
            times,
            np.asarray(baseline.slip[0, :, contact_index], dtype=int),
            where="post",
            color=contact_colors[contact_index],
            linewidth=1.6,
        )
        axis.set_ylabel(f"Contact {chr(65 + contact_index)}")
        axis.set_yticks([0, 1], labels=["STICK", "SLIP"])
        if contact_index == 0:
            axis.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    figure.savefig(response_path, dpi=300)
    plt.close(figure)

    history_path = OUTPUT_DIRECTORY / "objective_history.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    iterations = [entry["iteration"] for entry in history]
    objectives = [entry["objective"] for entry in history]
    axis.plot(
        iterations,
        objectives,
        color="#145DA0",
        marker="o",
        markersize=5,
        linewidth=1.8,
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    axis.set_xticks(iterations)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(history_path, dpi=300)
    plt.close(figure)
    return response_path, history_path


def _timed_forward(q, seeds):
    start = time.perf_counter()
    result = evaluate_batch(q, seeds)
    result.losses.block_until_ready()
    return result, time.perf_counter() - start


def main() -> int:
    q0, baseline = select_baseline(TRAINING_SEEDS)
    if q0 is None:
        print("## FAIL")
        print("No fixed preload candidate passed the two-contact switching gate.")
        return 1

    # Compile each batch shape once, then record one warm wall-time sample.
    evaluate_batch(q0, TRAINING_SEEDS[:1]).losses.block_until_ready()
    _, single_seed_time = _timed_forward(q0, TRAINING_SEEDS[:1])
    _, objective_time = _timed_forward(q0, TRAINING_SEEDS)
    gradient_start = time.perf_counter()
    nominal_jacobian = crn_fd_jacobian(q0, TRAINING_SEEDS)
    nominal_gradient = nominal_jacobian.mean(axis=0)
    gradient_time = time.perf_counter() - gradient_start
    performance_gate = (
        objective_time <= OBJECTIVE_TIME_LIMIT
        and gradient_time <= GRADIENT_TIME_LIMIT
    )

    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    objective = Tesseract.from_tesseract_api(OBJECTIVE_API)
    seeds = jnp.asarray(TRAINING_SEEDS)
    zero_coefficients = jnp.zeros((8, 5), dtype=jnp.float64)

    def pipeline(q):
        response = apply_tesseract(
            physics,
            {"q": q, "coeffs": zero_coefficients, "seeds": seeds},
        )
        return apply_tesseract(
            objective, {"seed_losses": response["seed_losses"]}
        )["objective"]

    physics_inputs = {
        "q": q0,
        "coeffs": np.zeros((8, 5), dtype=np.float64),
        "seeds": TRAINING_SEEDS,
    }
    physics_output = physics.apply(physics_inputs)
    physics_jvp = physics.jacobian_vector_product(
        physics_inputs,
        ["q"],
        ["seed_losses"],
        {"q": np.array([1.0, 0.0])},
    )
    physics_vjp = physics.vector_jacobian_product(
        physics_inputs,
        ["q"],
        ["seed_losses"],
        {"seed_losses": np.ones(8)},
    )
    objective0, tesseract_gradient = jax.value_and_grad(pipeline)(
        jnp.asarray(q0)
    )
    objective0 = float(objective0)
    tesseract_gradient = np.asarray(tesseract_gradient)

    gradient_scales = {
        0.5: crn_fd_jacobian(q0, TRAINING_SEEDS, 0.5).mean(axis=0),
        1.0: nominal_gradient,
        2.0: crn_fd_jacobian(q0, TRAINING_SEEDS, 2.0).mean(axis=0),
    }

    def direction_cosine(first, second):
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float(first @ second / denominator)

    direction_cosines = (
        direction_cosine(gradient_scales[0.5], gradient_scales[1.0]),
        direction_cosine(gradient_scales[1.0], gradient_scales[2.0]),
    )

    epsilon_n = FD_RELATIVE_EPSILON * q0[1]
    q_minus_n = q0.copy()
    q_plus_n = q0.copy()
    q_minus_n[1] -= epsilon_n
    q_plus_n[1] += epsilon_n
    minus_n = evaluate_batch(q_minus_n, TRAINING_SEEDS)
    plus_n = evaluate_batch(q_plus_n, TRAINING_SEEDS)
    changed_states = np.asarray(minus_n.slip) != np.asarray(plus_n.slip)
    changed_seed_locations = np.any(changed_states, axis=1)

    complete_cycles = np.logical_and(
        np.asarray(baseline.stick_to_slip) > 0,
        np.asarray(baseline.slip_to_stick) > 0,
    )
    switching_seeds = np.any(complete_cycles, axis=1)
    physical_gate = (
        np.count_nonzero(switching_seeds) >= 4
        and np.all(np.any(complete_cycles, axis=0))
        and bool(np.any(changed_seed_locations))
    )
    endpoint_gate = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (
            physics_output["seed_losses"],
            physics_jvp["seed_losses"],
            physics_vjp["q"],
        )
    )
    gradient_gate = (
        all(
            np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0.0
            for gradient in gradient_scales.values()
        )
        and direction_cosines[0] > 0.0
        and direction_cosines[1] > 0.0
        and np.all(np.isfinite(tesseract_gradient))
        and np.allclose(
            tesseract_gradient,
            nominal_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    history = [
        {
            "iteration": 0,
            "q": q0.copy(),
            "objective": objective0,
            "relative_change": 0.0,
            "alpha": None,
        }
    ]
    current_q = q0.copy()
    current_objective = objective0
    current_gradient = nominal_gradient
    if physical_gate and endpoint_gate and gradient_gate and performance_gate:
        for iteration in range(1, MAX_DESCENT_STEPS + 1):
            gradient_norm = np.linalg.norm(current_gradient)
            if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
                break
            accepted = None
            for alpha in STEP_SIZES:
                candidate = current_q - alpha * current_gradient / gradient_norm
                if np.any(candidate <= 0.0):
                    continue
                candidate_objective = float(pipeline(jnp.asarray(candidate)))
                if candidate_objective < current_objective:
                    accepted = (candidate, candidate_objective, alpha)
                    break
            if accepted is None:
                break
            candidate, candidate_objective, alpha = accepted
            relative_change = (
                candidate_objective - current_objective
            ) / current_objective
            history.append(
                {
                    "iteration": iteration,
                    "q": candidate.copy(),
                    "objective": candidate_objective,
                    "relative_change": relative_change,
                    "alpha": alpha,
                }
            )
            current_q = candidate
            current_objective = candidate_objective
            if iteration < MAX_DESCENT_STEPS:
                _, current_gradient = jax.value_and_grad(pipeline)(
                    jnp.asarray(current_q)
                )
                current_gradient = np.asarray(current_gradient)

    passed = len(history) >= 2
    print("## Summary")
    print(f"mesh: {NUM_ELEMENTS_X}x{NUM_ELEMENTS_Y} QUAD4")
    print(f"dofs: total={SYSTEM.num_total_dofs}, free={SYSTEM.num_free_dofs}")
    print(f"contact_coordinates: {SYSTEM.contact_coordinates.tolist()}")
    print(f"q0: {q0.tolist()}")
    print(f"omega_1: {SYSTEM.omega_1:.12g}")
    print("## Results")
    print(f"seed_losses: {np.asarray(baseline.losses).tolist()}")
    print(f"J(q0): {objective0:.16g}")
    print(
        "displacement_range: "
        f"[{np.min(baseline.displacement):.12g}, "
        f"{np.max(baseline.displacement):.12g}]"
    )
    print(
        "velocity_range: "
        f"[{np.min(baseline.velocity):.12g}, "
        f"{np.max(baseline.velocity):.12g}]"
    )
    print(f"stick_to_slip: {np.asarray(baseline.stick_to_slip).tolist()}")
    print(f"slip_to_stick: {np.asarray(baseline.slip_to_stick).tolist()}")
    print(f"switching_seeds: {np.flatnonzero(switching_seeds).tolist()}")
    print(f"N_perturbation_changed_seed_locations: {changed_seed_locations.tolist()}")
    print("## Gradient check")
    for multiplier, gradient in gradient_scales.items():
        print(f"{multiplier:g}x: {gradient.tolist()}")
    print(f"direction_cosines: {list(direction_cosines)}")
    print(f"Tesseract value_and_grad: {tesseract_gradient.tolist()}")
    print("## Performance")
    print(f"single_seed_forward_seconds: {single_seed_time:.9g}")
    print(f"eight_seed_objective_seconds: {objective_time:.9g}")
    print(f"nominal_gradient_seconds: {gradient_time:.9g}")
    print("## Descent")
    for entry in history:
        print(
            f"iter={entry['iteration']} q={entry['q'].tolist()} "
            f"J={entry['objective']:.16g} "
            f"relative_change={entry['relative_change']:.12g} "
            f"alpha={entry['alpha']}"
        )
    print("## PASS" if passed else "## FAIL")
    if passed:
        for path in _plot_results(baseline, history):
            print(f"figure: {path}")
    else:
        print(f"physical_gate: {physical_gate}")
        print(f"endpoint_gate: {endpoint_gate}")
        print(f"gradient_gate: {gradient_gate}")
        print(f"performance_gate: {performance_gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
