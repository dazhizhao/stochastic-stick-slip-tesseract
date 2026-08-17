"""Run the complete local Stage H1 validation and one-step descent."""

from pathlib import Path

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
    SYSTEM,
    TRAINING_SEEDS,
    calibrate_baseline,
    crn_fd_jacobian,
    evaluate_batch,
)


ROOT = Path(__file__).resolve().parents[1]
PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OBJECTIVE_API = ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/stage_h1"


def _plot_results(q0, q1, baseline, improved):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    times = np.asarray(SYSTEM.times)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "legend.frameon": False,
        }
    )

    def style_axis(axis):
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
            spine.set_color("#1A1A1A")
            spine.set_linewidth(0.99)

    trajectory_path = OUTPUT_DIRECTORY / "representative_trajectory.png"
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    axes[0].plot(
        times,
        np.asarray(baseline.displacement[0]),
        color="#145DA0",
        linewidth=1.8,
    )
    axes[0].set_ylabel("Displacement")
    axes[1].plot(
        times,
        np.asarray(baseline.velocity[0]),
        color="#B23A48",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Velocity")
    axes[2].step(
        times,
        np.asarray(baseline.slip[0], dtype=int),
        where="post",
        color="#1B4332",
        linewidth=1.8,
    )
    axes[2].set_ylabel("Slip state")
    axes[2].set_xlabel("Time")
    axes[2].set_yticks([0, 1], labels=["STICK", "SLIP"])
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    figure.savefig(trajectory_path, dpi=300)
    plt.close(figure)

    descent_path = OUTPUT_DIRECTORY / "one_step_descent.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        times,
        np.asarray(baseline.displacement[0]),
        label=f"q0 = ({q0[0]:.4f}, {q0[1]:.4f})",
        color="#145DA0",
        linewidth=1.8,
    )
    axis.plot(
        times,
        np.asarray(improved.displacement[0]),
        label=f"q1 = ({q1[0]:.4f}, {q1[1]:.4f})",
        color="#B23A48",
        linewidth=1.8,
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Displacement")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(descent_path, dpi=300)
    plt.close(figure)
    return trajectory_path, descent_path


def main() -> int:
    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    objective = Tesseract.from_tesseract_api(OBJECTIVE_API)
    seeds = jnp.asarray(TRAINING_SEEDS)

    def pipeline(q):
        response = apply_tesseract(physics, {"q": q, "seeds": seeds})
        return apply_tesseract(
            objective, {"seed_losses": response["seed_losses"]}
        )["objective"]

    q0 = calibrate_baseline(TRAINING_SEEDS)
    baseline = evaluate_batch(q0, TRAINING_SEEDS)
    physics_output = physics.apply({"q": q0, "seeds": TRAINING_SEEDS})
    objective0, tesseract_gradient = jax.value_and_grad(pipeline)(jnp.asarray(q0))
    objective0 = float(objective0)
    tesseract_gradient = np.asarray(tesseract_gradient)

    gradient_scales = {
        multiplier: crn_fd_jacobian(q0, TRAINING_SEEDS, multiplier).mean(axis=0)
        for multiplier in (0.5, 1.0, 2.0)
    }
    nominal_gradient = gradient_scales[1.0]
    direction_dots = (
        float(gradient_scales[0.5] @ gradient_scales[1.0]),
        float(gradient_scales[1.0] @ gradient_scales[2.0]),
    )

    epsilon_n = FD_RELATIVE_EPSILON * q0[1]
    q_minus_n = q0.copy()
    q_plus_n = q0.copy()
    q_minus_n[1] -= epsilon_n
    q_plus_n[1] += epsilon_n
    minus_n = evaluate_batch(q_minus_n, TRAINING_SEEDS)
    plus_n = evaluate_batch(q_plus_n, TRAINING_SEEDS)
    changed_seeds = np.any(
        np.asarray(minus_n.slip) != np.asarray(plus_n.slip), axis=1
    )

    finite_forward = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (
            physics_output["seed_losses"],
            physics_output["displacement_min"],
            physics_output["displacement_max"],
            physics_output["velocity_min"],
            physics_output["velocity_max"],
        )
    )
    dual_transition_seeds = np.logical_and(
        np.asarray(baseline.stick_to_slip) > 0,
        np.asarray(baseline.slip_to_stick) > 0,
    )
    physical_gate = finite_forward and np.count_nonzero(dual_transition_seeds) >= 2
    gradient_gate = (
        all(np.all(np.isfinite(gradient)) for gradient in gradient_scales.values())
        and np.all(np.isfinite(tesseract_gradient))
        and np.allclose(tesseract_gradient, nominal_gradient, rtol=1e-10, atol=1e-12)
        and direction_dots[0] > 0.0
        and direction_dots[1] > 0.0
    )
    event_gate = bool(np.any(changed_seeds))

    q1 = None
    objective1 = None
    improved = None
    accepted_step = None
    if physical_gate and gradient_gate and event_gate:
        direction = nominal_gradient / np.linalg.norm(nominal_gradient)
        for multiplier in (0.05, 0.10, 0.20):
            step = multiplier * float(np.min(q0))
            candidate = q0 - step * direction
            if np.any(candidate <= 0.0):
                continue
            candidate_objective = float(pipeline(jnp.asarray(candidate)))
            if candidate_objective < objective0:
                q1 = candidate
                objective1 = candidate_objective
                improved = evaluate_batch(q1, TRAINING_SEEDS)
                accepted_step = multiplier
                break

    passed = q1 is not None
    print("## Summary")
    print(f"q0: {q0.tolist()}")
    print(f"omega_1: {SYSTEM.omega_1:.12g}")
    print(f"time_step: {SYSTEM.time_step:.12g}")
    print("## Results")
    print(f"seed_losses: {np.asarray(baseline.losses).tolist()}")
    print(f"J(q0): {objective0:.16g}")
    print(f"displacement_range: [{np.min(baseline.displacement):.12g}, {np.max(baseline.displacement):.12g}]")
    print(f"velocity_range: [{np.min(baseline.velocity):.12g}, {np.max(baseline.velocity):.12g}]")
    print(f"stick_to_slip: {np.asarray(baseline.stick_to_slip).tolist()}")
    print(f"slip_to_stick: {np.asarray(baseline.slip_to_stick).tolist()}")
    print(f"dual_transition_seeds: {np.flatnonzero(dual_transition_seeds).tolist()}")
    print(f"N_perturbation_changed_seeds: {np.flatnonzero(changed_seeds).tolist()}")
    print("## Gradient check")
    for multiplier, gradient in gradient_scales.items():
        print(f"{multiplier:g}x: {gradient.tolist()}")
    print(f"Tesseract value_and_grad: {tesseract_gradient.tolist()}")
    print(f"adjacent_direction_dots: {list(direction_dots)}")
    print("## PASS" if passed else "## FAIL")
    if passed:
        relative_improvement = (objective0 - objective1) / objective0
        print(f"accepted_alpha_multiplier: {accepted_step}")
        print(f"q1: {q1.tolist()}")
        print(f"J(q1): {objective1:.16g}")
        print(f"relative_improvement: {relative_improvement:.12g}")
        for path in _plot_results(q0, q1, baseline, improved):
            print(f"figure: {path}")
    else:
        print(f"physical_gate: {physical_gate}")
        print(f"gradient_gate: {gradient_gate}")
        print(f"event_gate: {event_gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
