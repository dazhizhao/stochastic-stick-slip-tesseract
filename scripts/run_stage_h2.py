"""Run the complete local Stage H2 PyTorch/Tesseract training loop."""

from copy import deepcopy
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
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from stochastic_stick_slip.controller import (
    build_controller,
    parameter_gradient_norm,
)
from stochastic_stick_slip.model import (
    BASELINE_DAMPING,
    COEFFICIENT_FD_EPSILON,
    HELD_OUT_SEEDS,
    NUM_STEPS,
    SYSTEM,
    TRAINING_SEEDS,
    crn_fd_coefficient_jacobian,
    forcing_descriptor_batch,
    preload_history,
)


PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OBJECTIVE_API = ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/stage_h2"
BASE_Q = np.array([BASELINE_DAMPING, 0.04], dtype=np.float64)
ZERO_COEFFICIENTS = np.zeros((8, 5), dtype=np.float64)
EPSILON_SCALES = (0.01, 0.02, 0.04)
LEARNING_RATES = (1e-2, 5e-3, 1e-3)
MAX_ITERATIONS = 20


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


def _apply_pipeline(physics, objective, coefficients, seeds):
    response = apply_tesseract(
        physics,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds},
    )
    loss = apply_tesseract(
        objective, {"seed_losses": response["seed_losses"]}
    )["objective"]
    return loss, response


def _evaluate_numpy(physics, objective, coefficients, seeds):
    response = physics.apply(
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds}
    )
    result = objective.apply({"seed_losses": response["seed_losses"]})
    return float(result["objective"]), response


def _controller_coefficients(controller, descriptors):
    with torch.no_grad():
        return controller(descriptors).detach().cpu().numpy()


def _control_statistics(coefficients):
    histories = np.asarray(preload_history(BASE_Q[1], coefficients))
    return histories, {
        "mean": float(np.mean(histories)),
        "min": float(np.min(histories)),
        "max": float(np.max(histories)),
    }


def _direction_cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.vdot(first.ravel(), second.ravel()) / denominator)


def _plot_results(fixed_response, trained_response, trained_preload, history):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _configure_figure_style()
    times = np.asarray(SYSTEM.times)

    response_path = OUTPUT_DIRECTORY / "fourier_controlled_response.png"
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(
        times,
        np.full(NUM_STEPS, BASE_Q[1]),
        color="#606060",
        linewidth=1.6,
        label="Fixed",
    )
    axes[0].plot(
        times,
        trained_preload[0],
        color="#145DA0",
        linewidth=1.6,
        label="MLP",
    )
    axes[0].set_ylabel("Preload N(t)")
    axes[0].legend()
    axes[1].plot(
        times,
        fixed_response["representative_displacement"],
        color="#606060",
        linewidth=1.4,
        label="Fixed",
    )
    axes[1].plot(
        times,
        trained_response["representative_displacement"],
        color="#B23A48",
        linewidth=1.4,
        label="MLP",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Displacement")
    axes[1].legend()
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    figure.savefig(response_path, dpi=300)
    plt.close(figure)

    history_path = OUTPUT_DIRECTORY / "training_objective.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    iterations = [entry["iteration"] for entry in history]
    objectives = [entry["objective"] for entry in history]
    axis.plot(
        iterations,
        objectives,
        color="#145DA0",
        marker="o",
        markersize=4,
        linewidth=1.7,
        label="MLP",
    )
    axis.axhline(
        history[0]["objective"],
        color="#606060",
        linewidth=1.3,
        linestyle="--",
        label="Fixed",
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    axis.set_xticks(iterations[::2])
    axis.legend()
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(history_path, dpi=300)
    plt.close(figure)
    return response_path, history_path


def main() -> int:
    torch.set_default_dtype(torch.float64)
    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    objective = Tesseract.from_tesseract_api(OBJECTIVE_API)
    descriptors = torch.from_numpy(forcing_descriptor_batch(TRAINING_SEEDS))

    fixed_objective, fixed_response = _evaluate_numpy(
        physics, objective, ZERO_COEFFICIENTS, TRAINING_SEEDS
    )
    controller = build_controller()
    initial_coefficients = _controller_coefficients(controller, descriptors)
    initial_objective, _ = _evaluate_numpy(
        physics, objective, initial_coefficients, TRAINING_SEEDS
    )

    gradient_scales = {
        epsilon: crn_fd_coefficient_jacobian(
            BASE_Q, ZERO_COEFFICIENTS, TRAINING_SEEDS, epsilon
        )
        for epsilon in EPSILON_SCALES
    }
    direction_cosines = (
        _direction_cosine(gradient_scales[0.01], gradient_scales[0.02]),
        _direction_cosine(gradient_scales[0.02], gradient_scales[0.04]),
    )
    gradient_gate = (
        all(
            np.all(np.isfinite(gradient)) and np.linalg.norm(gradient) > 0.0
            for gradient in gradient_scales.values()
        )
        and all(cosine > 0.0 for cosine in direction_cosines)
    )

    physics_inputs = {
        "q": BASE_Q,
        "coeffs": ZERO_COEFFICIENTS,
        "seeds": TRAINING_SEEDS,
    }
    physics_output = physics.apply(physics_inputs)
    physics_jvp = physics.jacobian_vector_product(
        physics_inputs,
        ["coeffs"],
        ["seed_losses"],
        {"coeffs": np.ones((8, 5), dtype=np.float64)},
    )
    physics_vjp = physics.vector_jacobian_product(
        physics_inputs,
        ["coeffs"],
        ["seed_losses"],
        {"seed_losses": np.ones(8, dtype=np.float64)},
    )
    endpoint_gate = (
        np.all(np.isfinite(physics_output["seed_losses"]))
        and np.all(np.isfinite(physics_jvp["seed_losses"]))
        and np.all(np.isfinite(physics_vjp["coeffs"]))
        and np.allclose(
            physics_vjp["coeffs"],
            gradient_scales[COEFFICIENT_FD_EPSILON],
            rtol=1e-12,
            atol=1e-14,
        )
    )

    controller.zero_grad(set_to_none=True)
    initial_loss, _ = _apply_pipeline(
        physics, objective, controller(descriptors), TRAINING_SEEDS
    )
    initial_loss.backward()
    total_initial_gradient_norm = parameter_gradient_norm(
        controller.parameters()
    )
    final_initial_gradient_norm = parameter_gradient_norm(
        controller[-1].parameters()
    )
    backward_gate = (
        np.isfinite(float(initial_loss.detach()))
        and np.isfinite(total_initial_gradient_norm)
        and total_initial_gradient_norm > 0.0
        and np.isfinite(final_initial_gradient_norm)
        and final_initial_gradient_norm > 0.0
    )
    baseline_gate = abs(initial_objective - fixed_objective) <= 1e-12

    initial_state = deepcopy(controller.state_dict())
    history = [
        {
            "iteration": 0,
            "objective": initial_objective,
            "gradient_norm": total_initial_gradient_norm,
            "control": _control_statistics(initial_coefficients)[1],
        }
    ]
    accepted = None
    training_start = time.perf_counter()
    if gradient_gate and endpoint_gate and backward_gate and baseline_gate:
        for learning_rate in LEARNING_RATES:
            trial = build_controller()
            trial.load_state_dict(initial_state)
            optimizer = torch.optim.Adam(
                trial.parameters(), lr=learning_rate
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _apply_pipeline(
                physics, objective, trial(descriptors), TRAINING_SEEDS
            )
            loss.backward()
            gradient_norm = parameter_gradient_norm(trial.parameters())
            optimizer.step()
            coefficients = _controller_coefficients(trial, descriptors)
            candidate_objective, _ = _evaluate_numpy(
                physics, objective, coefficients, TRAINING_SEEDS
            )
            if candidate_objective < initial_objective:
                accepted = (
                    trial,
                    optimizer,
                    learning_rate,
                    gradient_norm,
                    coefficients,
                    candidate_objective,
                )
                break

    if accepted is not None:
        (
            controller,
            optimizer,
            accepted_learning_rate,
            first_gradient_norm,
            coefficients,
            current_objective,
        ) = accepted
        _, control_stats = _control_statistics(coefficients)
        history.append(
            {
                "iteration": 1,
                "objective": current_objective,
                "gradient_norm": first_gradient_norm,
                "control": control_stats,
            }
        )
        for iteration in range(2, MAX_ITERATIONS + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _apply_pipeline(
                physics, objective, controller(descriptors), TRAINING_SEEDS
            )
            loss.backward()
            gradient_norm = parameter_gradient_norm(controller.parameters())
            if not np.isfinite(gradient_norm):
                break
            optimizer.step()
            coefficients = _controller_coefficients(controller, descriptors)
            current_objective, _ = _evaluate_numpy(
                physics, objective, coefficients, TRAINING_SEEDS
            )
            if not np.isfinite(current_objective):
                break
            _, control_stats = _control_statistics(coefficients)
            history.append(
                {
                    "iteration": iteration,
                    "objective": current_objective,
                    "gradient_norm": gradient_norm,
                    "control": control_stats,
                }
            )
    else:
        accepted_learning_rate = None
        coefficients = initial_coefficients
        current_objective = initial_objective

    training_seconds = time.perf_counter() - training_start
    trained_preload, trained_control_stats = _control_statistics(coefficients)
    trained_objective, trained_response = _evaluate_numpy(
        physics, objective, coefficients, TRAINING_SEEDS
    )

    held_out_descriptors = torch.from_numpy(
        forcing_descriptor_batch(HELD_OUT_SEEDS)
    )
    held_out_coefficients = _controller_coefficients(
        controller, held_out_descriptors
    )
    held_out_fixed, _ = _evaluate_numpy(
        physics, objective, ZERO_COEFFICIENTS, HELD_OUT_SEEDS
    )
    held_out_trained, _ = _evaluate_numpy(
        physics, objective, held_out_coefficients, HELD_OUT_SEEDS
    )

    coefficient_distances = np.linalg.norm(
        coefficients[:, None, :] - coefficients[None, :, :], axis=2
    )
    preload_distances = np.sqrt(
        np.mean(
            (trained_preload[:, None, :] - trained_preload[None, :, :]) ** 2,
            axis=2,
        )
    )
    train_improvement = (fixed_objective - trained_objective) / fixed_objective
    held_out_change = (held_out_fixed - held_out_trained) / held_out_fixed
    first_step_gate = len(history) >= 2
    final_gate = (
        len(history) == MAX_ITERATIONS + 1
        and trained_objective < fixed_objective
    )
    passed = (
        gradient_gate
        and endpoint_gate
        and backward_gate
        and baseline_gate
        and first_step_gate
        and final_gate
    )

    print("## Summary")
    print("device: cpu")
    print("dtype: float64")
    print("architecture: 6 -> 16 -> 16 -> 5, tanh")
    print(f"q_fixed: {BASE_Q.tolist()}")
    print(f"training_seeds: {TRAINING_SEEDS.tolist()}")
    print(f"held_out_seeds: {HELD_OUT_SEEDS.tolist()}")
    print("## Results")
    print(f"J_fixed: {fixed_objective:.16g}")
    print(f"J_initial: {initial_objective:.16g}")
    print(f"J_final: {trained_objective:.16g}")
    print(f"train_relative_improvement: {train_improvement:.12g}")
    print(f"J_held_out_fixed: {held_out_fixed:.16g}")
    print(f"J_held_out_trained: {held_out_trained:.16g}")
    print(f"held_out_relative_improvement: {held_out_change:.12g}")
    print(f"accepted_learning_rate: {accepted_learning_rate}")
    print(f"trained_coefficients: {coefficients.tolist()}")
    print(
        "coefficient_column_std: "
        f"{np.std(coefficients, axis=0).tolist()}"
    )
    print(
        "max_pairwise_coefficient_distance: "
        f"{float(np.max(coefficient_distances)):.12g}"
    )
    print(f"N_mean: {trained_control_stats['mean']:.12g}")
    print(f"N_min: {trained_control_stats['min']:.12g}")
    print(f"N_max: {trained_control_stats['max']:.12g}")
    print(
        "N_seed_ranges: "
        f"{np.stack((trained_preload.min(axis=1), trained_preload.max(axis=1)), axis=1).tolist()}"
    )
    print(
        "max_pairwise_N_rms_difference: "
        f"{float(np.max(preload_distances)):.12g}"
    )
    print(
        "fixed_stick_to_slip: "
        f"{np.asarray(fixed_response['stick_to_slip']).tolist()}"
    )
    print(
        "fixed_slip_to_stick: "
        f"{np.asarray(fixed_response['slip_to_stick']).tolist()}"
    )
    print(
        "trained_stick_to_slip: "
        f"{np.asarray(trained_response['stick_to_slip']).tolist()}"
    )
    print(
        "trained_slip_to_stick: "
        f"{np.asarray(trained_response['slip_to_stick']).tolist()}"
    )
    print("## Gradient")
    for epsilon, gradient in gradient_scales.items():
        print(
            f"epsilon={epsilon:g} norm={np.linalg.norm(gradient):.12g} "
            f"mean_gradient={np.mean(gradient, axis=0).tolist()}"
        )
    print(f"direction_cosines: {list(direction_cosines)}")
    print(f"initial_total_gradient_norm: {total_initial_gradient_norm:.12g}")
    print(f"initial_final_layer_gradient_norm: {final_initial_gradient_norm:.12g}")
    print("physics_apply_jvp_vjp: finite" if endpoint_gate else "physics_apply_jvp_vjp: failed")
    print("torch_backward: finite_nonzero" if backward_gate else "torch_backward: failed")
    print("## Training")
    for entry in history:
        control = entry["control"]
        print(
            f"iter={entry['iteration']} J={entry['objective']:.16g} "
            f"gradient_norm={entry['gradient_norm']:.12g} "
            f"N_mean={control['mean']:.12g} "
            f"N_min={control['min']:.12g} N_max={control['max']:.12g}"
        )
    print("## Runtime")
    print(f"training_seconds: {training_seconds:.9g}")
    print("## PASS" if passed else "## FAIL")
    if passed:
        for path in _plot_results(
            fixed_response, trained_response, trained_preload, history
        ):
            print(f"figure: {path}")
    else:
        print(f"gradient_gate: {gradient_gate}")
        print(f"endpoint_gate: {endpoint_gate}")
        print(f"backward_gate: {backward_gate}")
        print(f"baseline_gate: {baseline_gate}")
        print(f"first_step_gate: {first_step_gate}")
        print(f"final_gate: {final_gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
