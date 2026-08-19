"""Compare branchwise AD with a Bernoulli score-function correction."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

import scripts.run_stochastic_event_probe as s1
from stochastic_stick_slip.model import forcing_descriptor_batch
from stochastic_stick_slip.score_function import (
    EVALUATION_STREAM,
    ITERATION_ZERO_STREAM,
    REFERENCE_STREAM,
    TRAINING_STREAM,
    branchwise_condition_gradients,
    condition_mean_losses,
    friction_uniform_realization_bank,
    mc_centered_fd_condition_gradients,
    score_function_condition_gradients,
)
from stochastic_stick_slip.stochastic_event import stochastic_inputs


OUTPUT_DIRECTORY = ROOT / "outputs/score_function_probe"
FIGURE_PATH = OUTPUT_DIRECTORY / "score_function_gradient.png"

NUM_INITIAL_REALIZATIONS = 8
NUM_REFERENCE_REALIZATIONS = 16
NUM_TRAINING_REALIZATIONS = 4
NUM_EVALUATION_REALIZATIONS = 8
REFERENCE_EPSILON = 0.01

BRANCH_COLOR = "#D9782D"
COMBINED_COLOR = "#6F5AA8"
REFERENCE_COLOR = "#555B63"
FRAME_COLOR = "#20242A"


def _event_inputs(num_realizations, stream_id, iteration=0):
    inputs = []
    for seeds in s1._seed_batches():
        forcing = np.asarray(stochastic_inputs(seeds)[0])
        uniforms = friction_uniform_realization_bank(
            seeds,
            num_realizations,
            stream_id,
            iteration=iteration,
        )
        inputs.append((seeds, forcing, uniforms))
    return inputs


def _coefficient_batches(controller, theta, inputs):
    return [
        s1._controller_coefficients(controller, theta, seeds)
        for seeds, _, _ in inputs
    ]


def _hard_expectation(controller, theta, inputs) -> float:
    condition_losses = []
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        condition_losses.append(
            condition_mean_losses(
                s1.BASE_Q, coefficients, forcing, uniforms
            )
        )
    return float(np.mean(np.concatenate(condition_losses)))


def _full_objective_scaling(controller, theta, inputs):
    coefficient_batches = _coefficient_batches(controller, theta, inputs)
    batched_objective = _hard_expectation(controller, theta, inputs)
    full_losses = condition_mean_losses(
        s1.BASE_Q,
        np.concatenate(coefficient_batches),
        np.concatenate([item[1] for item in inputs]),
        np.concatenate([item[2] for item in inputs]),
    )
    full_objective = float(np.mean(full_losses))
    if not np.isclose(
        batched_objective, full_objective, rtol=1e-12, atol=1e-14
    ):
        raise ValueError("four-batch and full 32-condition objectives differ")
    return batched_objective, full_objective


def _branchwise_coefficient_cotangent(controller, theta, inputs):
    objectives = []
    gradients = []
    start = time.perf_counter()
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        condition_objectives, condition_gradients = (
            branchwise_condition_gradients(
                s1.BASE_Q, coefficients, forcing, uniforms
            )
        )
        objectives.append(condition_objectives)
        gradients.append(condition_gradients)
    unweighted = np.concatenate(gradients)
    return (
        float(np.mean(np.concatenate(objectives))),
        unweighted / len(s1.H4_TRAINING_SEEDS),
        time.perf_counter() - start,
    )


def _combined_coefficient_cotangent(controller, theta, inputs):
    objectives = []
    branchwise_gradients = []
    score_gradients = []
    start = time.perf_counter()
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        condition_objectives, condition_branchwise = (
            branchwise_condition_gradients(
                s1.BASE_Q, coefficients, forcing, uniforms
            )
        )
        _, condition_score = score_function_condition_gradients(
            s1.BASE_Q, coefficients, forcing, uniforms
        )
        objectives.append(condition_objectives)
        branchwise_gradients.append(condition_branchwise)
        score_gradients.append(condition_score)
    branchwise = np.concatenate(branchwise_gradients)
    score = np.concatenate(score_gradients)
    scale = len(s1.H4_TRAINING_SEEDS)
    return (
        float(np.mean(np.concatenate(objectives))),
        branchwise / scale,
        score / scale,
        (branchwise + score) / scale,
        time.perf_counter() - start,
    )


def _mc_reference_coefficient_cotangent(controller, theta, inputs):
    gradients = []
    start = time.perf_counter()
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        gradients.append(
            mc_centered_fd_condition_gradients(
                s1.BASE_Q,
                coefficients,
                forcing,
                uniforms,
                epsilon=REFERENCE_EPSILON,
            )
        )
    return (
        np.concatenate(gradients) / len(s1.H4_TRAINING_SEEDS),
        time.perf_counter() - start,
    )


def _method_condition_gradient(method, coefficients, forcing, uniforms):
    _, branchwise = branchwise_condition_gradients(
        s1.BASE_Q, coefficients, forcing, uniforms
    )
    if method == "branchwise":
        return branchwise
    if method == "combined":
        _, score = score_function_condition_gradients(
            s1.BASE_Q, coefficients, forcing, uniforms
        )
        return branchwise + score
    raise ValueError(f"unknown gradient method: {method}")


def _train(controller, theta0, method, evaluation_inputs):
    theta = torch.nn.Parameter(
        torch.as_tensor(theta0, dtype=torch.float64).clone()
    )
    optimizer = torch.optim.Adam([theta], lr=s1.LEARNING_RATE)
    objective_history = np.empty(s1.NUM_ITERATIONS + 1, dtype=np.float64)
    gradient_history = np.empty(s1.NUM_ITERATIONS, dtype=np.float64)
    objective_history[0] = _hard_expectation(controller, theta0, evaluation_inputs)

    start = time.perf_counter()
    for iteration in range(1, s1.NUM_ITERATIONS + 1):
        training_inputs = _event_inputs(
            NUM_TRAINING_REALIZATIONS,
            TRAINING_STREAM,
            iteration=iteration,
        )
        optimizer.zero_grad(set_to_none=True)
        for seeds, forcing, uniforms in training_inputs:
            coefficients = apply_tesseract(
                controller,
                {
                    "theta": theta,
                    "descriptors": forcing_descriptor_batch(seeds),
                },
            )["coeffs"]
            condition_gradient = _method_condition_gradient(
                method,
                coefficients.detach().numpy(),
                forcing,
                uniforms,
            )
            coefficients.backward(
                torch.as_tensor(
                    condition_gradient / len(s1.H4_TRAINING_SEEDS),
                    dtype=torch.float64,
                )
            )
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError(f"{method} theta gradient is invalid")
        gradient_history[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        objective_history[iteration] = _hard_expectation(
            controller, theta.detach().numpy(), evaluation_inputs
        )
        print(
            f"method={method} iteration={iteration:02d} "
            f"evaluation_objective={objective_history[iteration]:.16g} "
            f"gradient_norm={gradient_history[iteration - 1]:.9g}",
            flush=True,
        )
    return objective_history, gradient_history, time.perf_counter() - start


def _plot(
    reference_gradient,
    branchwise_gradient,
    combined_gradient,
    branchwise_history,
    combined_history,
    branchwise_reference_cosine,
    combined_reference_cosine,
):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    reference_flat = reference_gradient.ravel()
    branchwise_flat = branchwise_gradient.ravel()
    combined_flat = combined_gradient.ravel()
    limits = np.array(
        [
            min(reference_flat.min(), branchwise_flat.min(), combined_flat.min()),
            max(reference_flat.max(), branchwise_flat.max(), combined_flat.max()),
        ]
    )
    padding = 0.08 * max(limits[1] - limits[0], np.finfo(float).eps)
    limits += np.array([-padding, padding])
    axes[0].scatter(
        reference_flat,
        branchwise_flat,
        s=14,
        marker="o",
        color=BRANCH_COLOR,
        alpha=0.68,
        edgecolors="none",
        rasterized=True,
        label=f"Branchwise ({branchwise_reference_cosine:.3f})",
    )
    axes[0].scatter(
        reference_flat,
        combined_flat,
        s=14,
        marker="s",
        color=COMBINED_COLOR,
        alpha=0.64,
        edgecolors="none",
        rasterized=True,
        label=f"AD + score ({combined_reference_cosine:.3f})",
    )
    axes[0].plot(
        limits, limits, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--"
    )
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="MC finite-difference gradient",
        ylabel="Comparison gradient",
    )
    axes[0].ticklabel_format(
        axis="both", style="sci", scilimits=(-3, 3), useMathText=True
    )
    axes[0].legend(loc="best")

    iterations = np.arange(s1.NUM_ITERATIONS + 1)
    axes[1].plot(
        iterations,
        branchwise_history,
        color=BRANCH_COLOR,
        linewidth=1.6,
        label="Branchwise AD",
    )
    axes[1].plot(
        iterations,
        combined_history,
        color=COMBINED_COLOR,
        linewidth=1.7,
        label="AD + score",
    )
    axes[1].set(
        xlim=(0, s1.NUM_ITERATIONS),
        xticks=np.arange(0, s1.NUM_ITERATIONS + 1, 5),
        xlabel="Iteration",
        ylabel="Evaluation objective",
    )
    axes[1].legend(loc="best")
    for label, axis in zip(("a", "b"), axes, strict=True):
        s1._style_axis(axis)
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=FRAME_COLOR,
        )
    figure.subplots_adjust(
        left=0.10, right=0.985, bottom=0.19, top=0.95, wspace=0.34
    )
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return FIGURE_PATH


def main():
    controller = Tesseract.from_tesseract_api(s1.CONTROLLER_API)
    theta0 = s1._initial_theta()
    initial_inputs = _event_inputs(
        NUM_INITIAL_REALIZATIONS, ITERATION_ZERO_STREAM
    )
    reference_inputs = _event_inputs(
        NUM_REFERENCE_REALIZATIONS, REFERENCE_STREAM
    )
    evaluation_inputs = _event_inputs(
        NUM_EVALUATION_REALIZATIONS, EVALUATION_STREAM
    )
    coefficient_batches = _coefficient_batches(controller, theta0, initial_inputs)
    if not np.array_equal(
        np.concatenate(coefficient_batches), np.zeros((32, 5), dtype=np.float64)
    ):
        raise ValueError("theta0 does not produce zero coefficients")

    batch_objective, full_objective = _full_objective_scaling(
        controller, theta0, initial_inputs
    )
    branch_objective, branchwise, branchwise_seconds = (
        _branchwise_coefficient_cotangent(controller, theta0, initial_inputs)
    )
    (
        combined_objective,
        combined_branchwise,
        score,
        combined,
        combined_seconds,
    ) = _combined_coefficient_cotangent(controller, theta0, initial_inputs)
    for value in (branch_objective, combined_objective):
        if not np.isclose(value, batch_objective, rtol=1e-12, atol=1e-14):
            raise ValueError("gradient objective and hard objective differ")
    if not np.allclose(
        combined_branchwise, branchwise, rtol=1e-12, atol=1e-14
    ):
        raise ValueError("matched-bank branchwise gradients differ")

    reference, reference_seconds = _mc_reference_coefficient_cotangent(
        controller, theta0, reference_inputs
    )
    for gradient in (branchwise, score, combined, reference):
        if gradient.shape != (32, 5) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("coefficient gradient is invalid")
        if np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError("coefficient gradient is zero")

    branchwise_theta = s1._theta_gradient(
        controller, theta0, branchwise, initial_inputs
    )
    score_theta = s1._theta_gradient(controller, theta0, score, initial_inputs)
    combined_theta = s1._theta_gradient(
        controller, theta0, combined, initial_inputs
    )
    reference_theta = s1._theta_gradient(
        controller, theta0, reference, reference_inputs
    )
    coefficient_cosines = {
        "branchwise_combined": s1._cosine(branchwise, combined),
        "branchwise_reference": s1._cosine(branchwise, reference),
        "combined_reference": s1._cosine(combined, reference),
    }
    theta_cosines = {
        "branchwise_combined": s1._cosine(branchwise_theta, combined_theta),
        "branchwise_reference": s1._cosine(branchwise_theta, reference_theta),
        "combined_reference": s1._cosine(combined_theta, reference_theta),
    }

    branchwise_history, branchwise_gradient_history, branchwise_training_seconds = (
        _train(controller, theta0, "branchwise", evaluation_inputs)
    )
    combined_history, combined_gradient_history, combined_training_seconds = (
        _train(controller, theta0, "combined", evaluation_inputs)
    )
    for values in (
        branchwise_history,
        combined_history,
        branchwise_gradient_history,
        combined_gradient_history,
    ):
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("training produced non-finite values")

    figure_path = _plot(
        reference,
        branchwise,
        combined,
        branchwise_history,
        combined_history,
        coefficient_cosines["branchwise_reference"],
        coefficient_cosines["combined_reference"],
    )
    branchwise_improvement = (
        branchwise_history[0] - branchwise_history[-1]
    ) / branchwise_history[0]
    combined_improvement = (
        combined_history[0] - combined_history[-1]
    ) / combined_history[0]

    print("\n=== Estimator configuration ===")
    print(f"initial_realizations={NUM_INITIAL_REALIZATIONS}")
    print(f"reference_realizations={NUM_REFERENCE_REALIZATIONS}")
    print(f"training_realizations={NUM_TRAINING_REALIZATIONS}")
    print(f"evaluation_realizations={NUM_EVALUATION_REALIZATIONS}")
    print(f"reference_epsilon={REFERENCE_EPSILON:.16g}")
    print(f"batch_objective={batch_objective:.16g}")
    print(f"full_32_condition_objective={full_objective:.16g}")
    print("controller_cotangent_scale=1/32")

    print("\n=== Coefficient-space gradients ===")
    print(f"branchwise_norm={np.linalg.norm(branchwise):.16g}")
    print(f"score_component_norm={np.linalg.norm(score):.16g}")
    print(f"combined_norm={np.linalg.norm(combined):.16g}")
    print(f"reference_norm={np.linalg.norm(reference):.16g}")
    print(
        "branchwise_combined_relative_difference="
        f"{s1._relative_difference(combined, branchwise):.16g}"
    )
    for name, value in coefficient_cosines.items():
        print(f"coefficient_cosine_{name}={value:.16g}")

    print("\n=== Theta-space gradients ===")
    print(f"branchwise_norm={np.linalg.norm(branchwise_theta):.16g}")
    print(f"score_component_norm={np.linalg.norm(score_theta):.16g}")
    print(f"combined_norm={np.linalg.norm(combined_theta):.16g}")
    print(f"reference_norm={np.linalg.norm(reference_theta):.16g}")
    print(
        "branchwise_combined_relative_difference="
        f"{s1._relative_difference(combined_theta, branchwise_theta):.16g}"
    )
    for name, value in theta_cosines.items():
        print(f"theta_cosine_{name}={value:.16g}")

    print("\n=== Fixed evaluation-bank hard objective ===")
    for iteration in (0, 1, 5, 10, 20):
        print(
            f"iteration={iteration:02d} "
            f"branchwise={branchwise_history[iteration]:.16g} "
            f"ad_plus_score={combined_history[iteration]:.16g}"
        )
    print(f"branchwise_improvement={branchwise_improvement:.16g}")
    print(f"ad_plus_score_improvement={combined_improvement:.16g}")

    print("\n=== Runtime ===")
    print(f"branchwise_iteration0_gradient_seconds={branchwise_seconds:.6f}")
    print(f"ad_plus_score_iteration0_gradient_seconds={combined_seconds:.6f}")
    print(f"mc_reference_seconds={reference_seconds:.6f}")
    print(f"branchwise_training_seconds={branchwise_training_seconds:.6f}")
    print(f"ad_plus_score_training_seconds={combined_training_seconds:.6f}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
