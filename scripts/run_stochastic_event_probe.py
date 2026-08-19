"""Probe parameter-dependent stochastic friction events with AD and FD."""

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

from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import (
    build_controller,
    flatten_controller_parameters,
)
from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    forcing_descriptor_batch,
)
from stochastic_stick_slip.stochastic_event import (
    centered_fd_coefficient_jacobian,
    direct_ad_batch_objective_and_gradient,
    evaluate_with_inputs,
    stochastic_inputs,
)


CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/stochastic_event_probe"
FIGURE_PATH = OUTPUT_DIRECTORY / "ad_vs_fd.png"
BASE_Q = np.array([0.2, 0.04], dtype=np.float64)
LEARNING_RATE = 0.01
NUM_ITERATIONS = 20
BATCH_SIZE = 8
NUM_BATCHES = 4

FD_COLOR = "#2F6DA4"
AD_COLOR = "#D9782D"
REFERENCE_COLOR = "#555B63"
FRAME_COLOR = "#20242A"


def _seed_batches():
    for start in range(0, len(H4_TRAINING_SEEDS), BATCH_SIZE):
        yield H4_TRAINING_SEEDS[start : start + BATCH_SIZE]


def _batch_inputs():
    return [
        (seeds, *stochastic_inputs(seeds)) for seeds in _seed_batches()
    ]


def _initial_theta() -> np.ndarray:
    return flatten_controller_parameters(build_controller()).detach().numpy()


def _controller_coefficients(controller, theta, seeds):
    return np.asarray(
        controller.apply(
            {
                "theta": np.asarray(theta, dtype=np.float64),
                "descriptors": forcing_descriptor_batch(seeds),
            }
        )["coeffs"]
    )


def _coefficient_batches(controller, theta, inputs):
    return [
        _controller_coefficients(controller, theta, seeds)
        for seeds, _, _ in inputs
    ]


def _hard_objective(controller, theta, inputs) -> float:
    objectives = []
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        result = evaluate_with_inputs(
            BASE_Q, coefficients, forcing, uniforms
        )
        objectives.append(float(np.mean(np.asarray(result.losses))))
    return float(np.mean(objectives))


def _full_forward_scaling(controller, theta, inputs):
    coefficient_batches = _coefficient_batches(controller, theta, inputs)
    batch_objective = _hard_objective(controller, theta, inputs)
    full_result = evaluate_with_inputs(
        BASE_Q,
        np.concatenate(coefficient_batches),
        np.concatenate([np.asarray(item[1]) for item in inputs]),
        np.concatenate([np.asarray(item[2]) for item in inputs]),
    )
    full_objective = float(np.mean(np.asarray(full_result.losses)))
    if not np.isclose(batch_objective, full_objective, rtol=1e-12, atol=1e-14):
        raise ValueError("four-batch and full 32-seed objectives differ")
    return batch_objective, full_objective, full_result


def _fd_coefficient_cotangent(controller, theta, inputs):
    gradients = []
    start = time.perf_counter()
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        per_seed_gradient = centered_fd_coefficient_jacobian(
            BASE_Q,
            coefficients,
            forcing,
            uniforms,
            epsilon=COEFFICIENT_FD_EPSILON,
        )
        gradients.append(per_seed_gradient / len(H4_TRAINING_SEEDS))
    return np.concatenate(gradients), time.perf_counter() - start


def _ad_coefficient_cotangent(controller, theta, inputs):
    objectives = []
    gradients = []
    start = time.perf_counter()
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        objective, batch_mean_gradient = direct_ad_batch_objective_and_gradient(
            BASE_Q, coefficients, forcing, uniforms
        )
        objectives.append(objective)
        gradients.append(batch_mean_gradient / NUM_BATCHES)
    return (
        float(np.mean(objectives)),
        np.concatenate(gradients),
        time.perf_counter() - start,
    )


def _theta_gradient(controller, theta, coefficient_cotangent, inputs):
    theta_parameter = torch.nn.Parameter(
        torch.as_tensor(theta, dtype=torch.float64).clone()
    )
    offset = 0
    for seeds, _, _ in inputs:
        coefficients = apply_tesseract(
            controller,
            {
                "theta": theta_parameter,
                "descriptors": forcing_descriptor_batch(seeds),
            },
        )["coeffs"]
        coefficients.backward(
            torch.as_tensor(
                coefficient_cotangent[offset : offset + BATCH_SIZE],
                dtype=torch.float64,
            )
        )
        offset += BATCH_SIZE
    return theta_parameter.grad.detach().numpy()


def _event_sensitivity(controller, theta, inputs):
    weak_sensitive = np.zeros(len(H4_TRAINING_SEEDS), dtype=bool)
    slip_sensitive = np.zeros(len(H4_TRAINING_SEEDS), dtype=bool)
    offset = 0
    for coefficients, (_, forcing, uniforms) in zip(
        _coefficient_batches(controller, theta, inputs), inputs, strict=True
    ):
        batch_weak = np.zeros(BATCH_SIZE, dtype=bool)
        batch_slip = np.zeros(BATCH_SIZE, dtype=bool)
        for column in range(coefficients.shape[1]):
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[:, column] += COEFFICIENT_FD_EPSILON
            minus[:, column] -= COEFFICIENT_FD_EPSILON
            plus_result = evaluate_with_inputs(
                BASE_Q, plus, forcing, uniforms
            )
            minus_result = evaluate_with_inputs(
                BASE_Q, minus, forcing, uniforms
            )
            batch_weak |= np.any(
                np.asarray(plus_result.weak_state)
                != np.asarray(minus_result.weak_state),
                axis=(1, 2),
            )
            batch_slip |= np.any(
                np.asarray(plus_result.slip) != np.asarray(minus_result.slip),
                axis=(1, 2),
            )
        weak_sensitive[offset : offset + BATCH_SIZE] = batch_weak
        slip_sensitive[offset : offset + BATCH_SIZE] = batch_slip
        offset += BATCH_SIZE
    return int(np.count_nonzero(weak_sensitive)), int(np.count_nonzero(slip_sensitive))


def _train(controller, theta0, inputs, method):
    theta = torch.nn.Parameter(
        torch.as_tensor(theta0, dtype=torch.float64).clone()
    )
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    objective_history = np.empty(NUM_ITERATIONS + 1, dtype=np.float64)
    gradient_history = np.empty(NUM_ITERATIONS, dtype=np.float64)
    objective_history[0] = _hard_objective(controller, theta0, inputs)

    start = time.perf_counter()
    for iteration in range(1, NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        for seeds, forcing, uniforms in inputs:
            coefficients = apply_tesseract(
                controller,
                {
                    "theta": theta,
                    "descriptors": forcing_descriptor_batch(seeds),
                },
            )["coeffs"]
            coefficients_array = coefficients.detach().numpy()
            if method == "fd":
                per_seed_gradient = centered_fd_coefficient_jacobian(
                    BASE_Q,
                    coefficients_array,
                    forcing,
                    uniforms,
                    epsilon=COEFFICIENT_FD_EPSILON,
                )
                cotangent = per_seed_gradient / len(H4_TRAINING_SEEDS)
            elif method == "ad":
                _, batch_mean_gradient = direct_ad_batch_objective_and_gradient(
                    BASE_Q, coefficients_array, forcing, uniforms
                )
                cotangent = batch_mean_gradient / NUM_BATCHES
            else:
                raise ValueError(f"unknown gradient method: {method}")
            coefficients.backward(torch.as_tensor(cotangent, dtype=torch.float64))
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError(f"{method} theta gradient is invalid")
        gradient_history[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        objective_history[iteration] = _hard_objective(
            controller, theta.detach().numpy(), inputs
        )
        print(
            f"method={method} iteration={iteration:02d} "
            f"hard_objective={objective_history[iteration]:.16g} "
            f"gradient_norm={gradient_history[iteration - 1]:.9g}",
            flush=True,
        )
    return objective_history, gradient_history, time.perf_counter() - start


def _cosine(left, right):
    return float(
        np.dot(left.ravel(), right.ravel())
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def _relative_difference(candidate, reference):
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _style_axis(axis):
    axis.set_facecolor("white")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.88)
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        color=FRAME_COLOR,
        labelcolor=FRAME_COLOR,
        width=0.8,
        length=3.5,
    )


def _plot(fd_gradient, ad_gradient, fd_history, ad_history, cosine):
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
    fd_flat = fd_gradient.ravel()
    ad_flat = ad_gradient.ravel()
    limits = np.array(
        [min(fd_flat.min(), ad_flat.min()), max(fd_flat.max(), ad_flat.max())]
    )
    padding = 0.08 * max(limits[1] - limits[0], np.finfo(float).eps)
    limits += np.array([-padding, padding])
    axes[0].scatter(
        fd_flat,
        ad_flat,
        s=13,
        color=AD_COLOR,
        alpha=0.78,
        edgecolors="none",
        rasterized=True,
    )
    axes[0].plot(
        limits, limits, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--"
    )
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="Centered-FD gradient",
        ylabel="Branchwise-AD gradient",
    )
    axes[0].text(
        0.04,
        0.96,
        f"cosine = {cosine:.4f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color=FRAME_COLOR,
    )

    iterations = np.arange(NUM_ITERATIONS + 1)
    axes[1].plot(
        iterations,
        fd_history,
        color=FD_COLOR,
        linewidth=1.6,
        label="Centered FD",
    )
    axes[1].plot(
        iterations,
        ad_history,
        color=AD_COLOR,
        linewidth=1.6,
        label="Branchwise AD",
    )
    axes[1].set(
        xlim=(0, NUM_ITERATIONS),
        xticks=np.arange(0, NUM_ITERATIONS + 1, 5),
        xlabel="Iteration",
        ylabel="Hard objective",
    )
    axes[1].legend(loc="best")
    for label, axis in zip(("a", "b"), axes, strict=True):
        _style_axis(axis)
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
    controller = Tesseract.from_tesseract_api(CONTROLLER_API)
    inputs = _batch_inputs()
    theta0 = _initial_theta()
    coefficients0 = _coefficient_batches(controller, theta0, inputs)
    if not np.array_equal(
        np.concatenate(coefficients0), np.zeros((32, 5), dtype=np.float64)
    ):
        raise ValueError("theta0 does not produce zero coefficients")

    forward_start = time.perf_counter()
    objective0, full_objective0, baseline = _full_forward_scaling(
        controller, theta0, inputs
    )
    forward_seconds = time.perf_counter() - forward_start
    fd_gradient, fd_gradient_seconds = _fd_coefficient_cotangent(
        controller, theta0, inputs
    )
    ad_objective0, ad_gradient, ad_gradient_seconds = (
        _ad_coefficient_cotangent(controller, theta0, inputs)
    )
    if not np.isclose(ad_objective0, objective0, rtol=1e-12, atol=1e-14):
        raise ValueError("AD and production objectives differ")
    fd_theta = _theta_gradient(controller, theta0, fd_gradient, inputs)
    ad_theta = _theta_gradient(controller, theta0, ad_gradient, inputs)
    weak_sensitive, slip_sensitive = _event_sensitivity(
        controller, theta0, inputs
    )

    coefficient_cosine = _cosine(fd_gradient, ad_gradient)
    theta_cosine = _cosine(fd_theta, ad_theta)
    coefficient_relative_difference = _relative_difference(
        ad_gradient, fd_gradient
    )
    theta_relative_difference = _relative_difference(ad_theta, fd_theta)

    fd_history, fd_gradient_history, fd_training_seconds = _train(
        controller, theta0, inputs, "fd"
    )
    ad_history, ad_gradient_history, ad_training_seconds = _train(
        controller, theta0, inputs, "ad"
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (
            fd_history,
            ad_history,
            fd_gradient_history,
            ad_gradient_history,
        )
    ):
        raise FloatingPointError("training produced non-finite values")
    figure_path = _plot(
        fd_gradient,
        ad_gradient,
        fd_history,
        ad_history,
        coefficient_cosine,
    )

    weak_selections = np.sum(np.asarray(baseline.weak_selections), axis=0)
    strong_selections = np.sum(np.asarray(baseline.strong_selections), axis=0)
    renewals = np.sum(np.asarray(baseline.renewals), axis=0)
    stick_to_slip = np.sum(np.asarray(baseline.stick_to_slip), axis=0)
    slip_to_stick = np.sum(np.asarray(baseline.slip_to_stick), axis=0)
    displacement = np.asarray(baseline.displacement)
    milestones = (0, 1, 5, 10, 20)
    fd_improvement = (fd_history[0] - fd_history[-1]) / fd_history[0]
    ad_improvement = (ad_history[0] - ad_history[-1]) / ad_history[0]

    print("\n=== Stochastic-event forward ===")
    print(f"batch_objective={objective0:.16g}")
    print(f"full_32_seed_objective={full_objective0:.16g}")
    print(f"displacement_range=[{displacement.min():.16g},{displacement.max():.16g}]")
    print(f"stick_to_slip_A_B={stick_to_slip.tolist()}")
    print(f"slip_to_stick_A_B={slip_to_stick.tolist()}")
    print(f"weak_selections_A_B={weak_selections.tolist()}")
    print(f"strong_selections_A_B={strong_selections.tolist()}")
    print(f"renewals_A_B={renewals.tolist()}")
    print(f"weak_state_sensitive_seeds={weak_sensitive}/32")
    print(f"stick_slip_sensitive_seeds={slip_sensitive}/32")

    print("\n=== Iteration-0 gradient comparison ===")
    print(f"coefficient_fd_norm={np.linalg.norm(fd_gradient):.16g}")
    print(f"coefficient_ad_norm={np.linalg.norm(ad_gradient):.16g}")
    print(f"coefficient_cosine={coefficient_cosine:.16g}")
    print(f"coefficient_relative_difference={coefficient_relative_difference:.16g}")
    print(f"theta_fd_norm={np.linalg.norm(fd_theta):.16g}")
    print(f"theta_ad_norm={np.linalg.norm(ad_theta):.16g}")
    print(f"theta_cosine={theta_cosine:.16g}")
    print(f"theta_relative_difference={theta_relative_difference:.16g}")

    print("\n=== Hard stochastic objective ===")
    for iteration in milestones:
        print(
            f"iteration={iteration:02d} "
            f"centered_fd={fd_history[iteration]:.16g} "
            f"branchwise_ad={ad_history[iteration]:.16g}"
        )
    print(f"centered_fd_improvement_0_to_20={fd_improvement:.16g}")
    print(f"branchwise_ad_improvement_0_to_20={ad_improvement:.16g}")
    print(f"final_ad_minus_fd={ad_history[-1] - fd_history[-1]:.16g}")

    print("\n=== Runtime ===")
    print(f"forward_32_seed_seconds={forward_seconds:.6f}")
    print(f"centered_fd_gradient_seconds={fd_gradient_seconds:.6f}")
    print(f"branchwise_ad_gradient_seconds={ad_gradient_seconds:.6f}")
    print(f"centered_fd_training_seconds={fd_training_seconds:.6f}")
    print(f"branchwise_ad_training_seconds={ad_training_seconds:.6f}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
