"""Compare fixed CRN Gaussian smoothing with S1 AD and coordinate FD."""

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
from stochastic_stick_slip.gaussian_smoothing import (
    GAUSSIAN_DIRECTION_SEED,
    GAUSSIAN_SIGMA,
    NUM_GAUSSIAN_DIRECTIONS,
    fixed_gaussian_directions,
    gaussian_smoothing_coefficient_gradient,
)
from stochastic_stick_slip.model import forcing_descriptor_batch
from stochastic_stick_slip.stochastic_event import evaluate_with_inputs


OUTPUT_DIRECTORY = ROOT / "outputs/gaussian_smoothing_probe"
FIGURE_PATH = OUTPUT_DIRECTORY / "gradient_and_optimization.png"

AD_COLOR = "#D9782D"
FD_COLOR = "#2F6DA4"
GAUSSIAN_COLOR = "#6F5AA8"
REFERENCE_COLOR = "#555B63"
FRAME_COLOR = "#20242A"


def _gaussian_coefficient_cotangent(controller, theta, inputs, directions):
    gradients = []
    start = time.perf_counter()
    coefficient_batches = s1._coefficient_batches(controller, theta, inputs)
    for batch_index, (coefficients, (_, forcing, uniforms)) in enumerate(
        zip(coefficient_batches, inputs, strict=True)
    ):
        per_seed_gradient = gaussian_smoothing_coefficient_gradient(
            s1.BASE_Q,
            coefficients,
            forcing,
            uniforms,
            directions[batch_index],
            sigma=GAUSSIAN_SIGMA,
        )
        gradients.append(per_seed_gradient / len(s1.H4_TRAINING_SEEDS))
    return np.concatenate(gradients), time.perf_counter() - start


def _gaussian_event_sensitivity(controller, theta, inputs, directions):
    weak_sensitive = np.zeros(len(s1.H4_TRAINING_SEEDS), dtype=bool)
    slip_sensitive = np.zeros(len(s1.H4_TRAINING_SEEDS), dtype=bool)
    coefficient_batches = s1._coefficient_batches(controller, theta, inputs)
    offset = 0
    for batch_index, (coefficients, (_, forcing, uniforms)) in enumerate(
        zip(coefficient_batches, inputs, strict=True)
    ):
        batch_weak = np.zeros(s1.BATCH_SIZE, dtype=bool)
        batch_slip = np.zeros(s1.BATCH_SIZE, dtype=bool)
        for direction in directions[batch_index]:
            plus_result = evaluate_with_inputs(
                s1.BASE_Q,
                coefficients + GAUSSIAN_SIGMA * direction,
                forcing,
                uniforms,
            )
            minus_result = evaluate_with_inputs(
                s1.BASE_Q,
                coefficients - GAUSSIAN_SIGMA * direction,
                forcing,
                uniforms,
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
        weak_sensitive[offset : offset + s1.BATCH_SIZE] = batch_weak
        slip_sensitive[offset : offset + s1.BATCH_SIZE] = batch_slip
        offset += s1.BATCH_SIZE
    return int(np.count_nonzero(weak_sensitive)), int(np.count_nonzero(slip_sensitive))


def _train_gaussian(controller, theta0, inputs, directions):
    theta = torch.nn.Parameter(
        torch.as_tensor(theta0, dtype=torch.float64).clone()
    )
    optimizer = torch.optim.Adam([theta], lr=s1.LEARNING_RATE)
    objective_history = np.empty(s1.NUM_ITERATIONS + 1, dtype=np.float64)
    gradient_history = np.empty(s1.NUM_ITERATIONS, dtype=np.float64)
    objective_history[0] = s1._hard_objective(controller, theta0, inputs)

    start = time.perf_counter()
    for iteration in range(1, s1.NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (seeds, forcing, uniforms) in enumerate(inputs):
            coefficients = apply_tesseract(
                controller,
                {
                    "theta": theta,
                    "descriptors": forcing_descriptor_batch(seeds),
                },
            )["coeffs"]
            per_seed_gradient = gaussian_smoothing_coefficient_gradient(
                s1.BASE_Q,
                coefficients.detach().numpy(),
                forcing,
                uniforms,
                directions[batch_index],
                sigma=GAUSSIAN_SIGMA,
            )
            coefficients.backward(
                torch.as_tensor(
                    per_seed_gradient / len(s1.H4_TRAINING_SEEDS),
                    dtype=torch.float64,
                )
            )
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError("Gaussian theta gradient is invalid")
        gradient_history[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        objective_history[iteration] = s1._hard_objective(
            controller, theta.detach().numpy(), inputs
        )
        print(
            f"method=gaussian iteration={iteration:02d} "
            f"hard_objective={objective_history[iteration]:.16g} "
            f"gradient_norm={gradient_history[iteration - 1]:.9g}",
            flush=True,
        )
    return objective_history, gradient_history, time.perf_counter() - start


def _plot(
    ad_gradient,
    fd_gradient,
    gaussian_gradient,
    ad_history,
    fd_history,
    gaussian_history,
    gaussian_ad_cosine,
    gaussian_fd_cosine,
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
    gaussian_flat = gaussian_gradient.ravel()
    ad_flat = ad_gradient.ravel()
    fd_flat = fd_gradient.ravel()
    limits = np.array(
        [
            min(gaussian_flat.min(), ad_flat.min(), fd_flat.min()),
            max(gaussian_flat.max(), ad_flat.max(), fd_flat.max()),
        ]
    )
    padding = 0.08 * max(limits[1] - limits[0], np.finfo(float).eps)
    limits += np.array([-padding, padding])
    axes[0].scatter(
        gaussian_flat,
        ad_flat,
        s=14,
        marker="o",
        color=AD_COLOR,
        alpha=0.68,
        edgecolors="none",
        rasterized=True,
        label=f"Branchwise AD ({gaussian_ad_cosine:.3f})",
    )
    axes[0].scatter(
        gaussian_flat,
        fd_flat,
        s=14,
        marker="s",
        color=FD_COLOR,
        alpha=0.64,
        edgecolors="none",
        rasterized=True,
        label=f"Coordinate FD ({gaussian_fd_cosine:.3f})",
    )
    axes[0].plot(
        limits, limits, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--"
    )
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="Gaussian gradient",
        ylabel="Comparison gradient",
    )
    axes[0].legend(loc="best")

    iterations = np.arange(s1.NUM_ITERATIONS + 1)
    axes[1].plot(
        iterations,
        ad_history,
        color=AD_COLOR,
        linewidth=1.5,
        label="Branchwise AD",
    )
    axes[1].plot(
        iterations,
        fd_history,
        color=FD_COLOR,
        linewidth=1.5,
        label="Coordinate FD",
    )
    axes[1].plot(
        iterations,
        gaussian_history,
        color=GAUSSIAN_COLOR,
        linewidth=1.7,
        label="Gaussian smoothing",
    )
    axes[1].set(
        xlim=(0, s1.NUM_ITERATIONS),
        xticks=np.arange(0, s1.NUM_ITERATIONS + 1, 5),
        xlabel="Iteration",
        ylabel="Hard objective",
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
    inputs = s1._batch_inputs()
    directions = fixed_gaussian_directions()
    theta0 = s1._initial_theta()
    objective0, full_objective0, _ = s1._full_forward_scaling(
        controller, theta0, inputs
    )

    ad_objective0, ad_gradient, ad_gradient_seconds = (
        s1._ad_coefficient_cotangent(controller, theta0, inputs)
    )
    fd_gradient, fd_gradient_seconds = s1._fd_coefficient_cotangent(
        controller, theta0, inputs
    )
    gaussian_gradient, gaussian_gradient_seconds = (
        _gaussian_coefficient_cotangent(
            controller, theta0, inputs, directions
        )
    )
    if not np.isclose(ad_objective0, objective0, rtol=1e-12, atol=1e-14):
        raise ValueError("AD and production objectives differ")
    for gradient in (ad_gradient, fd_gradient, gaussian_gradient):
        if gradient.shape != (32, 5) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("coefficient gradient is invalid")
        if np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError("coefficient gradient is zero")

    ad_theta = s1._theta_gradient(controller, theta0, ad_gradient, inputs)
    fd_theta = s1._theta_gradient(controller, theta0, fd_gradient, inputs)
    gaussian_theta = s1._theta_gradient(
        controller, theta0, gaussian_gradient, inputs
    )
    coefficient_cosines = {
        "gaussian_ad": s1._cosine(gaussian_gradient, ad_gradient),
        "gaussian_fd": s1._cosine(gaussian_gradient, fd_gradient),
        "fd_ad": s1._cosine(fd_gradient, ad_gradient),
    }
    theta_cosines = {
        "gaussian_ad": s1._cosine(gaussian_theta, ad_theta),
        "gaussian_fd": s1._cosine(gaussian_theta, fd_theta),
        "fd_ad": s1._cosine(fd_theta, ad_theta),
    }
    weak_sensitive, slip_sensitive = _gaussian_event_sensitivity(
        controller, theta0, inputs, directions
    )

    ad_history, ad_gradient_history, ad_training_seconds = s1._train(
        controller, theta0, inputs, "ad"
    )
    fd_history, fd_gradient_history, fd_training_seconds = s1._train(
        controller, theta0, inputs, "fd"
    )
    gaussian_history, gaussian_gradient_history, gaussian_training_seconds = (
        _train_gaussian(controller, theta0, inputs, directions)
    )
    if not all(
        np.all(np.isfinite(values))
        for values in (
            ad_history,
            fd_history,
            gaussian_history,
            ad_gradient_history,
            fd_gradient_history,
            gaussian_gradient_history,
        )
    ):
        raise FloatingPointError("training produced non-finite values")

    figure_path = _plot(
        ad_gradient,
        fd_gradient,
        gaussian_gradient,
        ad_history,
        fd_history,
        gaussian_history,
        coefficient_cosines["gaussian_ad"],
        coefficient_cosines["gaussian_fd"],
    )
    improvements = {
        "ad": (ad_history[0] - ad_history[-1]) / ad_history[0],
        "fd": (fd_history[0] - fd_history[-1]) / fd_history[0],
        "gaussian": (
            gaussian_history[0] - gaussian_history[-1]
        ) / gaussian_history[0],
    }

    print("\n=== Fixed Gaussian estimator ===")
    print(f"direction_seed={GAUSSIAN_DIRECTION_SEED}")
    print(f"directions_per_batch={NUM_GAUSSIAN_DIRECTIONS}")
    print(f"sigma={GAUSSIAN_SIGMA:.16g}")
    print(f"direction_shape={list(directions.shape)}")
    print(f"batch_objective={objective0:.16g}")
    print(f"full_32_seed_objective={full_objective0:.16g}")
    print(f"weak_state_sensitive_seeds={weak_sensitive}/32")
    print(f"stick_slip_sensitive_seeds={slip_sensitive}/32")

    print("\n=== Coefficient-space gradients ===")
    print(f"branchwise_ad_norm={np.linalg.norm(ad_gradient):.16g}")
    print(f"coordinate_fd_norm={np.linalg.norm(fd_gradient):.16g}")
    print(f"gaussian_norm={np.linalg.norm(gaussian_gradient):.16g}")
    for name, value in coefficient_cosines.items():
        print(f"coefficient_cosine_{name}={value:.16g}")

    print("\n=== Theta-space gradients ===")
    print(f"branchwise_ad_norm={np.linalg.norm(ad_theta):.16g}")
    print(f"coordinate_fd_norm={np.linalg.norm(fd_theta):.16g}")
    print(f"gaussian_norm={np.linalg.norm(gaussian_theta):.16g}")
    for name, value in theta_cosines.items():
        print(f"theta_cosine_{name}={value:.16g}")

    print("\n=== Original hard stochastic objective ===")
    for iteration in (0, 1, 5, 10, 20):
        print(
            f"iteration={iteration:02d} "
            f"branchwise_ad={ad_history[iteration]:.16g} "
            f"coordinate_fd={fd_history[iteration]:.16g} "
            f"gaussian={gaussian_history[iteration]:.16g}"
        )
    print(f"branchwise_ad_improvement={improvements['ad']:.16g}")
    print(f"coordinate_fd_improvement={improvements['fd']:.16g}")
    print(f"gaussian_improvement={improvements['gaussian']:.16g}")

    print("\n=== Runtime ===")
    print(f"branchwise_ad_gradient_seconds={ad_gradient_seconds:.6f}")
    print(f"coordinate_fd_gradient_seconds={fd_gradient_seconds:.6f}")
    print(f"gaussian_gradient_seconds={gaussian_gradient_seconds:.6f}")
    print(f"branchwise_ad_training_seconds={ad_training_seconds:.6f}")
    print(f"coordinate_fd_training_seconds={fd_training_seconds:.6f}")
    print(f"gaussian_training_seconds={gaussian_training_seconds:.6f}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
