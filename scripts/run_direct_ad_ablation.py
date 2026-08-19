"""Compare naive branchwise JAX AD with the production CRN-FD VJP."""

from __future__ import annotations

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
import torch

from scripts.run_showcase import (
    BASE_Q,
    LEARNING_RATE as SCIENTIFIC_LEARNING_RATE,
)
from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import (
    build_controller,
    flatten_controller_parameters,
    functional_controller,
)
from stochastic_stick_slip.model import (
    COEFFICIENT_FD_EPSILON,
    forcing_batch_for_system,
    forcing_descriptor_batch,
)
from stochastic_stick_slip.showcase import (
    SYSTEM,
    _SIMULATE_BATCH,
    crn_fd_coefficient_jacobian,
    evaluate_controlled_batch,
)


OUTPUT_DIRECTORY = ROOT / "outputs/direct_ad_ablation"
SCIENTIFIC_HISTORY = ROOT / "outputs/showcase/training_history.npz"
FIGURE_PATH = OUTPUT_DIRECTORY / "direct_ad_vs_crn_fd.png"
LEARNING_RATE = 0.01
NUM_ITERATIONS = 20
BATCH_SIZE = 8
NUM_BATCHES = len(H4_TRAINING_SEEDS) // BATCH_SIZE

CRN_COLOR = "#2F6DA4"
DIRECT_AD_COLOR = "#D9782D"
REFERENCE_COLOR = "#555B63"
FRAME_COLOR = "#20242A"


def _seed_batches(seeds: np.ndarray = H4_TRAINING_SEEDS):
    if len(seeds) % BATCH_SIZE:
        raise ValueError("seed count must be divisible by eight")
    for start in range(0, len(seeds), BATCH_SIZE):
        yield seeds[start : start + BATCH_SIZE]


def _initial_theta() -> torch.Tensor:
    return flatten_controller_parameters(build_controller()).detach().clone()


def _controller_coefficients(theta: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    controller = build_controller()
    with torch.no_grad():
        coefficients = functional_controller(
            controller,
            torch.as_tensor(theta, dtype=torch.float64),
            torch.as_tensor(
                forcing_descriptor_batch(seeds), dtype=torch.float64
            ),
        )
    return coefficients.numpy()


def _hard_objective(coefficients: jax.Array, forcing: jax.Array) -> jax.Array:
    displacement = _SIMULATE_BATCH(
        jnp.asarray(BASE_Q, dtype=jnp.float64), coefficients, forcing
    )[0]
    return jnp.mean(displacement**2)


_DIRECT_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_hard_objective))


def direct_ad_batch_objective_and_gradient(
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Differentiate the exact hard batch program with naive branchwise AD."""
    value, gradient = _DIRECT_VALUE_AND_GRAD(
        jnp.asarray(coefficients, dtype=jnp.float64),
        forcing_batch_for_system(seeds, SYSTEM),
    )
    return float(value), np.asarray(gradient)


def production_batch_objective(
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> float:
    """Evaluate the production hard forward for one seed batch."""
    result = evaluate_controlled_batch(BASE_Q, coefficients, seeds)
    return float(np.mean(np.asarray(result.losses)))


def production_mean_objective(theta: np.ndarray) -> float:
    batch_objectives = []
    for seeds in _seed_batches():
        coefficients = _controller_coefficients(theta, seeds)
        batch_objectives.append(production_batch_objective(coefficients, seeds))
    return float(np.mean(batch_objectives))


def production_full_objective(theta: np.ndarray) -> float:
    coefficients = _controller_coefficients(theta, H4_TRAINING_SEEDS)
    return production_batch_objective(coefficients, H4_TRAINING_SEEDS)


def _direct_coefficient_cotangent(theta: np.ndarray) -> tuple[float, np.ndarray]:
    objectives = []
    gradients = []
    for seeds in _seed_batches():
        coefficients = _controller_coefficients(theta, seeds)
        objective, batch_mean_gradient = direct_ad_batch_objective_and_gradient(
            coefficients, seeds
        )
        objectives.append(objective)
        gradients.append(batch_mean_gradient / NUM_BATCHES)
    return float(np.mean(objectives)), np.concatenate(gradients, axis=0)


def _crn_coefficient_cotangent(theta: np.ndarray) -> np.ndarray:
    gradients = []
    for seeds in _seed_batches():
        coefficients = _controller_coefficients(theta, seeds)
        per_seed_gradient = crn_fd_coefficient_jacobian(
            BASE_Q,
            coefficients,
            seeds,
            epsilon=COEFFICIENT_FD_EPSILON,
        )
        gradients.append(per_seed_gradient / len(H4_TRAINING_SEEDS))
    return np.concatenate(gradients, axis=0)


def _theta_gradient(theta: np.ndarray, coefficient_cotangent: np.ndarray) -> np.ndarray:
    controller = build_controller()
    theta_tensor = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    scalar = torch.zeros((), dtype=torch.float64)
    offset = 0
    for seeds in _seed_batches():
        coefficients = functional_controller(
            controller,
            theta_tensor,
            torch.as_tensor(
                forcing_descriptor_batch(seeds), dtype=torch.float64
            ),
        )
        batch_cotangent = torch.as_tensor(
            coefficient_cotangent[offset : offset + BATCH_SIZE],
            dtype=torch.float64,
        )
        scalar = scalar + torch.sum(coefficients * batch_cotangent)
        offset += BATCH_SIZE
    return torch.autograd.grad(scalar, theta_tensor)[0].detach().numpy()


def _switching_sensitive_seed_count(theta: np.ndarray) -> int:
    sensitive = np.zeros(len(H4_TRAINING_SEEDS), dtype=bool)
    offset = 0
    for seeds in _seed_batches():
        coefficients = _controller_coefficients(theta, seeds)
        batch_sensitive = np.zeros(BATCH_SIZE, dtype=bool)
        for column in range(coefficients.shape[1]):
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[:, column] += COEFFICIENT_FD_EPSILON
            minus[:, column] -= COEFFICIENT_FD_EPSILON
            plus_slip = np.asarray(
                evaluate_controlled_batch(BASE_Q, plus, seeds).slip
            )
            minus_slip = np.asarray(
                evaluate_controlled_batch(BASE_Q, minus, seeds).slip
            )
            batch_sensitive |= np.any(plus_slip != minus_slip, axis=(1, 2))
        sensitive[offset : offset + BATCH_SIZE] = batch_sensitive
        offset += BATCH_SIZE
    return int(np.count_nonzero(sensitive))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left.ravel(), right.ravel()) / denominator)


def _relative_difference(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _load_and_validate_crn_history(theta0: np.ndarray, objective0: float) -> np.ndarray:
    if not SCIENTIFIC_HISTORY.exists():
        raise FileNotFoundError(
            "outputs/showcase/training_history.npz is required; run the "
            "scientific showcase first"
        )
    with np.load(SCIENTIFIC_HISTORY) as data:
        theta_history = np.asarray(data["theta_history"])
        objective_history = np.asarray(data["objective_history"])
    expected_seeds = np.concatenate(
        (np.array([11, 23, 37, 41, 53, 67, 79, 97]), np.arange(201, 225))
    )
    if theta_history.shape != (501, 469) or objective_history.shape != (501,):
        raise ValueError("scientific history has unexpected shapes")
    if not np.array_equal(H4_TRAINING_SEEDS, expected_seeds):
        raise ValueError("scientific training seeds no longer match H4")
    if not np.array_equal(BASE_Q, np.array([0.2, 0.04])):
        raise ValueError("scientific q no longer matches the frozen benchmark")
    if SCIENTIFIC_LEARNING_RATE != LEARNING_RATE:
        raise ValueError("scientific learning rate no longer matches the ablation")
    if not np.array_equal(theta_history[0], theta0):
        raise ValueError("scientific history does not start from theta0")
    if not np.isclose(objective_history[0], objective0, rtol=1e-12, atol=1e-14):
        raise ValueError("scientific history J0 does not match the hard forward")
    return objective_history[: NUM_ITERATIONS + 1]


def _train_direct_ad(theta0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    controller = build_controller()
    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float64).clone())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    objective_history = np.empty(NUM_ITERATIONS + 1, dtype=np.float64)
    gradient_history = np.empty(NUM_ITERATIONS, dtype=np.float64)
    objective_history[0] = production_mean_objective(theta0)

    start = time.perf_counter()
    for iteration in range(1, NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        for seeds in _seed_batches():
            coefficients = functional_controller(
                controller,
                theta,
                torch.as_tensor(
                    forcing_descriptor_batch(seeds), dtype=torch.float64
                ),
            )
            _, batch_mean_gradient = direct_ad_batch_objective_and_gradient(
                coefficients.detach().numpy(), seeds
            )
            coefficients.backward(
                torch.as_tensor(
                    batch_mean_gradient / NUM_BATCHES, dtype=torch.float64
                )
            )
        gradient = theta.grad.detach().numpy()
        if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
            raise FloatingPointError("Direct-AD theta gradient is invalid")
        gradient_history[iteration - 1] = np.linalg.norm(gradient)
        optimizer.step()
        objective_history[iteration] = production_mean_objective(
            theta.detach().numpy()
        )
        print(
            f"direct_ad_iteration={iteration:02d} "
            f"hard_objective={objective_history[iteration]:.16g} "
            f"gradient_norm={gradient_history[iteration - 1]:.9g}",
            flush=True,
        )
    return objective_history, gradient_history, time.perf_counter() - start


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


def _plot_ablation(
    crn_coefficients: np.ndarray,
    direct_coefficients: np.ndarray,
    crn_history: np.ndarray,
    direct_history: np.ndarray,
    coefficient_cosine: float,
) -> Path:
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

    crn_flat = crn_coefficients.ravel()
    direct_flat = direct_coefficients.ravel()
    limits = np.array(
        [min(crn_flat.min(), direct_flat.min()), max(crn_flat.max(), direct_flat.max())]
    )
    padding = 0.08 * max(limits[1] - limits[0], np.finfo(float).eps)
    limits += np.array([-padding, padding])
    axes[0].scatter(
        crn_flat,
        direct_flat,
        s=13,
        color=DIRECT_AD_COLOR,
        alpha=0.78,
        edgecolors="none",
        rasterized=True,
    )
    axes[0].plot(limits, limits, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--")
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="CRN-FD coefficient gradient",
        ylabel="Direct-AD coefficient gradient",
    )
    axes[0].text(
        0.04,
        0.96,
        f"cosine = {coefficient_cosine:.4f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color=FRAME_COLOR,
    )

    iterations = np.arange(NUM_ITERATIONS + 1)
    axes[1].plot(
        iterations,
        crn_history,
        color=CRN_COLOR,
        linewidth=1.6,
        label="CRN-FD",
    )
    axes[1].plot(
        iterations,
        direct_history,
        color=DIRECT_AD_COLOR,
        linewidth=1.6,
        label="Direct AD",
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
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.95, wspace=0.34)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return FIGURE_PATH


def main() -> None:
    theta0 = _initial_theta().numpy()
    production_objective0 = production_mean_objective(theta0)
    full_production_objective0 = production_full_objective(theta0)
    if not np.isclose(
        production_objective0,
        full_production_objective0,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("four-batch and 32-seed production objectives differ")
    direct_objective0, direct_coefficient_gradient = (
        _direct_coefficient_cotangent(theta0)
    )
    if not np.isclose(
        direct_objective0, production_objective0, rtol=1e-12, atol=1e-14
    ):
        raise ValueError("Direct-AD and production objective scaling differ")

    crn_history = _load_and_validate_crn_history(theta0, production_objective0)
    crn_coefficient_gradient = _crn_coefficient_cotangent(theta0)
    crn_theta_gradient = _theta_gradient(theta0, crn_coefficient_gradient)
    direct_theta_gradient = _theta_gradient(theta0, direct_coefficient_gradient)
    sensitive_count = _switching_sensitive_seed_count(theta0)

    coefficient_cosine = _cosine(
        crn_coefficient_gradient, direct_coefficient_gradient
    )
    theta_cosine = _cosine(crn_theta_gradient, direct_theta_gradient)
    coefficient_relative_difference = _relative_difference(
        direct_coefficient_gradient, crn_coefficient_gradient
    )
    theta_relative_difference = _relative_difference(
        direct_theta_gradient, crn_theta_gradient
    )

    direct_history, direct_gradient_history, training_seconds = _train_direct_ad(
        theta0
    )
    if not np.all(np.isfinite(direct_history)) or not np.all(
        np.isfinite(direct_gradient_history)
    ):
        raise FloatingPointError("Direct-AD training produced non-finite values")
    figure_path = _plot_ablation(
        crn_coefficient_gradient,
        direct_coefficient_gradient,
        crn_history,
        direct_history,
        coefficient_cosine,
    )

    milestones = np.array([0, 1, 5, 10, 20])
    crn_improvement = (crn_history[0] - crn_history[-1]) / crn_history[0]
    direct_improvement = (
        direct_history[0] - direct_history[-1]
    ) / direct_history[0]
    print("\n=== Gradient comparison at theta0 ===")
    print(f"production_objective={production_objective0:.16g}")
    print(f"full_32_seed_objective={full_production_objective0:.16g}")
    print(f"direct_ad_objective={direct_objective0:.16g}")
    print(f"coefficient_crn_norm={np.linalg.norm(crn_coefficient_gradient):.16g}")
    print(f"coefficient_direct_ad_norm={np.linalg.norm(direct_coefficient_gradient):.16g}")
    print(f"coefficient_cosine={coefficient_cosine:.16g}")
    print(f"coefficient_relative_difference={coefficient_relative_difference:.16g}")
    print(f"theta_crn_norm={np.linalg.norm(crn_theta_gradient):.16g}")
    print(f"theta_direct_ad_norm={np.linalg.norm(direct_theta_gradient):.16g}")
    print(f"theta_cosine={theta_cosine:.16g}")
    print(f"theta_relative_difference={theta_relative_difference:.16g}")
    print(f"switching_sensitive_seeds={sensitive_count}/32")
    print("\n=== Hard-objective history ===")
    for iteration in milestones:
        print(
            f"iteration={iteration:02d} "
            f"crn_fd={crn_history[iteration]:.16g} "
            f"direct_ad={direct_history[iteration]:.16g}"
        )
    print(f"crn_fd_improvement_0_to_20={crn_improvement:.16g}")
    print(f"direct_ad_improvement_0_to_20={direct_improvement:.16g}")
    print(f"final_direct_minus_crn={direct_history[-1] - crn_history[-1]:.16g}")
    print(f"direct_ad_training_seconds={training_seconds:.6f}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
