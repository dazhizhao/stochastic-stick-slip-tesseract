"""Run the Stage H3 controller ablation and generalization study."""

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
    HELD_OUT_SEEDS,
    TRAINING_SEEDS,
    forcing_descriptor_batch,
    preload_history,
)


PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
OBJECTIVE_API = ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/stage_h3"
BASE_Q = np.array([BASELINE_DAMPING, 0.04], dtype=np.float64)
ZERO_COEFFICIENTS = np.zeros((8, 5), dtype=np.float64)
H3_TEST_SEEDS = np.arange(201, 233, dtype=np.int64)
CRN_SEED_PAIRS = tuple(
    (
        np.arange(301 + 16 * repeat, 309 + 16 * repeat, dtype=np.int64),
        np.arange(309 + 16 * repeat, 317 + 16 * repeat, dtype=np.int64),
    )
    for repeat in range(5)
)
LEARNING_RATE = 0.01
MAX_ITERATIONS = 20
FD_EPSILON = 0.02
H2_REFERENCE_OBJECTIVE = 0.005769207113962147


def create_tesseracts():
    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    objective = Tesseract.from_tesseract_api(OBJECTIVE_API)
    return physics, objective


def shared_coefficient_batch(shared):
    """Broadcast one five-vector to the eight-seed Tesseract interface."""
    if isinstance(shared, torch.Tensor):
        return shared.unsqueeze(0).expand(8, -1)
    return np.broadcast_to(np.asarray(shared, dtype=np.float64), (8, 5)).copy()


def apply_torch_pipeline(physics, objective, coefficients, seeds):
    response = apply_tesseract(
        physics,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds},
    )
    return apply_tesseract(
        objective, {"seed_losses": response["seed_losses"]}
    )["objective"]


def evaluate_numpy_batch(physics, objective, coefficients, seeds):
    response = physics.apply(
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds}
    )
    result = objective.apply({"seed_losses": response["seed_losses"]})
    return np.asarray(response["seed_losses"], dtype=np.float64), float(
        result["objective"]
    )


def controller_coefficients(controller, seeds):
    descriptors = torch.from_numpy(forcing_descriptor_batch(seeds))
    with torch.no_grad():
        return controller(descriptors).detach().cpu().numpy()


def evaluate_seed_set(physics, objective, seeds, coefficient_function):
    seeds = np.asarray(seeds, dtype=np.int64)
    if len(seeds) % 8 != 0:
        raise ValueError("seed sets must be evaluated in batches of eight")
    losses = []
    coefficients = []
    for start in range(0, len(seeds), 8):
        batch = seeds[start : start + 8]
        batch_coefficients = np.asarray(
            coefficient_function(batch), dtype=np.float64
        )
        batch_losses, _ = evaluate_numpy_batch(
            physics, objective, batch_coefficients, batch
        )
        losses.append(batch_losses)
        coefficients.append(batch_coefficients)
    return np.concatenate(losses), np.concatenate(coefficients)


def train_shared(physics, objective):
    shared = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = torch.optim.Adam([shared], lr=LEARNING_RATE)
    _, initial_objective = evaluate_numpy_batch(
        physics, objective, ZERO_COEFFICIENTS, TRAINING_SEEDS
    )
    history = [
        {
            "iteration": 0,
            "objective": initial_objective,
            "gradient_norm": np.nan,
        }
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = apply_torch_pipeline(
            physics,
            objective,
            shared_coefficient_batch(shared),
            TRAINING_SEEDS,
        )
        loss.backward()
        gradient_norm = float(torch.linalg.vector_norm(shared.grad))
        optimizer.step()
        _, hard_objective = evaluate_numpy_batch(
            physics,
            objective,
            shared_coefficient_batch(shared.detach().cpu().numpy()),
            TRAINING_SEEDS,
        )
        history.append(
            {
                "iteration": iteration,
                "objective": hard_objective,
                "gradient_norm": gradient_norm,
            }
        )
    return shared.detach().cpu().numpy(), history


def train_mlp(physics, objective):
    controller = build_controller()
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    initial_coefficients = controller_coefficients(controller, TRAINING_SEEDS)
    _, initial_objective = evaluate_numpy_batch(
        physics, objective, initial_coefficients, TRAINING_SEEDS
    )
    history = [
        {
            "iteration": 0,
            "objective": initial_objective,
            "gradient_norm": np.nan,
        }
    ]
    descriptors = torch.from_numpy(forcing_descriptor_batch(TRAINING_SEEDS))
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = apply_torch_pipeline(
            physics, objective, controller(descriptors), TRAINING_SEEDS
        )
        loss.backward()
        gradient_norm = parameter_gradient_norm(controller.parameters())
        optimizer.step()
        coefficients = controller_coefficients(controller, TRAINING_SEEDS)
        _, hard_objective = evaluate_numpy_batch(
            physics, objective, coefficients, TRAINING_SEEDS
        )
        history.append(
            {
                "iteration": iteration,
                "objective": hard_objective,
                "gradient_norm": gradient_norm,
            }
        )
    return controller, history


def shared_fd_gradient(physics, objective, plus_seeds, minus_seeds):
    gradient = np.empty(5, dtype=np.float64)
    for column in range(5):
        plus = ZERO_COEFFICIENTS.copy()
        minus = ZERO_COEFFICIENTS.copy()
        plus[:, column] += FD_EPSILON
        minus[:, column] -= FD_EPSILON
        _, plus_objective = evaluate_numpy_batch(
            physics, objective, plus, plus_seeds
        )
        _, minus_objective = evaluate_numpy_batch(
            physics, objective, minus, minus_seeds
        )
        gradient[column] = (
            plus_objective - minus_objective
        ) / (2.0 * FD_EPSILON)
    return gradient


def cosine_to_mean(gradients):
    gradients = np.asarray(gradients, dtype=np.float64)
    mean_gradient = np.mean(gradients, axis=0)
    mean_norm = np.linalg.norm(mean_gradient)
    cosines = np.array(
        [
            np.dot(gradient, mean_gradient)
            / (np.linalg.norm(gradient) * mean_norm)
            for gradient in gradients
        ],
        dtype=np.float64,
    )
    return mean_gradient, cosines


def relative_comparisons(losses):
    fixed = losses["Fixed"]
    shared = losses["Shared"]
    mlp = losses["MLP"]
    return {
        "Shared vs Fixed": (fixed - shared) / fixed,
        "MLP vs Fixed": (fixed - mlp) / fixed,
        "MLP vs Shared": (shared - mlp) / shared,
    }


def configure_figure_style():
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
        spine.set_visible(True)
        spine.set_color("#1A1A1A")
        spine.set_linewidth(0.99)


def plot_results(test_losses, train_means, test_means):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_figure_style()
    comparisons = relative_comparisons(test_losses)

    generalization_path = OUTPUT_DIRECTORY / "per_seed_generalization.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1)
    axis.plot(
        H3_TEST_SEEDS,
        100.0 * comparisons["Shared vs Fixed"],
        color="#3B6FB6",
        marker="o",
        markersize=4.2,
        linewidth=1.4,
        label="Shared Fourier",
    )
    axis.plot(
        H3_TEST_SEEDS,
        100.0 * comparisons["MLP vs Fixed"],
        color="#D17A22",
        marker="s",
        markersize=4.0,
        linewidth=1.4,
        label="MLP Fourier",
    )
    axis.set_xlabel("Test seed")
    axis.set_ylabel("Relative improvement (%)")
    axis.set_xticks(H3_TEST_SEEDS[::4])
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(generalization_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    objective_path = OUTPUT_DIRECTORY / "train_test_objectives.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    positions = np.arange(2, dtype=np.float64)
    width = 0.23
    colors = {"Fixed": "#737373", "Shared": "#3B6FB6", "MLP": "#D17A22"}
    for offset, method in zip((-1, 0, 1), ("Fixed", "Shared", "MLP")):
        axis.bar(
            positions + offset * width,
            [train_means[method], test_means[method]],
            width=width,
            color=colors[method],
            edgecolor="#1A1A1A",
            linewidth=0.7,
            label=method,
        )
    axis.set_xticks(positions, ("Training", "32-seed test"))
    axis.set_ylabel("Mean objective")
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(objective_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return generalization_path, objective_path


def main() -> int:
    torch.set_default_dtype(torch.float64)
    physics, objective = create_tesseracts()
    start = time.perf_counter()

    zero_preload = np.asarray(
        preload_history(0.04, shared_coefficient_batch(np.zeros(5)))
    )
    zero_gate = np.array_equal(
        zero_preload, np.full((8, zero_preload.shape[1]), 0.04)
    )

    shared, shared_history = train_shared(physics, objective)
    controller, mlp_history = train_mlp(physics, objective)

    fixed_function = lambda seeds: np.zeros((len(seeds), 5), dtype=np.float64)
    shared_function = lambda seeds: np.broadcast_to(
        shared, (len(seeds), 5)
    ).copy()
    mlp_function = lambda seeds: controller_coefficients(controller, seeds)
    coefficient_functions = {
        "Fixed": fixed_function,
        "Shared": shared_function,
        "MLP": mlp_function,
    }

    train_losses = {}
    test_losses = {}
    train_coefficients = {}
    test_coefficients = {}
    for method, coefficient_function in coefficient_functions.items():
        train_losses[method], train_coefficients[method] = evaluate_seed_set(
            physics, objective, TRAINING_SEEDS, coefficient_function
        )
        test_losses[method], test_coefficients[method] = evaluate_seed_set(
            physics, objective, H3_TEST_SEEDS, coefficient_function
        )

    crn_gradients = []
    independent_gradients = []
    for plus_seeds, independent_minus_seeds in CRN_SEED_PAIRS:
        crn_gradients.append(
            shared_fd_gradient(physics, objective, plus_seeds, plus_seeds)
        )
        independent_gradients.append(
            shared_fd_gradient(
                physics, objective, plus_seeds, independent_minus_seeds
            )
        )
    crn_gradients = np.asarray(crn_gradients)
    independent_gradients = np.asarray(independent_gradients)
    crn_mean, crn_cosines = cosine_to_mean(crn_gradients)
    independent_mean, independent_cosines = cosine_to_mean(
        independent_gradients
    )

    train_means = {
        method: float(np.mean(values)) for method, values in train_losses.items()
    }
    test_means = {
        method: float(np.mean(values)) for method, values in test_losses.items()
    }
    train_relative = relative_comparisons(train_losses)
    test_relative = relative_comparisons(test_losses)
    train_aggregate = {
        "Shared vs Fixed": (train_means["Fixed"] - train_means["Shared"])
        / train_means["Fixed"],
        "MLP vs Fixed": (train_means["Fixed"] - train_means["MLP"])
        / train_means["Fixed"],
        "MLP vs Shared": (train_means["Shared"] - train_means["MLP"])
        / train_means["Shared"],
    }
    test_aggregate = {
        "Shared vs Fixed": (test_means["Fixed"] - test_means["Shared"])
        / test_means["Fixed"],
        "MLP vs Fixed": (test_means["Fixed"] - test_means["MLP"])
        / test_means["Fixed"],
        "MLP vs Shared": (test_means["Shared"] - test_means["MLP"])
        / test_means["Shared"],
    }
    win_counts = {
        comparison: int(np.count_nonzero(values > 0.0))
        for comparison, values in test_relative.items()
    }
    test_medians = {
        comparison: float(np.median(values))
        for comparison, values in test_relative.items()
    }

    parameter_count = sum(parameter.numel() for parameter in controller.parameters())
    seed_sets_disjoint = (
        set(H3_TEST_SEEDS).isdisjoint(TRAINING_SEEDS)
        and set(H3_TEST_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    )
    all_values = [
        *train_losses.values(),
        *test_losses.values(),
        crn_gradients,
        independent_gradients,
        crn_cosines,
        independent_cosines,
        shared,
        train_coefficients["MLP"],
        test_coefficients["MLP"],
    ]
    finite_gate = all(np.all(np.isfinite(value)) for value in all_values)
    nonzero_gradient_gate = all(
        np.linalg.norm(gradient) > 0.0
        for gradient in (*crn_gradients, *independent_gradients)
    )
    training_gate = (
        len(shared_history) == MAX_ITERATIONS + 1
        and len(mlp_history) == MAX_ITERATIONS + 1
    )
    passed = (
        zero_gate
        and finite_gate
        and nonzero_gradient_gate
        and training_gate
        and seed_sets_disjoint
        and parameter_count == 469
    )

    shared_preload = np.asarray(
        preload_history(0.04, train_coefficients["Shared"])
    )
    mlp_train_preload = np.asarray(
        preload_history(0.04, train_coefficients["MLP"])
    )
    mlp_test_preload = np.asarray(
        preload_history(0.04, test_coefficients["MLP"])
    )
    h2_reference_delta = train_means["MLP"] - H2_REFERENCE_OBJECTIVE

    print("## Summary")
    print(f"training_seeds: {TRAINING_SEEDS.tolist()}")
    print(f"h3_test_seeds: {H3_TEST_SEEDS.tolist()}")
    print(f"zero_shared_is_constant_0.04: {zero_gate}")
    print("## Training")
    for name, history in (("Shared", shared_history), ("MLP", mlp_history)):
        for entry in history:
            print(
                f"{name} iter={entry['iteration']} "
                f"J={entry['objective']:.16g} "
                f"gradient_norm={entry['gradient_norm']:.12g}"
            )
    print("## Objectives")
    for split, losses, means in (
        ("train", train_losses, train_means),
        ("test", test_losses, test_means),
    ):
        for method in ("Fixed", "Shared", "MLP"):
            print(f"{split}_{method}_mean: {means[method]:.16g}")
            print(f"{split}_{method}_seed_losses: {losses[method].tolist()}")
    print(f"train_aggregate_relative_improvement: {train_aggregate}")
    print(f"test_aggregate_relative_improvement: {test_aggregate}")
    print(
        "test_per_seed_relative_improvement: "
        f"{dict((key, value.tolist()) for key, value in test_relative.items())}"
    )
    print(f"test_win_counts: {win_counts}")
    print(f"test_median_relative_improvement: {test_medians}")
    print("## Controllers")
    print(f"shared_coefficients: {shared.tolist()}")
    print(f"mlp_train_coefficients: {train_coefficients['MLP'].tolist()}")
    print(f"mlp_test_coefficients: {test_coefficients['MLP'].tolist()}")
    print(
        "shared_N_range: "
        f"{[float(shared_preload.min()), float(shared_preload.max())]}"
    )
    print(
        "mlp_train_N_ranges: "
        f"{np.stack((mlp_train_preload.min(axis=1), mlp_train_preload.max(axis=1)), axis=1).tolist()}"
    )
    print(
        "mlp_test_N_ranges: "
        f"{np.stack((mlp_test_preload.min(axis=1), mlp_test_preload.max(axis=1)), axis=1).tolist()}"
    )
    print(f"mlp_parameter_count: {parameter_count}")
    print(f"hypothetical_centered_weight_fd_evaluations: {2 * parameter_count}")
    print("coefficient_fd_batch_forwards: 10")
    print(f"h2_reference_objective: {H2_REFERENCE_OBJECTIVE:.16g}")
    print(f"h2_reference_delta: {h2_reference_delta:.16g}")
    print("## CRN comparison")
    for repeat, (crn, independent) in enumerate(
        zip(crn_gradients, independent_gradients), start=1
    ):
        print(
            f"repeat={repeat} crn_gradient={crn.tolist()} "
            f"crn_norm={np.linalg.norm(crn):.12g} "
            f"crn_cosine={crn_cosines[repeat - 1]:.12g}"
        )
        print(
            f"repeat={repeat} independent_gradient={independent.tolist()} "
            f"independent_norm={np.linalg.norm(independent):.12g} "
            f"independent_cosine={independent_cosines[repeat - 1]:.12g}"
        )
    print(f"crn_mean_gradient: {crn_mean.tolist()}")
    print(f"crn_mean_cosine: {float(np.mean(crn_cosines)):.12g}")
    print(f"independent_mean_gradient: {independent_mean.tolist()}")
    print(
        "independent_mean_cosine: "
        f"{float(np.mean(independent_cosines)):.12g}"
    )
    print("## Runtime")
    print(f"total_seconds: {time.perf_counter() - start:.9g}")
    print("## PASS" if passed else "## FAIL")
    if passed:
        for path in plot_results(test_losses, train_means, test_means):
            print(f"figure: {path}")
    else:
        print(f"finite_gate: {finite_gate}")
        print(f"nonzero_gradient_gate: {nonzero_gradient_gate}")
        print(f"training_gate: {training_gate}")
        print(f"seed_sets_disjoint: {seed_sets_disjoint}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
