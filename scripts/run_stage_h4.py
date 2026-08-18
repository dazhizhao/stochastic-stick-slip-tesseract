"""Run the Stage H4 larger-training-set generalization study."""

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.run_stage_h3 import (
    BASE_Q,
    ZERO_COEFFICIENTS,
    apply_torch_pipeline,
    configure_figure_style,
    controller_coefficients,
    create_tesseracts,
    evaluate_numpy_batch,
    evaluate_seed_set,
    relative_comparisons,
    shared_coefficient_batch,
    style_axis,
)
from stochastic_stick_slip.controller import (
    build_controller,
    parameter_gradient_norm,
)
from stochastic_stick_slip.model import (
    TRAINING_SEEDS,
    forcing_descriptor_batch,
    preload_history,
)


OUTPUT_DIRECTORY = ROOT / "outputs/stage_h4"
H4_TRAINING_SEEDS = np.concatenate(
    (TRAINING_SEEDS, np.arange(201, 225, dtype=np.int64))
)
H4_TEST_SEEDS = np.arange(1001, 1065, dtype=np.int64)
LEARNING_RATE = 0.01
MAX_ITERATIONS = 20


def _seed_batches(seeds):
    for start in range(0, len(seeds), 8):
        yield seeds[start : start + 8]


def _shared_full_batch_loss(physics, objective, shared):
    batch_losses = [
        apply_torch_pipeline(
            physics,
            objective,
            shared_coefficient_batch(shared),
            seeds,
        )
        for seeds in _seed_batches(H4_TRAINING_SEEDS)
    ]
    return torch.stack(batch_losses).mean()


def _mlp_full_batch_loss(physics, objective, controller, descriptors):
    batch_losses = []
    for start, seeds in zip(
        range(0, len(H4_TRAINING_SEEDS), 8),
        _seed_batches(H4_TRAINING_SEEDS),
    ):
        batch_losses.append(
            apply_torch_pipeline(
                physics,
                objective,
                controller(descriptors[start : start + 8]),
                seeds,
            )
        )
    return torch.stack(batch_losses).mean()


def _control_range(coefficients):
    histories = np.asarray(preload_history(BASE_Q[1], coefficients))
    return float(np.min(histories)), float(np.max(histories))


def _fixed_coefficients(seeds):
    return np.zeros((len(seeds), 5), dtype=np.float64)


def _evaluate_training_objective(physics, objective, coefficient_function):
    losses, coefficients = evaluate_seed_set(
        physics,
        objective,
        H4_TRAINING_SEEDS,
        coefficient_function,
    )
    return float(np.mean(losses)), coefficients


def train_shared(physics, objective):
    shared = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = torch.optim.Adam([shared], lr=LEARNING_RATE)
    initial_objective, initial_coefficients = _evaluate_training_objective(
        physics, objective, _fixed_coefficients
    )
    initial_min, initial_max = _control_range(initial_coefficients)
    history = [
        {
            "iteration": 0,
            "objective": initial_objective,
            "gradient_norm": np.nan,
            "N_min": initial_min,
            "N_max": initial_max,
        }
    ]

    start = time.perf_counter()
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _shared_full_batch_loss(physics, objective, shared)
        loss.backward()
        gradient_norm = float(torch.linalg.vector_norm(shared.grad))
        optimizer.step()

        shared_numpy = shared.detach().cpu().numpy()
        coefficient_function = lambda seeds: np.broadcast_to(
            shared_numpy, (len(seeds), 5)
        ).copy()
        hard_objective, coefficients = _evaluate_training_objective(
            physics, objective, coefficient_function
        )
        preload_min, preload_max = _control_range(coefficients)
        history.append(
            {
                "iteration": iteration,
                "objective": hard_objective,
                "gradient_norm": gradient_norm,
                "N_min": preload_min,
                "N_max": preload_max,
            }
        )
    elapsed = time.perf_counter() - start
    return shared.detach().cpu().numpy(), history, elapsed


def train_mlp(physics, objective):
    controller = build_controller()
    optimizer = torch.optim.Adam(controller.parameters(), lr=LEARNING_RATE)
    initial_coefficients = controller_coefficients(
        controller, H4_TRAINING_SEEDS
    )
    initial_losses, _ = evaluate_seed_set(
        physics,
        objective,
        H4_TRAINING_SEEDS,
        lambda seeds: controller_coefficients(controller, seeds),
    )
    initial_min, initial_max = _control_range(initial_coefficients)
    history = [
        {
            "iteration": 0,
            "objective": float(np.mean(initial_losses)),
            "gradient_norm": np.nan,
            "N_min": initial_min,
            "N_max": initial_max,
        }
    ]
    descriptors = torch.from_numpy(
        forcing_descriptor_batch(H4_TRAINING_SEEDS)
    )

    start = time.perf_counter()
    for iteration in range(1, MAX_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _mlp_full_batch_loss(
            physics, objective, controller, descriptors
        )
        loss.backward()
        gradient_norm = parameter_gradient_norm(controller.parameters())
        optimizer.step()

        coefficients = controller_coefficients(controller, H4_TRAINING_SEEDS)
        hard_losses, _ = evaluate_seed_set(
            physics,
            objective,
            H4_TRAINING_SEEDS,
            lambda seeds: controller_coefficients(controller, seeds),
        )
        preload_min, preload_max = _control_range(coefficients)
        history.append(
            {
                "iteration": iteration,
                "objective": float(np.mean(hard_losses)),
                "gradient_norm": gradient_norm,
                "N_min": preload_min,
                "N_max": preload_max,
            }
        )
    elapsed = time.perf_counter() - start
    return controller, history, elapsed


def _aggregate_improvements(means):
    return {
        "Shared vs Fixed": (means["Fixed"] - means["Shared"])
        / means["Fixed"],
        "MLP vs Fixed": (means["Fixed"] - means["MLP"])
        / means["Fixed"],
        "MLP vs Shared": (means["Shared"] - means["MLP"])
        / means["Shared"],
    }


def _plot_results(test_losses, histories, fixed_training_objective):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_figure_style()
    comparisons = relative_comparisons(test_losses)

    generalization_path = OUTPUT_DIRECTORY / "per_seed_generalization.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.axhline(
        0.0,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="Fixed",
    )
    axis.plot(
        H4_TEST_SEEDS,
        100.0 * comparisons["Shared vs Fixed"],
        color="#3B6FB6",
        marker="o",
        markersize=3.8,
        linewidth=1.3,
        label="Shared Fourier",
    )
    axis.plot(
        H4_TEST_SEEDS,
        100.0 * comparisons["MLP vs Fixed"],
        color="#D17A22",
        marker="s",
        markersize=3.6,
        linewidth=1.3,
        label="MLP Fourier",
    )
    axis.set_xlabel("Test seed")
    axis.set_ylabel("Relative improvement (%)")
    axis.set_xticks(H4_TEST_SEEDS[::8])
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(generalization_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    history_path = OUTPUT_DIRECTORY / "training_objective_history.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.axhline(
        fixed_training_objective,
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="Fixed",
    )
    for name, color, marker in (
        ("Shared", "#3B6FB6", "o"),
        ("MLP", "#D17A22", "s"),
    ):
        iterations = [entry["iteration"] for entry in histories[name]]
        objectives = [entry["objective"] for entry in histories[name]]
        axis.plot(
            iterations,
            objectives,
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.5,
            label=name,
        )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective")
    axis.set_xticks(np.arange(0, MAX_ITERATIONS + 1, 2))
    axis.legend()
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(history_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return generalization_path, history_path


def main() -> int:
    torch.set_default_dtype(torch.float64)
    physics, objective = create_tesseracts()

    shared, shared_history, shared_seconds = train_shared(physics, objective)
    controller, mlp_history, mlp_seconds = train_mlp(physics, objective)

    shared_function = lambda seeds: np.broadcast_to(
        shared, (len(seeds), 5)
    ).copy()
    mlp_function = lambda seeds: controller_coefficients(controller, seeds)
    coefficient_functions = {
        "Fixed": _fixed_coefficients,
        "Shared": shared_function,
        "MLP": mlp_function,
    }

    train_losses = {}
    train_coefficients = {}
    for method, coefficient_function in coefficient_functions.items():
        train_losses[method], train_coefficients[method] = evaluate_seed_set(
            physics,
            objective,
            H4_TRAINING_SEEDS,
            coefficient_function,
        )

    evaluation_start = time.perf_counter()
    test_losses = {}
    test_coefficients = {}
    for method, coefficient_function in coefficient_functions.items():
        test_losses[method], test_coefficients[method] = evaluate_seed_set(
            physics,
            objective,
            H4_TEST_SEEDS,
            coefficient_function,
        )
    evaluation_seconds = time.perf_counter() - evaluation_start

    train_means = {
        method: float(np.mean(losses))
        for method, losses in train_losses.items()
    }
    test_means = {
        method: float(np.mean(losses))
        for method, losses in test_losses.items()
    }
    train_improvements = _aggregate_improvements(train_means)
    test_improvements = _aggregate_improvements(test_means)
    test_relative = relative_comparisons(test_losses)
    win_counts = {
        comparison: int(np.count_nonzero(values > 0.0))
        for comparison, values in test_relative.items()
    }
    medians = {
        comparison: float(np.median(values))
        for comparison, values in test_relative.items()
    }

    if test_means["MLP"] < test_means["Shared"]:
        decision = "Case A / STRONG PASS"
    elif test_means["MLP"] < test_means["Fixed"]:
        decision = "Case B"
    else:
        decision = "Case C"

    shared_range = _control_range(train_coefficients["Shared"])
    mlp_train_range = _control_range(train_coefficients["MLP"])
    mlp_test_range = _control_range(test_coefficients["MLP"])
    shared_gap = (
        train_improvements["Shared vs Fixed"]
        - test_improvements["Shared vs Fixed"]
    )
    mlp_gap = (
        train_improvements["MLP vs Fixed"]
        - test_improvements["MLP vs Fixed"]
    )
    mlp_minus_shared = test_means["MLP"] - test_means["Shared"]

    histories = {"Shared": shared_history, "MLP": mlp_history}
    finite_values = [
        *train_losses.values(),
        *test_losses.values(),
        shared,
        [entry["objective"] for entry in shared_history],
        [entry["gradient_norm"] for entry in shared_history[1:]],
        [entry["objective"] for entry in mlp_history],
        [entry["gradient_norm"] for entry in mlp_history[1:]],
    ]
    seed_gate = (
        len(H4_TRAINING_SEEDS) == 32
        and len(H4_TEST_SEEDS) == 64
        and set(H4_TRAINING_SEEDS).isdisjoint(H4_TEST_SEEDS)
    )
    baseline_gate = (
        shared_history[0]["objective"] == train_means["Fixed"]
        and mlp_history[0]["objective"] == train_means["Fixed"]
    )
    passed = (
        seed_gate
        and baseline_gate
        and len(shared_history) == MAX_ITERATIONS + 1
        and len(mlp_history) == MAX_ITERATIONS + 1
        and all(np.all(np.isfinite(value)) for value in finite_values)
    )

    print("## Summary")
    print(f"training_seeds: {H4_TRAINING_SEEDS.tolist()}")
    print(f"test_seeds: {H4_TEST_SEEDS.tolist()}")
    print(f"full_batch_groups: {len(H4_TRAINING_SEEDS) // 8}")
    print("optimizer_updates_per_iteration: 1")
    print("## Training")
    for name, history in histories.items():
        for entry in history:
            print(
                f"{name} iter={entry['iteration']} "
                f"J={entry['objective']:.16g} "
                f"gradient_norm={entry['gradient_norm']:.12g} "
                f"N_min={entry['N_min']:.12g} "
                f"N_max={entry['N_max']:.12g}"
            )
    print("## Objectives")
    for split, losses, means in (
        ("train", train_losses, train_means),
        ("test", test_losses, test_means),
    ):
        for method in ("Fixed", "Shared", "MLP"):
            print(f"{split}_{method}_mean: {means[method]:.16g}")
            print(f"{split}_{method}_seed_losses: {losses[method].tolist()}")
    print(f"train_aggregate_relative_improvement: {train_improvements}")
    print(f"test_aggregate_relative_improvement: {test_improvements}")
    print(f"test_win_counts: {win_counts}")
    print(f"test_median_relative_improvement: {medians}")
    print(f"shared_train_test_gap: {shared_gap:.16g}")
    print(f"mlp_train_test_gap: {mlp_gap:.16g}")
    print(f"J_mlp_test_minus_J_shared_test: {mlp_minus_shared:.16g}")
    print("## Controllers")
    print(f"shared_coefficients: {shared.tolist()}")
    print(f"shared_N_range: {list(shared_range)}")
    print(f"mlp_train_N_range: {list(mlp_train_range)}")
    print(f"mlp_test_N_range: {list(mlp_test_range)}")
    print("## Runtime")
    print(f"shared_training_seconds: {shared_seconds:.9g}")
    print(f"mlp_training_seconds: {mlp_seconds:.9g}")
    print(f"test_evaluation_seconds: {evaluation_seconds:.9g}")
    print("## Decision")
    print(decision)
    print("## PASS" if passed else "## FAIL")
    if passed:
        for path in _plot_results(
            test_losses, histories, train_means["Fixed"]
        ):
            print(f"figure: {path}")
    else:
        print(f"seed_gate: {seed_gate}")
        print(f"baseline_gate: {baseline_gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
