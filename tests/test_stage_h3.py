import numpy as np
import torch

from scripts.run_stage_h3 import (
    BASE_Q,
    H3_TEST_SEEDS,
    apply_torch_pipeline,
    controller_coefficients,
    create_tesseracts,
    evaluate_numpy_batch,
    shared_coefficient_batch,
)
from stochastic_stick_slip.controller import build_controller
from stochastic_stick_slip.model import (
    HELD_OUT_SEEDS,
    NUM_STEPS,
    TRAINING_SEEDS,
    preload_history,
)

ZERO_COEFFICIENTS = np.zeros((8, 5), dtype=np.float64)


def test_shared_zero_is_exact_fixed_preload_and_objective() -> None:
    shared_zero = shared_coefficient_batch(np.zeros(5, dtype=np.float64))
    preload = np.asarray(preload_history(BASE_Q[1], shared_zero))
    assert np.array_equal(preload, np.full((8, NUM_STEPS), 0.04))
    physics, objective = create_tesseracts()
    _, fixed = evaluate_numpy_batch(
        physics, objective, ZERO_COEFFICIENTS, TRAINING_SEEDS
    )
    _, shared = evaluate_numpy_batch(
        physics, objective, shared_zero, TRAINING_SEEDS
    )
    assert shared == fixed


def test_shared_gradient_crosses_both_tesseracts() -> None:
    physics, objective = create_tesseracts()
    shared = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    loss = apply_torch_pipeline(
        physics,
        objective,
        shared_coefficient_batch(shared),
        TRAINING_SEEDS,
    )
    loss.backward()
    gradient = shared.grad.detach().cpu().numpy()
    assert torch.isfinite(loss)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_h3_test_seeds_are_disjoint() -> None:
    assert len(H3_TEST_SEEDS) == 32
    assert set(H3_TEST_SEEDS).isdisjoint(TRAINING_SEEDS)
    assert set(H3_TEST_SEEDS).isdisjoint(HELD_OUT_SEEDS)


def test_all_three_controller_modes_are_finite() -> None:
    physics, objective = create_tesseracts()
    controller = build_controller()
    fixed = ZERO_COEFFICIENTS
    shared = shared_coefficient_batch(
        np.array([0.1, -0.1, 0.05, -0.05, 0.02], dtype=np.float64)
    )
    mlp = controller_coefficients(controller, TRAINING_SEEDS)
    for coefficients in (fixed, shared, mlp):
        losses, value = evaluate_numpy_batch(
            physics, objective, coefficients, TRAINING_SEEDS
        )
        assert np.all(np.isfinite(losses))
        assert np.isfinite(value)
