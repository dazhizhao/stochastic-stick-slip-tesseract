from pathlib import Path

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
    NUM_STEPS,
    TRAINING_SEEDS,
    crn_fd_coefficient_jacobian,
    evaluate_controlled_batch,
    forcing_descriptor_batch,
    preload_history,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_Q = np.array([BASELINE_DAMPING, 0.04], dtype=np.float64)
ZERO_COEFFICIENTS = np.zeros((8, 5), dtype=np.float64)


def _tesseracts():
    physics = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
    )
    objective = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
    )
    return physics, objective


def test_forcing_descriptors_and_controller_are_reproducible() -> None:
    first = forcing_descriptor_batch(TRAINING_SEEDS)
    second = forcing_descriptor_batch(TRAINING_SEEDS)
    controller = build_controller()
    coefficients = controller(torch.from_numpy(first))
    assert np.array_equal(first, second)
    assert coefficients.shape == (8, 5)
    assert torch.count_nonzero(coefficients) == 0


def test_zero_coefficients_reproduce_constant_preload() -> None:
    preload = np.asarray(preload_history(0.04, ZERO_COEFFICIENTS))
    assert preload.shape == (8, NUM_STEPS)
    assert np.array_equal(preload, np.full((8, NUM_STEPS), 0.04))


def test_controlled_forward_and_coefficient_gradient_are_finite() -> None:
    result = evaluate_controlled_batch(
        BASE_Q, ZERO_COEFFICIENTS, TRAINING_SEEDS
    )
    gradient = crn_fd_coefficient_jacobian(
        BASE_Q, ZERO_COEFFICIENTS, TRAINING_SEEDS
    )
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert gradient.shape == (8, 5)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_torch_backward_crosses_both_tesseracts() -> None:
    physics, objective = _tesseracts()
    controller = build_controller()
    descriptors = torch.from_numpy(forcing_descriptor_batch(TRAINING_SEEDS))
    coefficients = controller(descriptors)
    response = apply_tesseract(
        physics,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": TRAINING_SEEDS},
    )
    loss = apply_tesseract(
        objective, {"seed_losses": response["seed_losses"]}
    )["objective"]
    loss.backward()
    total_norm = parameter_gradient_norm(controller.parameters())
    final_norm = parameter_gradient_norm(controller[-1].parameters())
    assert torch.isfinite(loss)
    assert np.isfinite(total_norm) and total_norm > 0.0
    assert np.isfinite(final_norm) and final_norm > 0.0
