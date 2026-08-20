from pathlib import Path

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from stochastic_stick_slip.controller import (
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    flatten_controller_parameters,
)
from stochastic_stick_slip.engineering_markov import (
    GATE_A_FORCING_SEEDS,
    markov_uniform_bank,
)
from stochastic_stick_slip.model import forcing_descriptor_batch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tesseracts():
    controller = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/fourier_controller/tesseract_api.py"
    )
    physics = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/markov_jump_fem/tesseract_api.py"
    )
    return controller, physics


def _physics_inputs():
    return {
        "coeffs": np.zeros((8, 5), dtype=np.float64),
        "forcing_seeds": GATE_A_FORCING_SEEDS,
        "markov_uniforms": markov_uniform_bank(4, stream_id=5),
    }


def test_markov_jump_tesseract_apply_is_finite(tesseracts) -> None:
    _, physics = tesseracts
    result = physics.apply(_physics_inputs())

    assert np.asarray(result["seed_losses"]).shape == (8,)
    assert np.asarray(result["transition_counts"]).shape == (8, 4, 2)
    assert np.asarray(result["high_mode_fraction"]).shape == (8, 4, 2)
    assert np.all(np.isfinite(result["seed_losses"]))
    assert np.all(np.isfinite(result["high_mode_fraction"]))


def test_markov_jump_tesseract_vjp_is_reproducible_and_nonzero(
    tesseracts,
) -> None:
    _, physics = tesseracts
    inputs = _physics_inputs()
    cotangent = {"seed_losses": np.ones(8, dtype=np.float64)}
    first = physics.vector_jacobian_product(
        inputs, ["coeffs"], ["seed_losses"], cotangent
    )["coeffs"]
    second = physics.vector_jacobian_product(
        inputs, ["coeffs"], ["seed_losses"], cotangent
    )["coeffs"]

    assert np.asarray(first).shape == (8, 5)
    assert np.all(np.isfinite(first))
    assert np.linalg.norm(first) > 0.0
    assert np.array_equal(first, second)


def test_markov_jump_two_tesseract_backward_reaches_theta(tesseracts) -> None:
    controller, physics = tesseracts
    theta = torch.nn.Parameter(
        flatten_controller_parameters(build_controller()).detach().clone()
    )
    coefficients = apply_tesseract(
        controller,
        {
            "theta": theta,
            "descriptors": forcing_descriptor_batch(GATE_A_FORCING_SEEDS),
        },
    )["coeffs"]
    response = apply_tesseract(
        physics,
        {
            "coeffs": coefficients,
            "forcing_seeds": GATE_A_FORCING_SEEDS,
            "markov_uniforms": markov_uniform_bank(4, stream_id=5),
        },
    )
    response["seed_losses"].mean().backward()
    gradient = theta.grad.detach().numpy()

    assert gradient.shape == (NUM_CONTROLLER_PARAMETERS,)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
