from pathlib import Path

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_stage_h3 import BASE_Q
from stochastic_stick_slip.controller import (
    CONTROLLER_PARAMETER_LAYOUT,
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    controller_parameter_dict,
    flatten_controller_parameters,
)
from stochastic_stick_slip.model import TRAINING_SEEDS, forcing_descriptor_batch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tesseracts():
    controller = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/fourier_controller/tesseract_api.py"
    )
    physics = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
    )
    return controller, physics


def test_controller_parameter_layout_is_fixed() -> None:
    theta = flatten_controller_parameters(build_controller())
    parameters = controller_parameter_dict(theta)
    assert theta.shape == (NUM_CONTROLLER_PARAMETERS,)
    assert NUM_CONTROLLER_PARAMETERS == 469
    assert tuple(parameters) == tuple(
        name for name, _ in CONTROLLER_PARAMETER_LAYOUT
    )
    for name, shape in CONTROLLER_PARAMETER_LAYOUT:
        assert parameters[name].shape == shape


def test_controller_tesseract_forward_matches_mlp(tesseracts) -> None:
    controller_tesseract, _ = tesseracts
    controller = build_controller()
    descriptors = torch.from_numpy(forcing_descriptor_batch(TRAINING_SEEDS))
    theta = flatten_controller_parameters(controller)
    expected = controller(descriptors).detach().numpy()
    actual = controller_tesseract.apply(
        {"theta": theta.detach().numpy(), "descriptors": descriptors.numpy()}
    )["coeffs"]
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_controller_tesseract_vjp_matches_pytorch(tesseracts) -> None:
    controller_tesseract, _ = tesseracts
    controller = build_controller()
    descriptors = torch.from_numpy(forcing_descriptor_batch(TRAINING_SEEDS))
    theta = flatten_controller_parameters(controller).detach().numpy()
    cotangent = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64).reshape(8, 5)
    coefficients = controller(descriptors)
    gradients = torch.autograd.grad(
        coefficients,
        tuple(controller.parameters()),
        grad_outputs=cotangent,
    )
    expected = torch.cat([gradient.reshape(-1) for gradient in gradients]).numpy()
    actual = controller_tesseract.vector_jacobian_product(
        {"theta": theta, "descriptors": descriptors.numpy()},
        ["theta"],
        ["coeffs"],
        {"coeffs": cotangent.numpy()},
    )["theta"]
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual) > 0.0
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_two_tesseract_backward_reaches_theta(tesseracts) -> None:
    controller_tesseract, physics_tesseract = tesseracts
    theta = torch.nn.Parameter(
        flatten_controller_parameters(build_controller()).detach().clone()
    )
    descriptors = forcing_descriptor_batch(TRAINING_SEEDS)
    coefficients = apply_tesseract(
        controller_tesseract,
        {"theta": theta, "descriptors": descriptors},
    )["coeffs"]
    response = apply_tesseract(
        physics_tesseract,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": TRAINING_SEEDS},
    )
    response["seed_losses"].mean().backward()
    gradient = theta.grad.detach().numpy()
    assert gradient.shape == (469,)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
