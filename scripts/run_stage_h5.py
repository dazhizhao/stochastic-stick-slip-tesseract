"""Run the Stage H5 two-Tesseract PyTorch/JAX regression pipeline."""

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_stage_h3 import BASE_Q
from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.controller import (
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    flatten_controller_parameters,
)
from stochastic_stick_slip.model import forcing_descriptor_batch


CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
LEARNING_RATE = 0.01
MAX_ITERATIONS = 20
H4_TRAIN_OBJECTIVE = 0.006108705858227541
H4_TEST_OBJECTIVE = 0.005971312227273379


def create_tesseracts():
    controller = Tesseract.from_tesseract_api(CONTROLLER_API)
    physics = Tesseract.from_tesseract_api(PHYSICS_API)
    return controller, physics


def initial_theta():
    return flatten_controller_parameters(build_controller()).detach().clone()


def _old_controller_gradient(controller, descriptors, cotangent):
    coefficients = controller(descriptors)
    gradients = torch.autograd.grad(
        coefficients,
        tuple(controller.parameters()),
        grad_outputs=cotangent,
    )
    return coefficients.detach().cpu().numpy(), torch.cat(
        [gradient.reshape(-1) for gradient in gradients]
    ).detach().cpu().numpy()


def equivalence_check(controller_tesseract):
    seeds = H4_TRAINING_SEEDS[:8]
    descriptors_numpy = forcing_descriptor_batch(seeds)
    descriptors = torch.from_numpy(descriptors_numpy)
    controller = build_controller()
    theta = flatten_controller_parameters(controller).detach().cpu().numpy()
    cotangent = torch.linspace(
        -1.0, 1.0, 40, dtype=torch.float64
    ).reshape(8, 5)
    old_coefficients, old_gradient = _old_controller_gradient(
        controller, descriptors, cotangent
    )
    new_coefficients = np.asarray(
        controller_tesseract.apply(
            {"theta": theta, "descriptors": descriptors_numpy}
        )["coeffs"]
    )
    new_gradient = np.asarray(
        controller_tesseract.vector_jacobian_product(
            {"theta": theta, "descriptors": descriptors_numpy},
            ["theta"],
            ["coeffs"],
            {"coeffs": cotangent.numpy()},
        )["theta"]
    )
    cosine = float(
        np.dot(old_gradient, new_gradient)
        / (np.linalg.norm(old_gradient) * np.linalg.norm(new_gradient))
    )
    return {
        "forward_max_abs_error": float(
            np.max(np.abs(old_coefficients - new_coefficients))
        ),
        "vjp_max_abs_error": float(
            np.max(np.abs(old_gradient - new_gradient))
        ),
        "cosine": cosine,
        "forward_gate": np.allclose(
            old_coefficients, new_coefficients, rtol=1e-12, atol=1e-14
        ),
        "vjp_gate": np.allclose(
            old_gradient, new_gradient, rtol=1e-10, atol=1e-12
        ),
    }


def _differentiable_batch_loss(
    controller_tesseract,
    physics_tesseract,
    theta,
    seeds,
):
    descriptors = forcing_descriptor_batch(seeds)
    coefficients = apply_tesseract(
        controller_tesseract,
        {"theta": theta, "descriptors": descriptors},
    )["coeffs"]
    response = apply_tesseract(
        physics_tesseract,
        {"q": BASE_Q, "coeffs": coefficients, "seeds": seeds},
    )
    return response["seed_losses"]


def full_training_loss(controller_tesseract, physics_tesseract, theta):
    losses = []
    for start in range(0, len(H4_TRAINING_SEEDS), 8):
        seeds = H4_TRAINING_SEEDS[start : start + 8]
        losses.append(
            _differentiable_batch_loss(
                controller_tesseract,
                physics_tesseract,
                theta,
                seeds,
            )
        )
    return torch.cat(losses).mean()


def evaluate_controller(
    controller_tesseract,
    physics_tesseract,
    theta,
    seeds,
):
    theta_numpy = np.asarray(theta, dtype=np.float64)
    losses = []
    for start in range(0, len(seeds), 8):
        batch_seeds = seeds[start : start + 8]
        coefficients = controller_tesseract.apply(
            {
                "theta": theta_numpy,
                "descriptors": forcing_descriptor_batch(batch_seeds),
            }
        )["coeffs"]
        response = physics_tesseract.apply(
            {
                "q": BASE_Q,
                "coeffs": coefficients,
                "seeds": batch_seeds,
            }
        )
        losses.append(np.asarray(response["seed_losses"]))
    return np.concatenate(losses)


def evaluate_fixed(physics_tesseract, seeds):
    losses = []
    for start in range(0, len(seeds), 8):
        batch_seeds = seeds[start : start + 8]
        response = physics_tesseract.apply(
            {
                "q": BASE_Q,
                "coeffs": np.zeros((8, 5), dtype=np.float64),
                "seeds": batch_seeds,
            }
        )
        losses.append(np.asarray(response["seed_losses"]))
    return np.concatenate(losses)


def end_to_end_backward(controller_tesseract, physics_tesseract):
    theta = torch.nn.Parameter(initial_theta())
    start = time.perf_counter()
    losses = _differentiable_batch_loss(
        controller_tesseract,
        physics_tesseract,
        theta,
        H4_TRAINING_SEEDS[:8],
    )
    loss = losses.mean()
    loss.backward()
    elapsed = time.perf_counter() - start
    gradient = theta.grad.detach().cpu().numpy()
    return float(loss.detach()), gradient, elapsed


def train(controller_tesseract, physics_tesseract):
    theta = torch.nn.Parameter(initial_theta())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    initial_losses = evaluate_controller(
        controller_tesseract,
        physics_tesseract,
        theta.detach().cpu().numpy(),
        H4_TRAINING_SEEDS,
    )
    history = [float(np.mean(initial_losses))]
    gradient_norms = []

    start = time.perf_counter()
    for _ in range(MAX_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        loss = full_training_loss(
            controller_tesseract, physics_tesseract, theta
        )
        loss.backward()
        gradient_norms.append(float(torch.linalg.vector_norm(theta.grad)))
        optimizer.step()
        hard_losses = evaluate_controller(
            controller_tesseract,
            physics_tesseract,
            theta.detach().cpu().numpy(),
            H4_TRAINING_SEEDS,
        )
        history.append(float(np.mean(hard_losses)))
    elapsed = time.perf_counter() - start
    return theta.detach().cpu().numpy(), history, gradient_norms, elapsed


def main() -> int:
    torch.set_default_dtype(torch.float64)
    controller_tesseract, physics_tesseract = create_tesseracts()

    equivalence = equivalence_check(controller_tesseract)
    backward_loss, backward_gradient, backward_seconds = end_to_end_backward(
        controller_tesseract, physics_tesseract
    )
    backward_gate = (
        backward_gradient.shape == (NUM_CONTROLLER_PARAMETERS,)
        and np.all(np.isfinite(backward_gradient))
        and np.linalg.norm(backward_gradient) > 0.0
    )

    theta, history, gradient_norms, training_seconds = train(
        controller_tesseract, physics_tesseract
    )
    train_losses = evaluate_controller(
        controller_tesseract,
        physics_tesseract,
        theta,
        H4_TRAINING_SEEDS,
    )
    fixed_train_losses = evaluate_fixed(physics_tesseract, H4_TRAINING_SEEDS)

    evaluation_start = time.perf_counter()
    test_losses = evaluate_controller(
        controller_tesseract,
        physics_tesseract,
        theta,
        H4_TEST_SEEDS,
    )
    fixed_test_losses = evaluate_fixed(physics_tesseract, H4_TEST_SEEDS)
    evaluation_seconds = time.perf_counter() - evaluation_start

    train_objective = float(np.mean(train_losses))
    test_objective = float(np.mean(test_losses))
    fixed_train = float(np.mean(fixed_train_losses))
    fixed_test = float(np.mean(fixed_test_losses))
    train_improvement = (fixed_train - train_objective) / fixed_train
    test_improvement = (fixed_test - test_objective) / fixed_test
    train_reference_gate = np.isclose(
        train_objective, H4_TRAIN_OBJECTIVE, rtol=1e-8, atol=1e-12
    )
    test_reference_gate = np.isclose(
        test_objective, H4_TEST_OBJECTIVE, rtol=1e-8, atol=1e-12
    )
    training_gate = (
        len(history) == MAX_ITERATIONS + 1
        and history[-1] < history[0]
        and np.all(np.isfinite(history))
        and np.all(np.isfinite(gradient_norms))
    )
    passed = (
        equivalence["forward_gate"]
        and equivalence["vjp_gate"]
        and equivalence["cosine"] > 0.999999999
        and backward_gate
        and training_gate
        and test_objective < fixed_test
        and train_reference_gate
        and test_reference_gate
    )

    print("## Summary")
    print("core_tesseracts: fourier_controller -> stick_slip_fem")
    print(f"theta_shape: {list(theta.shape)}")
    print("## Tesseract pipeline")
    print(f"controller_forward_max_abs_error: {equivalence['forward_max_abs_error']:.16g}")
    print(f"controller_vjp_max_abs_error: {equivalence['vjp_max_abs_error']:.16g}")
    print(f"controller_vjp_direction_cosine: {equivalence['cosine']:.16g}")
    print(f"end_to_end_initial_loss: {backward_loss:.16g}")
    print(f"end_to_end_gradient_norm: {np.linalg.norm(backward_gradient):.16g}")
    print(f"end_to_end_gradient_shape: {list(backward_gradient.shape)}")
    print("physics_vjp: existing_crn_centered_fd")
    print("## Results")
    print(f"H4_reference_train: {H4_TRAIN_OBJECTIVE:.16g}")
    print(f"H5_train: {train_objective:.16g}")
    print(f"H5_fixed_train: {fixed_train:.16g}")
    print(f"H5_train_relative_improvement: {train_improvement:.16g}")
    print(f"H4_reference_test: {H4_TEST_OBJECTIVE:.16g}")
    print(f"H5_test: {test_objective:.16g}")
    print(f"H5_fixed_test: {fixed_test:.16g}")
    print(f"H5_test_relative_improvement: {test_improvement:.16g}")
    print(f"train_reference_delta: {train_objective - H4_TRAIN_OBJECTIVE:.16g}")
    print(f"test_reference_delta: {test_objective - H4_TEST_OBJECTIVE:.16g}")
    print(f"training_history: {history}")
    print(f"gradient_norms: {gradient_norms}")
    print("## Runtime")
    print(f"end_to_end_backward_seconds: {backward_seconds:.9g}")
    print(f"training_seconds: {training_seconds:.9g}")
    print(f"test_evaluation_seconds: {evaluation_seconds:.9g}")
    print("## PASS" if passed else "## FAIL")
    if not passed:
        print(f"forward_gate: {equivalence['forward_gate']}")
        print(f"vjp_gate: {equivalence['vjp_gate']}")
        print(f"backward_gate: {backward_gate}")
        print(f"training_gate: {training_gate}")
        print(f"train_reference_gate: {train_reference_gate}")
        print(f"test_reference_gate: {test_reference_gate}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
