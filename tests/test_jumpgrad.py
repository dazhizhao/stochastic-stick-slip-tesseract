from pathlib import Path

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from scripts.run_jumpgrad_end_to_end import differentiable_loss
from stochastic_stick_slip.jumpgrad import (
    AUDIT_STREAM,
    NUM_CONTROLLER_PARAMETERS,
    OMEGA_R,
    TRAINING_CONDITIONS,
    build_jumpgrad_controller,
    condition_descriptors,
    crn_fd_condition_gradient,
    deterministic_condition_objectives,
    evaluate_jumpgrad_bank,
    flatten_jumpgrad_parameters,
    generate_jumpgrad_histories,
    jumpgrad_uniform_bank,
)
from stochastic_stick_slip.wu_v2 import (
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    PRELOAD_HIGH,
    PRELOAD_LOW,
    evaluate_markov_bank,
)


ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture(scope="module")
def tesseracts():
    controller = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/jumpgrad_controller/tesseract_api.py"
    )
    physics = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/wu_v2_markov_fem/tesseract_api.py"
    )
    return controller, physics


def test_controller_layout_descriptors_and_neutral_output() -> None:
    controller = build_jumpgrad_controller()
    theta = flatten_jumpgrad_parameters(controller)
    descriptors = condition_descriptors(TRAINING_CONDITIONS)
    output = controller(torch.from_numpy(descriptors)).detach().numpy()
    assert theta.shape == (NUM_CONTROLLER_PARAMETERS,) == (354,)
    assert np.allclose(descriptors[0], [-0.2, 0.0])
    assert np.allclose(descriptors[-1], [0.4, 0.4])
    assert output.shape == (8, 2)
    assert np.array_equal(output, np.zeros((8, 2)))


def test_controller_tesseract_forward_and_vjp_match_torch(tesseracts) -> None:
    controller_tesseract, _ = tesseracts
    controller = build_jumpgrad_controller()
    theta = flatten_jumpgrad_parameters(controller).detach().numpy()
    descriptors = condition_descriptors(TRAINING_CONDITIONS[:2])
    expected_output = controller(torch.from_numpy(descriptors)).detach().numpy()
    actual_output = controller_tesseract.apply(
        {"theta": theta, "descriptors": descriptors}
    )["q"]
    assert np.allclose(actual_output, expected_output, rtol=RTOL, atol=ATOL)

    cotangent = torch.tensor([[0.3, -0.2], [-0.5, 0.7]], dtype=torch.float64)
    direct_output = controller(torch.from_numpy(descriptors))
    direct_gradients = torch.autograd.grad(
        direct_output,
        tuple(controller.parameters()),
        grad_outputs=cotangent,
    )
    expected_gradient = torch.cat(
        [gradient.reshape(-1) for gradient in direct_gradients]
    ).numpy()
    actual_gradient = controller_tesseract.vector_jacobian_product(
        {"theta": theta, "descriptors": descriptors},
        ["theta"],
        ["q"],
        {"q": cotangent.numpy()},
    )["theta"]
    assert np.allclose(
        actual_gradient, expected_gradient, rtol=RTOL, atol=ATOL
    )


def test_hard_histories_and_dynamic_shapes() -> None:
    q = np.zeros((8, 2), dtype=np.float64)
    tapes = jumpgrad_uniform_bank(8, 8, AUDIT_STREAM, iteration=0)
    histories = generate_jumpgrad_histories(q, TRAINING_CONDITIONS, tapes)
    mechanics = evaluate_jumpgrad_bank(q, TRAINING_CONDITIONS, tapes)
    assert histories["preload"].shape == (8, 8, 2400, 2)
    assert histories["transition_counts"].shape == (8, 8, 2)
    assert mechanics["objectives"].shape == (8,)
    assert mechanics["trajectory_objectives"].shape == (8, 8)
    assert np.all(np.isfinite(mechanics["objectives"]))
    assert set(np.unique(histories["preload"])) == {
        PRELOAD_LOW,
        PRELOAD_HIGH,
    }
    assert not np.array_equal(tapes[..., 0], tapes[..., 1])


def test_condition_batch_matches_independent_frequency_forwards() -> None:
    conditions = TRAINING_CONDITIONS[:2]
    q = np.asarray([[0.2, -0.1], [-0.3, 0.15]], dtype=np.float64)
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    batch = evaluate_jumpgrad_bank(q, conditions, tapes)["objectives"]
    independent = []
    for index, (force_ratio, frequency_ratio) in enumerate(conditions):
        omega = OMEGA_R * frequency_ratio
        time_step, forcing = single_tone_forcing(
            FORCING_AMPLITUDE * force_ratio,
            omega,
            DIAGNOSTIC_NUM_PERIODS,
        )
        times = time_step * np.arange(1, 2401, dtype=np.float64)
        result = evaluate_markov_bank(
            q[index], forcing=forcing, uniforms=tapes[index],
            times=times, omega=omega, time_step=time_step,
        )
        independent.append(np.mean(result["trajectory_objectives"]))
    assert np.allclose(batch, independent, rtol=RTOL, atol=ATOL)


def test_physics_tesseract_forward_and_crn_vjp(tesseracts) -> None:
    _, physics = tesseracts
    conditions = TRAINING_CONDITIONS[:2]
    q = np.asarray([[0.2, -0.1], [-0.3, 0.15]], dtype=np.float64)
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    standalone = evaluate_jumpgrad_bank(q, conditions, tapes)
    actual = physics.apply(
        {"q": q, "conditions": conditions, "markov_tapes": tapes}
    )
    assert np.allclose(
        actual["objectives"], standalone["objectives"], rtol=RTOL, atol=ATOL
    )
    assert np.asarray(actual["transition_counts"]).shape == (2, 2, 2)

    cotangent = np.asarray([0.3, -0.7], dtype=np.float64)
    expected = cotangent[:, None] * crn_fd_condition_gradient(
        q, conditions, tapes
    )["gradient"]
    gradient = physics.vector_jacobian_product(
        {"q": q, "conditions": conditions, "markov_tapes": tapes},
        ["q"],
        ["objectives"],
        {"objectives": cotangent},
    )["q"]
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
    assert np.allclose(gradient, expected, rtol=1e-8, atol=1e-10)


def test_composed_backward_and_adam_update(tesseracts) -> None:
    controller, physics = tesseracts
    conditions = TRAINING_CONDITIONS[:2]
    theta0 = flatten_jumpgrad_parameters(
        build_jumpgrad_controller()
    ).detach().numpy()
    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    passive = deterministic_condition_objectives(conditions, "passive")
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    optimizer = torch.optim.Adam([theta], lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss, q = differentiable_loss(
        controller, physics, theta, conditions, tapes, passive
    )
    loss.backward()
    gradient = theta.grad.detach().numpy().copy()
    optimizer.step()
    assert q.shape == (2, 2)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
    assert not np.array_equal(theta.detach().numpy(), theta0)
