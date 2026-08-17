from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from stochastic_stick_slip.model import (
    SYSTEM,
    TRAINING_SEEDS,
    crn_fd_jacobian,
    forcing_history,
    select_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline():
    q, result = select_baseline()
    assert q is not None
    return q, result


@pytest.fixture(scope="module")
def tesseracts():
    physics = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/stick_slip_fem/tesseract_api.py"
    )
    objective = Tesseract.from_tesseract_api(
        ROOT / "tesseracts/stochastic_objective/tesseract_api.py"
    )
    return physics, objective


def test_fixed_seed_forcing_is_reproducible() -> None:
    first = forcing_history(int(TRAINING_SEEDS[0]))
    second = forcing_history(int(TRAINING_SEEDS[0]))
    assert np.array_equal(first, second)


def test_larger_forward_is_finite_and_both_contacts_switch(baseline) -> None:
    _, result = baseline
    complete_cycles = np.logical_and(
        np.asarray(result.stick_to_slip) > 0,
        np.asarray(result.slip_to_stick) > 0,
    )
    assert SYSTEM.num_free_dofs == 96
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert np.all(np.any(complete_cycles, axis=0))
    assert np.count_nonzero(np.any(complete_cycles, axis=1)) >= 4


def test_crn_gradient_is_finite(baseline) -> None:
    q, _ = baseline
    gradient = crn_fd_jacobian(q, TRAINING_SEEDS).mean(axis=0)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_physics_apply_jvp_and_vjp_are_finite(baseline, tesseracts) -> None:
    q, _ = baseline
    physics, _ = tesseracts
    inputs = {
        "q": q,
        "coeffs": np.zeros((8, 5), dtype=np.float64),
        "seeds": TRAINING_SEEDS,
    }
    output = physics.apply(inputs)
    jvp = physics.jacobian_vector_product(
        inputs,
        ["q"],
        ["seed_losses"],
        {"q": np.array([1.0, 0.0])},
    )
    vjp = physics.vector_jacobian_product(
        inputs,
        ["q"],
        ["seed_losses"],
        {"seed_losses": np.ones(8)},
    )
    assert output["stick_to_slip"].shape == (8, 2)
    assert np.all(np.isfinite(output["seed_losses"]))
    assert np.all(np.isfinite(jvp["seed_losses"]))
    assert np.all(np.isfinite(vjp["q"]))


def test_value_and_grad_crosses_two_local_tesseracts(baseline, tesseracts) -> None:
    q, _ = baseline
    physics, objective = tesseracts
    seeds = jnp.asarray(TRAINING_SEEDS)
    zero_coefficients = jnp.zeros((8, 5), dtype=jnp.float64)

    def pipeline(design):
        response = apply_tesseract(
            physics,
            {"q": design, "coeffs": zero_coefficients, "seeds": seeds},
        )
        return apply_tesseract(
            objective, {"seed_losses": response["seed_losses"]}
        )["objective"]

    value, gradient = jax.value_and_grad(pipeline)(jnp.asarray(q))
    direct_gradient = crn_fd_jacobian(q, TRAINING_SEEDS).mean(axis=0)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(gradient, direct_gradient, rtol=1e-10, atol=1e-12)
