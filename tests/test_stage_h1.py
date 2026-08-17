from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from stochastic_stick_slip.model import (
    TRAINING_SEEDS,
    calibrate_baseline,
    evaluate_batch,
    forcing_history,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_forward_is_finite_and_switches_both_ways() -> None:
    result = evaluate_batch(calibrate_baseline(), TRAINING_SEEDS)
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert np.count_nonzero(np.asarray(result.stick_to_slip) > 0) >= 2
    assert np.count_nonzero(np.asarray(result.slip_to_stick) > 0) >= 2


def test_physics_apply_jvp_and_vjp_are_finite(tesseracts) -> None:
    physics, _ = tesseracts
    inputs = {"q": calibrate_baseline(), "seeds": TRAINING_SEEDS}
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
    assert np.all(np.isfinite(output["seed_losses"]))
    assert np.all(np.isfinite(jvp["seed_losses"]))
    assert np.all(np.isfinite(vjp["q"]))


def test_value_and_grad_crosses_two_local_tesseracts(tesseracts) -> None:
    physics, objective = tesseracts
    seeds = jnp.asarray(TRAINING_SEEDS)

    def pipeline(q):
        response = apply_tesseract(physics, {"q": q, "seeds": seeds})
        return apply_tesseract(
            objective, {"seed_losses": response["seed_losses"]}
        )["objective"]

    value, gradient = jax.value_and_grad(pipeline)(
        jnp.asarray(calibrate_baseline())
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
