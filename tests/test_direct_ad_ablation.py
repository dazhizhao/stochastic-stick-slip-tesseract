import numpy as np

from scripts.run_direct_ad_ablation import (
    direct_ad_batch_objective_and_gradient,
    production_batch_objective,
)
from scripts.run_stage_h4 import H4_TRAINING_SEEDS


def test_direct_ad_matches_hard_forward_and_has_finite_gradient() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    coefficients = np.zeros((8, 5), dtype=np.float64)
    direct_objective, gradient = direct_ad_batch_objective_and_gradient(
        coefficients, seeds
    )
    production_objective = production_batch_objective(coefficients, seeds)

    assert np.isclose(
        direct_objective, production_objective, rtol=1e-12, atol=1e-14
    )
    assert gradient.shape == (8, 5)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0
