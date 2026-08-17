"""Small stochastic stick-slip finite-element demonstration."""

from .model import (
    BASELINE_DAMPING,
    TRAINING_SEEDS,
    calibrate_baseline,
    crn_fd_jacobian,
    evaluate_batch,
)

__all__ = [
    "BASELINE_DAMPING",
    "TRAINING_SEEDS",
    "calibrate_baseline",
    "crn_fd_jacobian",
    "evaluate_batch",
]
