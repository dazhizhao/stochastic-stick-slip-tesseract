"""Small stochastic stick-slip finite-element demonstration."""

from .model import (
    BASELINE_DAMPING,
    TRAINING_SEEDS,
    crn_fd_jacobian,
    evaluate_batch,
    select_baseline,
)

__all__ = [
    "BASELINE_DAMPING",
    "TRAINING_SEEDS",
    "crn_fd_jacobian",
    "evaluate_batch",
    "select_baseline",
]
