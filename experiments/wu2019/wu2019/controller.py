"""Published constant and harmonic normal-force histories."""

import numpy as np

from wu2019.dynamics import SimulationSettings


def phase_grid(settings: SimulationSettings) -> np.ndarray:
    steps = np.arange(1, settings.num_steps + 1, dtype=np.float64)
    return 2.0 * np.pi * steps / settings.steps_per_period


def constant_normal_force(
    value: float,
    settings: SimulationSettings,
) -> np.ndarray:
    if value < 0.0:
        raise ValueError("normal force must be non-negative")
    return np.full(settings.num_steps, value, dtype=np.float64)


def harmonic_normal_force(
    mean: float,
    amplitude: float,
    order: int,
    phase: float,
    settings: SimulationSettings,
) -> np.ndarray:
    normal_force = mean + amplitude * np.sin(
        order * phase_grid(settings) + phase
    )
    if np.min(normal_force) < -1e-12:
        raise ValueError("normal force history becomes negative")
    return normal_force
