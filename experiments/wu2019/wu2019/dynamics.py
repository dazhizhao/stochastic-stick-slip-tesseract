"""Fixed dimensional parameters and small result types for Wu 2019."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WuParameters:
    mass: float = 1.0
    damping: float = 5.0
    stiffness: float = 3.55e4
    tangential_stiffness: float = 1.0e4
    forcing_amplitude: float = 10.0
    friction_coefficient: float = 0.3


@dataclass(frozen=True)
class SimulationSettings:
    steps_per_period: int = 400
    num_periods: int = 100
    measurement_periods: int = 20

    @property
    def num_steps(self) -> int:
        return self.steps_per_period * self.num_periods


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    slider_displacement: np.ndarray
    friction_force: np.ndarray
    normal_force: np.ndarray
    slip: np.ndarray


@dataclass(frozen=True)
class SummaryBatch:
    amplitude: np.ndarray
    previous_ten_amplitude: np.ndarray
    last_ten_amplitude: np.ndarray
    dissipated_energy: np.ndarray
    slip_fraction: np.ndarray
    transition_count: np.ndarray
    friction_excess: np.ndarray


DEFAULT_PARAMETERS = WuParameters()
DEFAULT_SETTINGS = SimulationSettings()


def display_frequency_grid() -> np.ndarray:
    broad = np.arange(140.0, 260.0 + 1e-12, 2.0)
    resonant = np.arange(190.0, 220.0 + 1e-12, 0.5)
    return np.unique(np.concatenate((broad, resonant)))


def dense_frequency_grid() -> np.ndarray:
    return np.arange(190.0, 220.0 + 1e-12, 0.25)


def sweep_frequency_grid() -> np.ndarray:
    return np.arange(190.0, 220.0 + 1e-12, 1.0)
