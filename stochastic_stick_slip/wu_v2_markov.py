"""Wu-V2 hard two-state stochastic preload actuator for Gate A."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.model import STEPS_PER_PERIOD, TRAINING_SEEDS
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    MECHANICS_SIMULATOR,
    excitation_grid,
)


PRELOAD_LOW = 0.03
PRELOAD_HIGH = 0.05
FD_EPSILON = 0.02
MARKOV_BASE_SEED = 20260819
CONDITION_LABELS = TRAINING_SEEDS.copy()
NUM_CONDITIONS = len(CONDITION_LABELS)
NUM_REALIZATIONS = 4
NUM_STEPS = DIAGNOSTIC_NUM_PERIODS * STEPS_PER_PERIOD
LANDSCAPE_RADII = np.asarray([0.25, 0.50, 1.00], dtype=np.float64)
LANDSCAPE_PHASES = 2.0 * np.pi * np.arange(16, dtype=np.float64) / 16.0


def deterministic_binary_preload(
    omega: float,
    phase: float | np.ndarray,
    num_periods: int = DIAGNOSTIC_NUM_PERIODS,
) -> np.ndarray:
    """Quantize one or more deterministic two-omega commands to LOW/HIGH."""
    _, times = excitation_grid(omega, num_periods)
    phases = np.atleast_1d(np.asarray(phase, dtype=np.float64))
    if phases.ndim != 1:
        raise ValueError("phase must be a scalar or one-dimensional array")
    high = (
        np.sin(
            2.0 * float(omega) * times[None, :]
            + phases[:, None]
        )
        >= 0.0
    )
    scalar = np.where(high, PRELOAD_HIGH, PRELOAD_LOW)
    return np.repeat(scalar[:, :, None], 2, axis=2)


def deterministic_policy_hard_limit_preload(
    omega: float,
    q: np.ndarray,
    num_periods: int = DIAGNOSTIC_NUM_PERIODS,
) -> np.ndarray:
    """Return the deterministic HIGH state where the Markov score is nonnegative."""
    _, times = excitation_grid(omega, num_periods)
    coefficients = np.asarray(q, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    if coefficients.ndim != 2 or coefficients.shape[1] != 2:
        raise ValueError("q must have shape (2,) or (batch, 2)")
    argument = 2.0 * float(omega) * times[None, :]
    signal = (
        coefficients[:, 0, None] * np.cos(argument)
        + coefficients[:, 1, None] * np.sin(argument)
    )
    scalar = np.where(signal >= 0.0, PRELOAD_HIGH, PRELOAD_LOW)
    return np.repeat(scalar[:, :, None], 2, axis=2)


def landscape_polar_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the registered neutral plus three-radius, 16-phase grid."""
    radii = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.repeat(LANDSCAPE_RADII, 16))
    )
    phases = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.tile(LANDSCAPE_PHASES, 3))
    )
    q_values = np.column_stack(
        (radii * np.cos(phases), radii * np.sin(phases))
    )
    return q_values, radii, phases


def markov_uniform_bank(
    num_realizations: int = NUM_REALIZATIONS,
    stream_id: int = 0,
    iteration: int = 0,
    num_contacts: int = 2,
) -> np.ndarray:
    """Return deterministic [condition, realization, time+1, contact] tapes."""
    def seed_entropy(condition: int, realization: int, contact: int):
        entropy = [MARKOV_BASE_SEED, condition, realization, contact]
        if stream_id != 0 or iteration != 0:
            entropy.extend((stream_id, iteration))
        return entropy

    return np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            np.random.default_rng(
                                np.random.SeedSequence(
                                    seed_entropy(
                                        int(condition), realization, contact
                                    )
                                )
                            ).uniform(0.0, 1.0, size=NUM_STEPS + 1)
                            for contact in range(num_contacts)
                        ],
                        axis=-1,
                    )
                    for realization in range(num_realizations)
                ]
            )
            for condition in CONDITION_LABELS
        ]
    )


def policy_polar_coordinates(q: np.ndarray | jax.Array) -> tuple[float, float]:
    """Return magnitude and wrapped coefficient angle atan2(b2, a2)."""
    q = np.asarray(q, dtype=np.float64)
    magnitude = float(np.linalg.norm(q))
    phase = float(np.mod(np.arctan2(q[1], q[0]), 2.0 * np.pi))
    return magnitude, phase


def two_omega_signal(q: jax.Array, times: jax.Array, omega: jax.Array):
    """Return a2*cos(2*omega*t) + b2*sin(2*omega*t)."""
    q = jnp.asarray(q, dtype=jnp.float64)
    phase = 2.0 * omega * times
    return q[0] * jnp.cos(phase) + q[1] * jnp.sin(phase)


def transition_rates(q: jax.Array, times: jax.Array, omega: jax.Array):
    """Return the frozen LOW-to-HIGH and HIGH-to-LOW rates."""
    period = 2.0 * jnp.pi / omega
    lambda_0 = 4.0 / period
    signal = two_omega_signal(q, times, omega)
    return lambda_0 * jnp.exp(signal), lambda_0 * jnp.exp(-signal)


def transition_probabilities(
    q: jax.Array,
    times: jax.Array,
    omega: jax.Array,
    time_step: jax.Array,
):
    """Return exact per-step hard-jump probabilities."""
    rate_low_to_high, rate_high_to_low = transition_rates(q, times, omega)
    return (
        1.0 - jnp.exp(-rate_low_to_high * time_step),
        1.0 - jnp.exp(-rate_high_to_low * time_step),
    )


def generate_hard_preload_history(
    q: jax.Array,
    times: jax.Array,
    uniforms: jax.Array,
    omega: jax.Array,
    time_step: jax.Array,
):
    """Generate hard LOW/HIGH histories from explicit uniform tapes."""
    uniforms = jnp.asarray(uniforms, dtype=jnp.float64)
    probability_low_to_high, probability_high_to_low = (
        transition_probabilities(q, times, omega, time_step)
    )
    initial_high = uniforms[..., 0, :] < 0.5
    transition_uniforms = jnp.moveaxis(uniforms[..., 1:, :], -2, 0)

    def transition(current_high, inputs):
        current_uniform, probability_lh, probability_hl = inputs
        probability = jnp.where(
            current_high, probability_hl, probability_lh
        )
        next_high = jnp.logical_xor(
            current_high, current_uniform < probability
        )
        return next_high, next_high

    _, endpoint_modes = jax.lax.scan(
        transition,
        initial_high,
        (
            transition_uniforms,
            probability_low_to_high,
            probability_high_to_low,
        ),
    )
    modes = jnp.moveaxis(endpoint_modes, 0, -2)
    previous_modes = jnp.concatenate(
        (initial_high[..., None, :], modes[..., :-1, :]), axis=-2
    )
    transition_counts = jnp.sum(modes != previous_modes, axis=-2)
    high_mode_fraction = jnp.mean(modes, axis=-2, dtype=jnp.float64)
    preload = jnp.where(modes, PRELOAD_HIGH, PRELOAD_LOW)
    return modes, preload, transition_counts, high_mode_fraction


@jax.jit
def mechanics_forward(
    forcing: jax.Array,
    preload: jax.Array,
    time_step: jax.Array,
):
    """Run the frozen mechanics with no controller parameter input."""
    return MECHANICS_SIMULATOR(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        forcing,
        preload,
        time_step,
    )


def _simulate_markov_bank(
    q: jax.Array,
    forcing: jax.Array,
    uniforms: jax.Array,
    times: jax.Array,
    omega: jax.Array,
    time_step: jax.Array,
):
    modes, preload, transition_counts, high_mode_fraction = (
        generate_hard_preload_history(q, times, uniforms, omega, time_step)
    )
    leading_shape = preload.shape[:-2]
    forcing_bank = jnp.broadcast_to(
        forcing, leading_shape + (forcing.shape[0],)
    )
    outputs = mechanics_forward(
        forcing_bank.reshape((-1, forcing.shape[0])),
        preload.reshape((-1, preload.shape[-2], 2)),
        time_step,
    )
    reshaped_outputs = tuple(
        output.reshape(leading_shape + output.shape[1:]) for output in outputs
    )
    return (
        *reshaped_outputs,
        modes,
        preload,
        transition_counts,
        high_mode_fraction,
    )


SIMULATE_MARKOV_BANK = jax.jit(_simulate_markov_bank)


def steady_state_amplitude(displacement: jax.Array):
    """Return the mean half peak-to-peak amplitude over cycles 21-24."""
    displacement = jnp.asarray(displacement, dtype=jnp.float64)
    cycles = displacement.reshape(
        displacement.shape[:-1]
        + (DIAGNOSTIC_NUM_PERIODS, STEPS_PER_PERIOD)
    )
    amplitudes = 0.5 * (
        jnp.max(cycles, axis=-1) - jnp.min(cycles, axis=-1)
    )
    return jnp.mean(amplitudes[..., 20:24], axis=-1)


def stochastic_objective(
    q: jax.Array,
    forcing: jax.Array,
    uniforms: jax.Array,
    times: jax.Array,
    omega: jax.Array,
    time_step: jax.Array,
):
    """Return the 32-trajectory mean Wu steady-state amplitude."""
    displacement = SIMULATE_MARKOV_BANK(
        q, forcing, uniforms, times, omega, time_step
    )[0]
    return jnp.mean(steady_state_amplitude(displacement))


VALUE_AND_GRAD = jax.jit(jax.value_and_grad(stochastic_objective))


def evaluate_markov_bank(
    q: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array,
    omega: float | jax.Array,
    time_step: float | jax.Array,
) -> dict[str, jax.Array]:
    """Evaluate one explicit hard-Markov tape bank."""
    outputs = SIMULATE_MARKOV_BANK(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(omega, dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    (
        displacement,
        velocity,
        slip,
        stick_to_slip,
        slip_to_stick,
        modes,
        preload,
        transition_counts,
        high_mode_fraction,
    ) = outputs
    return {
        "trajectory_objectives": steady_state_amplitude(displacement),
        "displacement": displacement,
        "velocity": velocity,
        "slip": slip,
        "stick_to_slip": stick_to_slip,
        "slip_to_stick": slip_to_stick,
        "modes": modes,
        "preload": preload,
        "transition_counts": transition_counts,
        "high_mode_fraction": high_mode_fraction,
    }


def direct_ad_objective_and_gradient(
    q: np.ndarray | jax.Array,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array,
    omega: float | jax.Array,
    time_step: float | jax.Array,
) -> tuple[float, np.ndarray]:
    """Return raw JAX value-and-grad through the real hard pipeline."""
    value, gradient = VALUE_AND_GRAD(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        jnp.asarray(omega, dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    return float(value), np.asarray(gradient)


def fixed_history_objective(
    forcing: np.ndarray | jax.Array,
    preload: np.ndarray | jax.Array,
    time_step: float | jax.Array,
) -> float:
    """Replay mechanics from a fixed preload history, with no q argument."""
    outputs = mechanics_forward(
        jnp.asarray(forcing, dtype=jnp.float64)[None, :],
        jnp.asarray(preload, dtype=jnp.float64)[None, :, :],
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    return float(steady_state_amplitude(outputs[0])[0])


def crn_centered_fd(
    q: np.ndarray,
    forcing: np.ndarray | jax.Array,
    uniforms: np.ndarray | jax.Array,
    times: np.ndarray | jax.Array,
    omega: float | jax.Array,
    time_step: float | jax.Array,
) -> dict[str, np.ndarray | list[float] | list[int]]:
    """Return the two-coordinate same-tape centered finite difference."""
    q = np.asarray(q, dtype=np.float64)
    gradient = []
    plus_objectives = []
    minus_objectives = []
    mode_difference_counts = []
    for index in range(2):
        plus = q.copy()
        minus = q.copy()
        plus[index] += FD_EPSILON
        minus[index] -= FD_EPSILON
        plus_result = evaluate_markov_bank(
            plus, forcing, uniforms, times, omega, time_step
        )
        minus_result = evaluate_markov_bank(
            minus, forcing, uniforms, times, omega, time_step
        )
        plus_objective = float(
            np.mean(np.asarray(plus_result["trajectory_objectives"]))
        )
        minus_objective = float(
            np.mean(np.asarray(minus_result["trajectory_objectives"]))
        )
        gradient.append(
            (plus_objective - minus_objective) / (2.0 * FD_EPSILON)
        )
        plus_objectives.append(plus_objective)
        minus_objectives.append(minus_objective)
        mode_difference_counts.append(
            int(
                np.count_nonzero(
                    np.any(
                        np.asarray(plus_result["modes"])
                        != np.asarray(minus_result["modes"]),
                        axis=(-2, -1),
                    )
                )
            )
        )
    return {
        "gradient": np.asarray(gradient, dtype=np.float64),
        "plus_objectives": plus_objectives,
        "minus_objectives": minus_objectives,
        "mode_difference_counts": mode_difference_counts,
    }
