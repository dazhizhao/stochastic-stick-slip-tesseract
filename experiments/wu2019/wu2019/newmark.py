"""Average-acceleration Newmark solver for the Wu 2019 SDOF model."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from wu2019.dynamics import (
    DEFAULT_PARAMETERS,
    SimulationSettings,
    SummaryBatch,
    Trajectory,
)
from wu2019.friction import select_contact_state


NEWMARK_BETA = 0.25
NEWMARK_GAMMA = 0.5


def _initial_newmark_state():
    return (
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(False),
    )


def _advance_newmark_state(
    state,
    external_force,
    current_normal_force,
    dt,
):
    parameters = DEFAULT_PARAMETERS
    mass = parameters.mass
    damping = parameters.damping
    stiffness = parameters.stiffness
    contact_stiffness = parameters.tangential_stiffness
    mu = parameters.friction_coefficient

    mass_term = mass / (NEWMARK_BETA * dt**2)
    damping_term = (
        damping * NEWMARK_GAMMA / (NEWMARK_BETA * dt)
    )
    effective_stiffness = stiffness + mass_term + damping_term

    displacement, velocity, acceleration, slider, was_slipping = state
    displacement_predictor = (
        displacement
        + dt * velocity
        + dt**2 * (0.5 - NEWMARK_BETA) * acceleration
    )
    velocity_predictor = (
        velocity + dt * (1.0 - NEWMARK_GAMMA) * acceleration
    )
    effective_rhs = (
        external_force
        + (mass_term + damping_term) * displacement_predictor
        - damping * velocity_predictor
    )

    stick_displacement = (
        effective_rhs + contact_stiffness * slider
    ) / (effective_stiffness + contact_stiffness)
    friction_limit = mu * current_normal_force
    trial_force, is_slipping, direction = select_contact_state(
        stick_displacement,
        slider,
        contact_stiffness,
        friction_limit,
    )
    slip_force = friction_limit * direction
    slip_displacement = (
        effective_rhs - slip_force
    ) / effective_stiffness

    next_displacement = jnp.where(
        is_slipping, slip_displacement, stick_displacement
    )
    friction_force = jnp.where(
        is_slipping, slip_force, trial_force
    )
    next_slider = jnp.where(
        is_slipping,
        next_displacement - friction_force / contact_stiffness,
        slider,
    )
    next_acceleration = (
        next_displacement - displacement_predictor
    ) / (NEWMARK_BETA * dt**2)
    next_velocity = velocity_predictor + (
        NEWMARK_GAMMA * dt * next_acceleration
    )
    dissipated_increment = friction_force * (next_slider - slider)
    transitioned = is_slipping != was_slipping
    next_state = (
        next_displacement,
        next_velocity,
        next_acceleration,
        next_slider,
        is_slipping,
    )
    output = (
        next_displacement,
        next_velocity,
        next_acceleration,
        next_slider,
        friction_force,
        is_slipping,
        dissipated_increment,
        transitioned,
    )
    return next_state, output


def _scan_trajectory(omega, normal_force, steps_per_period):
    parameters = DEFAULT_PARAMETERS
    dt = 2.0 * jnp.pi / (omega * steps_per_period)
    times = dt * jnp.arange(1, normal_force.shape[0] + 1, dtype=jnp.float64)
    forcing = parameters.forcing_amplitude * jnp.sin(omega * times)

    def step(state, step_inputs):
        external_force, current_normal_force = step_inputs
        return _advance_newmark_state(
            state,
            external_force,
            current_normal_force,
            dt,
        )

    _, outputs = jax.lax.scan(
        step, _initial_newmark_state(), (forcing, normal_force)
    )
    return times, outputs


@partial(
    jax.jit,
    static_argnames=("steps_per_period", "measurement_periods"),
)
def simulate_summary_raw(
    omega,
    normal_force,
    steps_per_period: int,
    measurement_periods: int,
):
    _, outputs = _scan_trajectory(
        omega, normal_force, steps_per_period
    )
    (
        displacement,
        _velocity,
        _acceleration,
        _slider,
        friction_force,
        slip,
        dissipated_increment,
        transitioned,
    ) = outputs
    num_periods = normal_force.shape[0] // steps_per_period
    period_maxima = displacement.reshape(
        (num_periods, steps_per_period)
    ).max(axis=1)
    measured_maxima = period_maxima[-measurement_periods:]
    amplitude = jnp.mean(measured_maxima)
    previous_ten = jnp.mean(measured_maxima[-20:-10])
    last_ten = jnp.mean(measured_maxima[-10:])
    measurement_steps = measurement_periods * steps_per_period
    measured_slice = slice(-measurement_steps, None)
    friction_limit = (
        DEFAULT_PARAMETERS.friction_coefficient * normal_force
    )
    return (
        amplitude,
        previous_ten,
        last_ten,
        jnp.sum(dissipated_increment[measured_slice]),
        jnp.mean(slip[measured_slice], dtype=jnp.float64),
        jnp.sum(transitioned[measured_slice], dtype=jnp.int64),
        jnp.max(jnp.abs(friction_force) - friction_limit),
    )


@partial(
    jax.jit,
    static_argnames=("steps_per_period", "measurement_periods"),
)
def _simulate_summary_batch_raw(
    omegas,
    normal_force_histories,
    steps_per_period: int,
    measurement_periods: int,
):
    return jax.vmap(
        lambda omega, normal_force: simulate_summary_raw(
            omega,
            normal_force,
            steps_per_period,
            measurement_periods,
        )
    )(omegas, normal_force_histories)


def simulate_summary_batch(
    omegas,
    normal_force_histories,
    settings: SimulationSettings,
) -> SummaryBatch:
    omegas = np.asarray(omegas, dtype=np.float64)
    normal_force_histories = np.asarray(
        normal_force_histories, dtype=np.float64
    )
    if normal_force_histories.ndim == 1:
        normal_force_histories = np.broadcast_to(
            normal_force_histories,
            (len(omegas), settings.num_steps),
        )
    if normal_force_histories.shape != (len(omegas), settings.num_steps):
        raise ValueError("normal-force histories have the wrong shape")
    if not np.all(np.isfinite(normal_force_histories)):
        raise ValueError("normal-force histories must be finite")
    if np.min(normal_force_histories) < -1e-12:
        raise ValueError("normal force must be non-negative")
    outputs = _simulate_summary_batch_raw(
        jnp.asarray(omegas),
        jnp.asarray(normal_force_histories),
        settings.steps_per_period,
        settings.measurement_periods,
    )
    arrays = [np.asarray(value) for value in outputs]
    return SummaryBatch(*arrays)


@partial(jax.jit, static_argnames=("steps_per_period",))
def _simulate_trajectory_jit(omega, normal_force, steps_per_period: int):
    return _scan_trajectory(omega, normal_force, steps_per_period)


def simulate_trajectory(
    omega: float,
    normal_force,
    settings: SimulationSettings,
) -> Trajectory:
    normal_force = np.asarray(normal_force, dtype=np.float64)
    if normal_force.shape != (settings.num_steps,):
        raise ValueError("normal-force history has the wrong shape")
    times, outputs = _simulate_trajectory_jit(
        jnp.asarray(omega, dtype=jnp.float64),
        jnp.asarray(normal_force),
        settings.steps_per_period,
    )
    displacement, velocity, acceleration, slider, friction, slip = outputs[:6]
    return Trajectory(
        time=np.asarray(times),
        displacement=np.asarray(displacement),
        velocity=np.asarray(velocity),
        acceleration=np.asarray(acceleration),
        slider_displacement=np.asarray(slider),
        friction_force=np.asarray(friction),
        normal_force=normal_force.copy(),
        slip=np.asarray(slip),
    )
