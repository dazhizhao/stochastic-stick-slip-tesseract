"""JAX-FEM cantilever dynamics with a hard Jenkins friction element."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem


TRAINING_SEEDS = np.array([11, 23, 37, 41, 53, 67, 79, 97], dtype=np.int64)
BASELINE_DAMPING = 0.2
FRICTION_COEFFICIENT = 0.4
CONTACT_STIFFNESS = 0.2
FD_RELATIVE_EPSILON = 0.05

YOUNG_MODULUS = 1000.0
POISSON_RATIO = 0.3
DENSITY = 1.0
BEAM_LENGTH = 1.0
BEAM_HEIGHT = 0.1
NUM_ELEMENTS_X = 4
NUM_ELEMENTS_Y = 1
STEPS_PER_PERIOD = 100
NUM_PERIODS = 8
NUM_STEPS = STEPS_PER_PERIOD * NUM_PERIODS
FORCING_AMPLITUDE = 0.02
PRELOAD_QUANTILE = 0.6


class _StiffnessProblem(Problem):
    def get_tensor_map(self):
        shear = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        lame = (
            YOUNG_MODULUS
            * POISSON_RATIO
            / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
        )

        def stress(displacement_gradient):
            strain = 0.5 * (
                displacement_gradient + displacement_gradient.T
            )
            return (
                lame * jnp.trace(strain) * jnp.eye(self.dim)
                + 2.0 * shear * strain
            )

        return stress


class _MassProblem(Problem):
    def get_mass_map(self):
        def mass(displacement, _point):
            return DENSITY * displacement

        return mass


class _UnitLoadProblem(Problem):
    def get_mass_map(self):
        def zero_volume(displacement, _point):
            return jnp.zeros_like(displacement)

        return zero_volume

    def get_surface_maps(self):
        def downward_traction(_displacement, _point):
            return jnp.array([0.0, -1.0])

        return [downward_traction]


@dataclass(frozen=True)
class FEMSystem:
    stiffness: jax.Array
    mass: jax.Array
    load: jax.Array
    observation: jax.Array
    omega_1: float
    time_step: float
    times: jax.Array


@dataclass(frozen=True)
class BatchResult:
    losses: jax.Array
    displacement: jax.Array
    velocity: jax.Array
    slip: jax.Array
    stick_to_slip: jax.Array
    slip_to_stick: jax.Array


def _assemble_tangent(problem: Problem) -> np.ndarray:
    zero_solution = [
        jnp.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
    ]
    flat_zero, unflatten = jax.flatten_util.ravel_pytree(zero_solution)

    def residual(flat_solution):
        solution = unflatten(flat_solution)
        return jax.flatten_util.ravel_pytree(
            problem.compute_residual(solution)
        )[0]

    return np.asarray(jax.jacfwd(residual)(flat_zero))


def _assemble_system() -> FEMSystem:
    meshio_mesh = rectangle_mesh(
        Nx=NUM_ELEMENTS_X,
        Ny=NUM_ELEMENTS_Y,
        domain_x=BEAM_LENGTH,
        domain_y=BEAM_HEIGHT,
    )
    cell_type = get_meshio_cell_type("QUAD4")
    mesh = Mesh(
        meshio_mesh.points,
        meshio_mesh.cells_dict[cell_type],
        ele_type="QUAD4",
    )

    def right(point):
        return jnp.isclose(point[0], BEAM_LENGTH, atol=1e-8)

    common = dict(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="QUAD4",
        quadrature_order=2,
    )
    stiffness_problem = _StiffnessProblem(**common)
    mass_problem = _MassProblem(**common)
    load_problem = _UnitLoadProblem(**common, location_fns=[right])

    stiffness_full = _assemble_tangent(stiffness_problem)
    mass_full = _assemble_tangent(mass_problem)
    zero_solution = [jnp.zeros((len(mesh.points), 2))]
    surface_residual = np.asarray(
        jax.flatten_util.ravel_pytree(
            load_problem.compute_residual(zero_solution)
        )[0]
    )

    points = np.asarray(mesh.points)
    left_nodes = np.flatnonzero(np.isclose(points[:, 0], 0.0))
    fixed_dofs = np.sort(
        np.concatenate((2 * left_nodes, 2 * left_nodes + 1))
    )
    all_dofs = np.arange(stiffness_full.shape[0])
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    right_nodes = np.flatnonzero(np.isclose(points[:, 0], BEAM_LENGTH))
    observation_full = np.zeros(stiffness_full.shape[0])
    observation_full[2 * right_nodes + 1] = 1.0 / len(right_nodes)

    stiffness = stiffness_full[np.ix_(free_dofs, free_dofs)]
    mass = mass_full[np.ix_(free_dofs, free_dofs)]
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)

    load_full = surface_residual / surface_residual[1::2].sum()
    load = load_full[free_dofs]
    observation = observation_full[free_dofs]

    eigenvalues = scipy.linalg.eigh(stiffness, mass, eigvals_only=True)
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    omega_1 = float(np.sqrt(positive_eigenvalues[0]))
    period = 2.0 * np.pi / omega_1
    time_step = period / STEPS_PER_PERIOD
    times = time_step * jnp.arange(1, NUM_STEPS + 1, dtype=jnp.float64)

    return FEMSystem(
        stiffness=jnp.asarray(stiffness),
        mass=jnp.asarray(mass),
        load=jnp.asarray(load),
        observation=jnp.asarray(observation),
        omega_1=omega_1,
        time_step=time_step,
        times=times,
    )


SYSTEM = _assemble_system()


def forcing_history(seed: int) -> np.ndarray:
    """Return the complete deterministic force history for one seed."""
    generator = np.random.default_rng(int(seed))
    amplitudes = generator.uniform(0.85, 1.15, size=2)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=2)
    times = np.asarray(SYSTEM.times)
    return FORCING_AMPLITUDE * (
        amplitudes[0]
        * np.sin(0.9 * SYSTEM.omega_1 * times + phases[0])
        + 0.6
        * amplitudes[1]
        * np.sin(1.35 * SYSTEM.omega_1 * times + phases[1])
    )


def forcing_batch(seeds: np.ndarray) -> jax.Array:
    return jnp.asarray(np.stack([forcing_history(int(seed)) for seed in seeds]))


def _simulate_seed(q: jax.Array, forcing: jax.Array):
    damping, preload = q
    dt = SYSTEM.time_step
    mass = SYSTEM.mass
    stiffness = SYSTEM.stiffness
    load = SYSTEM.load
    observation = SYSTEM.observation
    friction_limit = FRICTION_COEFFICIENT * preload

    effective_matrix = stiffness + mass / dt**2 + damping * mass / dt
    stick_matrix = effective_matrix + CONTACT_STIFFNESS * jnp.outer(
        load, observation
    )
    zero_displacement = jnp.zeros(stiffness.shape[0], dtype=jnp.float64)
    initial_state = (
        zero_displacement,
        zero_displacement,
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array(False),
    )

    def step(state, external_force):
        previous, previous_previous, slider_position, was_slipping = state
        history = (
            mass @ (2.0 * previous - previous_previous) / dt**2
            + damping * mass @ previous / dt
        )

        stick_solution = jnp.linalg.solve(
            stick_matrix,
            history
            + load
            * (external_force + CONTACT_STIFFNESS * slider_position),
        )
        stick_displacement = observation @ stick_solution
        required_friction = -CONTACT_STIFFNESS * (
            stick_displacement - slider_position
        )
        is_sticking = jnp.abs(required_friction) <= friction_limit

        slip_direction = jnp.sign(stick_displacement - slider_position)
        slip_direction = jnp.where(slip_direction == 0.0, 1.0, slip_direction)
        slip_friction = -friction_limit * slip_direction
        slip_solution = jnp.linalg.solve(
            effective_matrix,
            history + load * (external_force + slip_friction),
        )

        displacement_vector = jnp.where(
            is_sticking, stick_solution, slip_solution
        )
        displacement = observation @ displacement_vector
        previous_displacement = observation @ previous
        velocity = (displacement - previous_displacement) / dt
        is_slipping = jnp.logical_not(is_sticking)
        friction = jnp.where(is_sticking, required_friction, slip_friction)
        next_slider_position = jnp.where(
            is_sticking,
            slider_position,
            displacement + friction / CONTACT_STIFFNESS,
        )
        stick_to_slip = jnp.logical_and(
            jnp.logical_not(was_slipping), is_slipping
        )
        slip_to_stick = jnp.logical_and(
            was_slipping, jnp.logical_not(is_slipping)
        )

        next_state = (
            displacement_vector,
            previous,
            next_slider_position,
            is_slipping,
        )
        output = (
            displacement,
            velocity,
            is_slipping,
            stick_to_slip,
            slip_to_stick,
        )
        return next_state, output

    _, outputs = jax.lax.scan(step, initial_state, forcing)
    return outputs


_simulate_batch = jax.jit(jax.vmap(_simulate_seed, in_axes=(None, 0)))


def evaluate_batch(q: np.ndarray | jax.Array, seeds: np.ndarray) -> BatchResult:
    """Evaluate the hard forward response for a fixed seed batch."""
    displacement, velocity, slip, stick_to_slip, slip_to_stick = (
        _simulate_batch(jnp.asarray(q, dtype=jnp.float64), forcing_batch(seeds))
    )
    losses = jnp.mean(displacement**2, axis=1)
    return BatchResult(
        losses=losses,
        displacement=displacement,
        velocity=velocity,
        slip=slip,
        stick_to_slip=jnp.sum(stick_to_slip, axis=1),
        slip_to_stick=jnp.sum(slip_to_stick, axis=1),
    )


def calibrate_baseline(seeds: np.ndarray = TRAINING_SEEDS) -> np.ndarray:
    """Choose one preload from a single friction-free reference response."""
    reference = evaluate_batch(
        np.array([BASELINE_DAMPING, 0.0], dtype=np.float64), seeds
    )
    trial_force = CONTACT_STIFFNESS * np.abs(
        np.asarray(reference.displacement)
    )
    preload = float(np.quantile(trial_force, PRELOAD_QUANTILE))
    preload /= FRICTION_COEFFICIENT
    return np.array([BASELINE_DAMPING, preload], dtype=np.float64)


def crn_fd_jacobian(
    q: np.ndarray | jax.Array,
    seeds: np.ndarray,
    epsilon_multiplier: float = 1.0,
) -> np.ndarray:
    """Return d(seed_losses)/dq using centered FD and common random numbers."""
    q_array = np.asarray(q, dtype=np.float64)
    epsilons = FD_RELATIVE_EPSILON * q_array * epsilon_multiplier
    columns = []
    for index, epsilon in enumerate(epsilons):
        plus = q_array.copy()
        minus = q_array.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_losses = np.asarray(evaluate_batch(plus, seeds).losses)
        minus_losses = np.asarray(evaluate_batch(minus, seeds).losses)
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)
