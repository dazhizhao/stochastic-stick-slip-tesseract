"""JAX-FEM cantilever dynamics with coupled hard Jenkins contacts."""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem


TRAINING_SEEDS = np.array([11, 23, 37, 41, 53, 67, 79, 97], dtype=np.int64)
HELD_OUT_SEEDS = np.array(
    [101, 103, 107, 109, 113, 127, 131, 137], dtype=np.int64
)
BASELINE_DAMPING = 0.2
BASELINE_PRELOADS = (0.04, 0.02, 0.06)
FRICTION_COEFFICIENT = 0.4
CONTACT_STIFFNESS = 0.2
FD_RELATIVE_EPSILON = 0.05
COEFFICIENT_FD_EPSILON = 0.02
PRELOAD_MODULATION = 0.02
NUM_FOURIER_COEFFICIENTS = 5

YOUNG_MODULUS = 1000.0
POISSON_RATIO = 0.3
DENSITY = 1.0
BEAM_LENGTH = 1.0
BEAM_HEIGHT = 0.1
NUM_ELEMENTS_X = 16
NUM_ELEMENTS_Y = 2
CONTACT_COLUMNS = (11, 15)
STEPS_PER_PERIOD = 100
NUM_PERIODS = 8
NUM_STEPS = STEPS_PER_PERIOD * NUM_PERIODS
FORCING_AMPLITUDE = 0.02

# Zero denotes STICK; +/-1 denote the two possible SLIP directions.  Ordering
# candidates by slip count gives deterministic priority to sticking at equality.
CONTACT_REGIMES = jnp.array(
    [
        [0, 0],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ],
    dtype=jnp.int64,
)


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
    contacts: jax.Array
    omega_1: float
    time_step: float
    times: jax.Array
    points: np.ndarray
    cells: np.ndarray
    contact_coordinates: np.ndarray
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    contact_nodes: np.ndarray
    num_total_dofs: int
    num_free_dofs: int


@dataclass(frozen=True)
class BatchResult:
    losses: jax.Array
    displacement: jax.Array
    velocity: jax.Array
    slip: jax.Array
    stick_to_slip: jax.Array
    slip_to_stick: jax.Array


@dataclass(frozen=True)
class TrajectoryResult:
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


def _assemble_system(
    num_elements_x: int = NUM_ELEMENTS_X,
    num_elements_y: int = NUM_ELEMENTS_Y,
    contact_columns: tuple[int, ...] = CONTACT_COLUMNS,
) -> FEMSystem:
    meshio_mesh = rectangle_mesh(
        Nx=num_elements_x,
        Ny=num_elements_y,
        domain_x=BEAM_LENGTH,
        domain_y=BEAM_HEIGHT,
    )
    cell_type = get_meshio_cell_type("QUAD4")
    cells = meshio_mesh.cells_dict[cell_type]
    mesh = Mesh(meshio_mesh.points, cells, ele_type="QUAD4")

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

    contact_x = (
        np.asarray(contact_columns) * BEAM_LENGTH / num_elements_x
    )
    contact_nodes = []
    for x_coordinate in contact_x:
        matches = np.flatnonzero(
            np.logical_and(
                np.isclose(points[:, 0], x_coordinate),
                np.isclose(points[:, 1], 0.0),
            )
        )
        contact_nodes.append(int(matches[0]))
    num_contacts = len(contact_nodes)
    contacts_full = np.zeros((stiffness_full.shape[0], num_contacts))
    contacts_full[
        2 * np.asarray(contact_nodes) + 1, np.arange(num_contacts)
    ] = 1.0

    stiffness = stiffness_full[np.ix_(free_dofs, free_dofs)]
    mass = mass_full[np.ix_(free_dofs, free_dofs)]
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)

    load_full = surface_residual / surface_residual[1::2].sum()
    load = load_full[free_dofs]
    observation = observation_full[free_dofs]
    contacts = contacts_full[free_dofs]

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
        contacts=jnp.asarray(contacts),
        omega_1=omega_1,
        time_step=time_step,
        times=times,
        points=points[:, :2],
        cells=np.asarray(cells),
        contact_coordinates=points[np.asarray(contact_nodes), :2],
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
        contact_nodes=np.asarray(contact_nodes),
        num_total_dofs=stiffness_full.shape[0],
        num_free_dofs=len(free_dofs),
    )


SYSTEM = _assemble_system()


def forcing_parameters(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the amplitudes and phases that completely define one forcing."""
    generator = np.random.default_rng(int(seed))
    amplitudes = generator.uniform(0.85, 1.15, size=2)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=2)
    return amplitudes, phases


def forcing_history_for_system(seed: int, system: FEMSystem) -> np.ndarray:
    """Return the complete deterministic force history for one seed."""
    amplitudes, phases = forcing_parameters(seed)
    times = np.asarray(system.times)
    return FORCING_AMPLITUDE * (
        amplitudes[0]
        * np.sin(0.9 * system.omega_1 * times + phases[0])
        + 0.6
        * amplitudes[1]
        * np.sin(1.35 * system.omega_1 * times + phases[1])
    )


def forcing_history(seed: int) -> np.ndarray:
    return forcing_history_for_system(seed, SYSTEM)


def forcing_batch_for_system(seeds: np.ndarray, system: FEMSystem) -> jax.Array:
    return jnp.asarray(
        np.stack(
            [forcing_history_for_system(int(seed), system) for seed in seeds]
        )
    )


def forcing_batch(seeds: np.ndarray) -> jax.Array:
    return forcing_batch_for_system(seeds, SYSTEM)


def forcing_descriptor(seed: int) -> np.ndarray:
    """Return the six normalized forcing features used by the controller."""
    amplitudes, phases = forcing_parameters(seed)
    return np.array(
        [
            (amplitudes[0] - 1.0) / 0.15,
            (amplitudes[1] - 1.0) / 0.15,
            np.sin(phases[0]),
            np.cos(phases[0]),
            np.sin(phases[1]),
            np.cos(phases[1]),
        ],
        dtype=np.float64,
    )


def forcing_descriptor_batch(seeds: np.ndarray) -> np.ndarray:
    return np.stack([forcing_descriptor(int(seed)) for seed in seeds])


def build_fourier_basis(system: FEMSystem) -> jax.Array:
    return jnp.stack(
        (
            jnp.ones(NUM_STEPS, dtype=jnp.float64),
            jnp.cos(0.9 * system.omega_1 * system.times),
            jnp.sin(0.9 * system.omega_1 * system.times),
            jnp.cos(1.35 * system.omega_1 * system.times),
            jnp.sin(1.35 * system.omega_1 * system.times),
        ),
        axis=1,
    )


FOURIER_BASIS = build_fourier_basis(SYSTEM)


def preload_history_with_basis(
    base_preload: float | jax.Array,
    coefficients: np.ndarray | jax.Array,
    fourier_basis: jax.Array,
) -> jax.Array:
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    signal = coefficients @ fourier_basis.T
    return base_preload + PRELOAD_MODULATION * jnp.tanh(signal)


def preload_history(
    base_preload: float | jax.Array,
    coefficients: np.ndarray | jax.Array,
) -> jax.Array:
    """Map the five fixed Fourier coefficients to a bounded preload history."""
    return preload_history_with_basis(base_preload, coefficients, FOURIER_BASIS)


def _factor_solve(cholesky_factor: jax.Array, right_hand_side: jax.Array):
    intermediate = jax.scipy.linalg.solve_triangular(
        cholesky_factor, right_hand_side, lower=True
    )
    return jax.scipy.linalg.solve_triangular(
        cholesky_factor.T, intermediate, lower=False
    )


def _select_contact_regime(
    free_contact_displacement,
    slider_position,
    contact_compliance,
    friction_limit,
):
    identity = jnp.eye(2, dtype=jnp.float64)
    force_tolerance = 1e-11 * (1.0 + friction_limit)
    displacement_tolerance = force_tolerance / CONTACT_STIFFNESS

    def candidate(regime):
        sticking = regime == 0
        matrix = identity + (
            sticking[:, None] * CONTACT_STIFFNESS * contact_compliance
        )
        right_hand_side = jnp.where(
            sticking,
            -CONTACT_STIFFNESS
            * (free_contact_displacement - slider_position),
            -friction_limit * regime,
        )
        contact_force = jnp.linalg.solve(matrix, right_hand_side)
        contact_displacement = (
            free_contact_displacement + contact_compliance @ contact_force
        )
        relative_displacement = contact_displacement - slider_position
        elastic_force = -CONTACT_STIFFNESS * relative_displacement
        valid_contact = jnp.where(
            sticking,
            jnp.abs(elastic_force) <= friction_limit + force_tolerance,
            regime * relative_displacement
            >= friction_limit / CONTACT_STIFFNESS - displacement_tolerance,
        )
        return contact_force, contact_displacement, jnp.all(valid_contact)

    forces, displacements, valid = jax.vmap(candidate)(CONTACT_REGIMES)
    selected = jnp.argmax(valid.astype(jnp.int64))
    any_valid = jnp.any(valid)
    contact_force = jnp.where(
        any_valid, forces[selected], jnp.full(2, jnp.nan)
    )
    contact_displacement = jnp.where(
        any_valid, displacements[selected], jnp.full(2, jnp.nan)
    )
    regime = jnp.where(any_valid, CONTACT_REGIMES[selected], jnp.ones(2))
    return contact_force, contact_displacement, regime


def _select_contact_regime_box(
    free_contact_displacement,
    slider_position,
    contact_compliance,
    friction_limit,
):
    """Solve the same Jenkins law for more than two coupled contacts."""
    force_tolerance = 1e-11 * (1.0 + friction_limit)

    def project(_iteration, contact_force):
        relative_displacement = (
            free_contact_displacement
            + contact_compliance @ contact_force
            - slider_position
        )
        elastic_force = -CONTACT_STIFFNESS * relative_displacement
        return jnp.clip(elastic_force, -friction_limit, friction_limit)

    contact_force = jax.lax.fori_loop(
        0,
        8,
        project,
        jnp.zeros_like(free_contact_displacement),
    )
    contact_displacement = (
        free_contact_displacement + contact_compliance @ contact_force
    )
    relative_displacement = contact_displacement - slider_position
    elastic_force = -CONTACT_STIFFNESS * relative_displacement
    projected_force = jnp.clip(
        elastic_force, -friction_limit, friction_limit
    )
    converged = jnp.all(
        jnp.abs(contact_force - projected_force) <= force_tolerance
    )
    contact_force = jnp.where(
        converged, contact_force, jnp.full_like(contact_force, jnp.nan)
    )
    contact_displacement = jnp.where(
        converged,
        contact_displacement,
        jnp.full_like(contact_displacement, jnp.nan),
    )
    sticking = jnp.abs(elastic_force) <= friction_limit + force_tolerance
    regime = jnp.where(sticking, 0, jnp.sign(relative_displacement)).astype(
        jnp.int64
    )
    regime = jnp.where(converged, regime, jnp.ones_like(regime))
    return contact_force, contact_displacement, regime


def _build_mechanics_batch_impl(
    system: FEMSystem,
    return_full_displacement: bool,
    return_friction_work: bool,
):
    def simulate_batch_impl(
        damping: jax.Array,
        forcing: jax.Array,
        preload: jax.Array,
        time_step: jax.Array,
    ):
        dt = time_step
        mass = system.mass
        stiffness = system.stiffness
        load = system.load
        observation = system.observation
        contacts = system.contacts

        effective_matrix = stiffness + mass / dt**2 + damping * mass / dt
        cholesky_factor = jnp.linalg.cholesky(effective_matrix)
        contact_response = _factor_solve(cholesky_factor, contacts)
        contact_compliance = contacts.T @ contact_response
        zero_displacement = jnp.zeros(stiffness.shape[0], dtype=jnp.float64)
        num_contacts = contacts.shape[1]
        select_contact_regime = (
            _select_contact_regime
            if num_contacts == 2
            else _select_contact_regime_box
        )

        def simulate_seed(seed_forcing, seed_preload):
            initial_state = (
                zero_displacement,
                zero_displacement,
                jnp.zeros(num_contacts, dtype=jnp.float64),
                jnp.zeros(num_contacts, dtype=jnp.bool_),
            )

            def step(state, step_inputs):
                external_force, current_preload = step_inputs
                previous, previous_previous, slider_position, was_slipping = state
                history = (
                    mass @ (2.0 * previous - previous_previous) / dt**2
                    + damping * mass @ previous / dt
                )
                free_solution = _factor_solve(
                    cholesky_factor, history + load * external_force
                )
                free_contact_displacement = contacts.T @ free_solution
                contact_force, contact_displacement, regime = (
                    select_contact_regime(
                        free_contact_displacement,
                        slider_position,
                        contact_compliance,
                        FRICTION_COEFFICIENT * current_preload,
                    )
                )
                displacement_vector = (
                    free_solution + contact_response @ contact_force
                )
                displacement = observation @ displacement_vector
                previous_displacement = observation @ previous
                velocity = (displacement - previous_displacement) / dt
                is_slipping = regime != 0
                next_slider_position = jnp.where(
                    is_slipping,
                    contact_displacement + contact_force / CONTACT_STIFFNESS,
                    slider_position,
                )
                dissipated_friction_work = jnp.abs(
                    contact_force * (next_slider_position - slider_position)
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
                if return_full_displacement:
                    output = output + (displacement_vector,)
                if return_friction_work:
                    output = output + (dissipated_friction_work,)
                return next_state, output

            _, outputs = jax.lax.scan(
                step, initial_state, (seed_forcing, seed_preload)
            )
            return outputs

        return jax.vmap(simulate_seed)(forcing, preload)

    return simulate_batch_impl


def build_mechanics_batch_simulator(
    system: FEMSystem,
    return_full_displacement: bool = False,
    return_friction_work: bool = False,
):
    """Build hard mechanics driven only by forcing and contact preloads."""
    implementation = _build_mechanics_batch_impl(
        system, return_full_displacement, return_friction_work
    )

    def simulate_batch(damping, forcing, preload):
        return implementation(
            damping,
            forcing,
            preload,
            jnp.asarray(system.time_step, dtype=jnp.float64),
        )

    return jax.jit(simulate_batch)


def build_variable_time_step_mechanics_batch_simulator(
    system: FEMSystem,
    return_full_displacement: bool = False,
    return_friction_work: bool = False,
):
    """Build the same hard mechanics with an explicit scalar time step."""
    return jax.jit(
        _build_mechanics_batch_impl(
            system, return_full_displacement, return_friction_work
        )
    )


def build_batch_simulator(system: FEMSystem, fourier_basis: jax.Array):
    """Build the legacy continuous-preload simulator."""
    mechanics = build_mechanics_batch_simulator(system)

    def simulate_batch_impl(
        q: jax.Array,
        coefficients: jax.Array,
        forcing: jax.Array,
    ):
        damping, base_preload = q
        shared_preload = preload_history_with_basis(
            base_preload, coefficients, fourier_basis
        )
        per_contact_preload = jnp.repeat(
            shared_preload[..., None], system.contacts.shape[1], axis=-1
        )
        return mechanics(damping, forcing, per_contact_preload)

    return jax.jit(simulate_batch_impl)


_simulate_batch = build_batch_simulator(SYSTEM, FOURIER_BASIS)


def build_trajectory_simulator(system: FEMSystem, fourier_basis: jax.Array):
    def simulate_trajectory_impl(q, coefficients, seed_forcing):
        damping, base_preload = q
        dt = system.time_step
        mass = system.mass
        stiffness = system.stiffness
        load = system.load
        contacts = system.contacts
        seed_preload = preload_history_with_basis(
            base_preload, coefficients[None, :], fourier_basis
        )[0]

        effective_matrix = stiffness + mass / dt**2 + damping * mass / dt
        cholesky_factor = jnp.linalg.cholesky(effective_matrix)
        contact_response = _factor_solve(cholesky_factor, contacts)
        contact_compliance = contacts.T @ contact_response
        zero_displacement = jnp.zeros(stiffness.shape[0], dtype=jnp.float64)
        num_contacts = contacts.shape[1]
        select_contact_regime = (
            _select_contact_regime
            if num_contacts == 2
            else _select_contact_regime_box
        )
        initial_state = (
            zero_displacement,
            zero_displacement,
            jnp.zeros(num_contacts, dtype=jnp.float64),
            jnp.zeros(num_contacts, dtype=jnp.bool_),
        )

        def step(state, step_inputs):
            external_force, current_preload = step_inputs
            previous, previous_previous, slider_position, was_slipping = state
            history = (
                mass @ (2.0 * previous - previous_previous) / dt**2
                + damping * mass @ previous / dt
            )
            free_solution = _factor_solve(
                cholesky_factor, history + load * external_force
            )
            free_contact_displacement = contacts.T @ free_solution
            contact_force, contact_displacement, regime = (
                select_contact_regime(
                    free_contact_displacement,
                    slider_position,
                    contact_compliance,
                    FRICTION_COEFFICIENT * current_preload,
                )
            )
            displacement = free_solution + contact_response @ contact_force
            velocity = (displacement - previous) / dt
            is_slipping = regime != 0
            next_slider_position = jnp.where(
                is_slipping,
                contact_displacement + contact_force / CONTACT_STIFFNESS,
                slider_position,
            )
            stick_to_slip = jnp.logical_and(
                jnp.logical_not(was_slipping), is_slipping
            )
            slip_to_stick = jnp.logical_and(
                was_slipping, jnp.logical_not(is_slipping)
            )
            next_state = (
                displacement,
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

        _, outputs = jax.lax.scan(
            step, initial_state, (seed_forcing, seed_preload)
        )
        return outputs

    return jax.jit(simulate_trajectory_impl)


_simulate_trajectory = build_trajectory_simulator(SYSTEM, FOURIER_BASIS)


def evaluate_controlled_batch_for_system(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
    system: FEMSystem,
    simulator,
) -> BatchResult:
    displacement, velocity, slip, stick_to_slip, slip_to_stick = simulator(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(coefficients, dtype=jnp.float64),
        forcing_batch_for_system(seeds, system),
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


def evaluate_controlled_batch(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
) -> BatchResult:
    """Evaluate the hard response with one Fourier control per seed."""
    return evaluate_controlled_batch_for_system(
        q,
        coefficients,
        seeds,
        SYSTEM,
        _simulate_batch,
    )


def evaluate_trajectory_for_system(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seed: int,
    system: FEMSystem,
    simulator,
) -> TrajectoryResult:
    outputs = simulator(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(forcing_history_for_system(seed, system)),
    )
    return TrajectoryResult(
        displacement=outputs[0],
        velocity=outputs[1],
        slip=outputs[2],
        stick_to_slip=jnp.sum(outputs[3], axis=0),
        slip_to_stick=jnp.sum(outputs[4], axis=0),
    )


def evaluate_batch(q: np.ndarray | jax.Array, seeds: np.ndarray) -> BatchResult:
    """Evaluate the H1.5 constant-preload response."""
    coefficients = np.zeros(
        (len(seeds), NUM_FOURIER_COEFFICIENTS), dtype=np.float64
    )
    return evaluate_controlled_batch(q, coefficients, seeds)


def switching_gate(result: BatchResult) -> bool:
    """Return whether the bounded H1.5 two-contact switching gate passes."""
    stick_to_slip = np.asarray(result.stick_to_slip)
    slip_to_stick = np.asarray(result.slip_to_stick)
    complete_cycles = np.logical_and(stick_to_slip > 0, slip_to_stick > 0)
    switching_seeds = np.any(complete_cycles, axis=1)
    locations_switch = np.any(complete_cycles, axis=0)
    return bool(np.count_nonzero(switching_seeds) >= 4 and np.all(locations_switch))


def select_baseline(
    seeds: np.ndarray = TRAINING_SEEDS,
) -> tuple[np.ndarray | None, BatchResult | None]:
    """Select the first preload in the fixed, bounded H1.5 candidate order."""
    for preload in BASELINE_PRELOADS:
        q = np.array([BASELINE_DAMPING, preload], dtype=np.float64)
        result = evaluate_batch(q, seeds)
        if switching_gate(result):
            return q, result
    return None, None


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


def crn_fd_controlled_q_jacobian(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
    epsilon_multiplier: float = 1.0,
) -> np.ndarray:
    """Return d(seed_losses)/dq at fixed Fourier coefficients."""
    q_array = np.asarray(q, dtype=np.float64)
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    epsilons = FD_RELATIVE_EPSILON * q_array * epsilon_multiplier
    columns = []
    for index, epsilon in enumerate(epsilons):
        plus = q_array.copy()
        minus = q_array.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_losses = np.asarray(
            evaluate_controlled_batch(plus, coefficients_array, seeds).losses
        )
        minus_losses = np.asarray(
            evaluate_controlled_batch(minus, coefficients_array, seeds).losses
        )
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)


def crn_fd_coefficient_jacobian(
    q: np.ndarray | jax.Array,
    coefficients: np.ndarray | jax.Array,
    seeds: np.ndarray,
    epsilon: float = COEFFICIENT_FD_EPSILON,
) -> np.ndarray:
    """Return the eight independent five-dimensional CRN-FD gradients."""
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    columns = []
    for index in range(NUM_FOURIER_COEFFICIENTS):
        plus = coefficients_array.copy()
        minus = coefficients_array.copy()
        plus[:, index] += epsilon
        minus[:, index] -= epsilon
        plus_losses = np.asarray(
            evaluate_controlled_batch(q, plus, seeds).losses
        )
        minus_losses = np.asarray(
            evaluate_controlled_batch(q, minus, seeds).losses
        )
        columns.append((plus_losses - minus_losses) / (2.0 * epsilon))
    return np.stack(columns, axis=1)
