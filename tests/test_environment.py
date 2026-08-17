import jax
import jax.numpy as jnp
import jax_fem
import basix
import gmsh
import meshio
import pytest
import torch
from petsc4py import PETSc
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
import tesseract_jax
import tesseract_torch
from tesseract_core import Tesseract


def test_required_imports() -> None:
    assert jax is not None
    assert jax_fem is not None
    assert basix is not None
    assert gmsh is not None
    assert meshio is not None
    assert PETSc.Sys.getVersion()[0] >= 3
    assert torch is not None
    assert Tesseract is not None
    assert tesseract_jax is not None
    assert tesseract_torch is not None


def test_jax_grad() -> None:
    gradient = jax.grad(lambda value: value**2)(2.0)
    assert gradient == pytest.approx(4.0)


def test_tiny_jax_fem_forward_is_finite() -> None:
    class TinyElasticity(Problem):
        def get_tensor_map(self):
            youngs_modulus = 100.0
            poisson_ratio = 0.3
            shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_first_parameter = (
                youngs_modulus
                * poisson_ratio
                / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            )

            def stress(displacement_gradient, *unused_args):
                strain = 0.5 * (displacement_gradient + displacement_gradient.T)
                return (
                    lame_first_parameter * jnp.trace(strain) * jnp.eye(2)
                    + 2.0 * shear_modulus * strain
                )

            return stress

        def get_surface_maps(self):
            def traction(displacement, point):
                return jnp.array([1.0, 0.0])

            return [traction]

    meshio_mesh = rectangle_mesh(Nx=2, Ny=1, domain_x=1.0, domain_y=0.1)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[get_meshio_cell_type("QUAD4")])

    def left(point):
        return jnp.isclose(point[0], 0.0, atol=1e-6)

    def right(point):
        return jnp.isclose(point[0], 1.0, atol=1e-6)

    def zero(point):
        return 0.0

    problem = TinyElasticity(
        mesh,
        vec=2,
        dim=2,
        ele_type="QUAD4",
        quadrature_order=2,
        dirichlet_bc_info=[[left, left], [0, 1], [zero, zero]],
        location_fns=[right],
    )
    solution = solver(
        problem,
        {"newton": {"linear": {"spsolve_solver": {}}}},
    )[0]

    assert bool(jnp.all(jnp.isfinite(solution)))
