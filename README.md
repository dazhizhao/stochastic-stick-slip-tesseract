# Stochastic Stick-Slip Vibration Suppression

Stage H1 is a minimal local Tesseract optimization loop for stochastic vibration suppression. A 4×1 QUAD4 cantilever is assembled with JAX-FEM, advanced with small dense JAX solves, and coupled to a hard Jenkins friction element. The two design variables are damping `c` and normal preload `N`.

The hard forward model retains exact STICK/SLIP switching. Gradients across that non-smooth, stochastic boundary use centered finite differences with common random numbers. The local pipeline is:

```text
q = (c, N)
  → stick_slip_fem Tesseract (JAX-FEM + hard Jenkins + CRN-FD)
  → stochastic_objective Tesseract (mean of 8 fixed-seed losses)
  → jax.value_and_grad
```

## Run locally

```bash
uv sync
uv run pytest -q
uv run python -m scripts.run_stage_h1
```

No Docker image, PETSc solve, server, or background job is used by Stage H1.

## Stage H1 result

The native Mac run passed:

- baseline `q0 = (0.2, 0.0403246841)`;
- baseline objective `J(q0) = 1.757140786e-3`;
- 6 of 8 seeds exhibited both STICK→SLIP and SLIP→STICK transitions;
- perturbing `N` changed all 8 hard state sequences;
- nominal two-Tesseract gradient `dJ/dq = (-7.545866764e-4, 7.403175242e-4)`;
- the first allowed step produced `q1 = (0.2014392358, 0.0389126641)` and `J(q1) = 1.755606281e-3`;
- relative objective reduction: `8.732963214e-4` (0.08733%).

Plots are generated only after all H1 gates pass and are written to `outputs/stage_h1/`.

## License

This repository is licensed under Apache-2.0. JAX-FEM is an external GPL-3.0 dependency and is not relicensed by this repository.
