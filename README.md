# Stochastic Stick-Slip Vibration Suppression

Stage H1.5 is a local Tesseract optimization loop for stochastic vibration suppression. A 16×2 QUAD4 cantilever is assembled with JAX-FEM, advanced with dense JAX time stepping, and coupled to two hard Jenkins friction elements. The two design variables remain damping `c` and shared normal preload `N`.

The hard forward model retains exact STICK/SLIP switching. Gradients across that non-smooth, stochastic boundary use centered finite differences with common random numbers. The local pipeline is:

```text
q = (c, N)
  → stick_slip_fem Tesseract (JAX-FEM + two hard Jenkins contacts + CRN-FD)
  → stochastic_objective Tesseract (mean of 8 fixed-seed losses)
  → jax.value_and_grad
```

## Run locally

```bash
uv sync
uv run pytest -q
uv run python -m scripts.run_stage_h15
```

No Docker image, PETSc solve, server, or background job is used by Stage H1.5.

## Stage H1.5 result

The native Mac run passed:

- mesh: 32 elements, 102 total DOF, 96 free DOF;
- baseline `q0 = (0.2, 0.04)` and `J(q0) = 6.642794744e-3`;
- both contact locations exhibited complete STICK→SLIP→STICK cycles in all 8 seeds;
- perturbing `N` changed both contact state sequences for all 8 seeds;
- nominal two-Tesseract gradient `dJ/dq = (-4.820083778e-3, 6.233154825e-2)`;
- the first allowed step produced `q1 = (0.2000963745, 0.0387537208)` and `J(q1) = 6.566509297e-3`, a 1.1484% reduction;
- five accepted hard-forward steps reached `q5 = (0.2005889997, 0.0337787434)` and `J(q5) = 6.343376252e-3`, a 4.5074% total reduction;
- warm Mac timings were approximately 0.009 s for one seed, 0.017 s for the 8-seed objective, and 0.068 s for the nominal 2D CRN gradient.

Plots are generated only after all H1.5 gates pass and are written to `outputs/stage_h15/`.

## License

This repository is licensed under Apache-2.0. JAX-FEM is an external GPL-3.0 dependency and is not relicensed by this repository.
