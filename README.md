# Stochastic Stick-Slip Vibration Suppression with Tesseract

This repository demonstrates mixed-gradient stochastic vibration suppression. A 16×2 QUAD4 cantilever is assembled with JAX-FEM, advanced with dense JAX time stepping, and coupled to two hard Jenkins friction elements.

Stage H2 uses a small PyTorch MLP to map each stochastic forcing descriptor to five fixed Fourier coefficients. The hard forward model retains exact STICK/SLIP switching, while centered finite differences with common random numbers connect the low-dimensional control interface back to PyTorch:

```text
forcing descriptor
  → PyTorch MLP (6 → 16 → 16 → 5)
  → five Fourier coefficients per seed
  → stick_slip_fem Tesseract (CRN-FD + JAX-FEM + hard Jenkins contacts)
  → stochastic_objective Tesseract (mean of 8 fixed-seed losses)
  → PyTorch backward
```

## Run locally

```bash
uv sync
uv run pytest -q
uv run python scripts/run_stage_h15.py
uv run python scripts/run_stage_h2.py
```

No Docker image, PETSc solve, server, GPU, or background job is used.

## Stage H2 result

The native Mac CPU/Float64 run passed:

- the zero-initialized MLP exactly reproduced the fixed `N=0.04` objective, `J_fixed = J_initial = 6.642794744e-3`;
- `loss.backward()` crossed both Tesseracts with total and final-layer gradient norms of `1.616181728e-3`;
- coefficient-FD direction cosines were `0.9993548471` and `0.9973847367` for epsilon `0.01/0.02/0.04`;
- the first Adam step at `lr=0.01` reduced the real hard objective to `6.570569575e-3`;
- 20 steps reached `J_final = 5.769207114e-3`, a 13.1509% training reduction;
- held-out objective decreased from `6.385067390e-3` to `6.199324621e-3`, a 2.9090% reduction;
- trained preload histories remained bounded in `[0.0200217, 0.0597595]`;
- the eight coefficient rows were distinct, with maximum pairwise coefficient distance `2.62153` and maximum pairwise preload-history RMS difference `0.0215773`;
- two complete runs reproduced the same objectives, coefficients, accepted learning rate, and PASS result.

Representative results:

![Fixed and learned preload and response](outputs/stage_h2/fourier_controlled_response.png)

![Twenty-step training objective](outputs/stage_h2/training_objective.png)

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

Representative results:

![Mesh, two contact locations, and representative response](outputs/stage_h15/mesh_and_two_contact_response.png)

![Five-step objective history](outputs/stage_h15/objective_history.png)

Plots are generated only after all H1.5 gates pass and are written to `outputs/stage_h15/`.

## License

This repository is licensed under Apache-2.0. JAX-FEM is used as an external dependency and is distributed under GPL-3.0; it is not relicensed by this repository.
