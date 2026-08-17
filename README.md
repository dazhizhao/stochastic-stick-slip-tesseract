# Stochastic Stick-Slip Vibration Suppression with Tesseract

A small Mac-native mechanics demo in which a PyTorch controller learns a
seed-dependent friction preload through a low-dimensional Fourier interface.
The structural response uses JAX-FEM, the contact law keeps exact hard
STICK/SLIP switching, and Tesseract composes the different gradient systems.

The central design choice is deliberate: finite differences never touch the
MLP weights. They only see five Fourier coefficients for each stochastic seed.
PyTorch autograd handles the high-dimensional network parameters, while a
common-random-number (CRN) finite difference handles the non-smooth stochastic
mechanics boundary.

## Pipeline

```text
forcing parameters (8 fixed seeds)
        │
        ▼
6 forcing descriptors per seed
        │  PyTorch autograd
        ▼
MLP: 6 → 16 → 16 → 5
        │
        ▼
Fourier coefficients z ∈ R^(8×5)
        │  only this low-dimensional interface is FD'd
        ▼
┌──────────────────────────────────────────────────────────────┐
│ stick_slip_fem Tesseract                                     │
│ forward: JAX-FEM + dense dynamics + two hard Jenkins contacts │
│ JVP/VJP: block-diagonal CRN-centered FD in z                  │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
seed_losses ∈ R^8
        │
        ▼
┌──────────────────────────────────────────────┐
│ stochastic_objective Tesseract               │
│ forward: mean(seed_losses)                   │
│ JVP: mean tangent; VJP: cotangent / 8         │
└──────────────────────────────────────────────┘
        │
        ▼
scalar stochastic objective J
        │  loss.backward()
        ▼
gradients of all PyTorch MLP parameters
```

For H2, the physics Tesseract receives fixed `q=(c,N_base)=(0.2,0.04)` and
the MLP produces

```text
z = [a0, a1, b1, a2, b2]
s(t) = a0 + a1 cos(w1 t) + b1 sin(w1 t)
     + a2 cos(w2 t) + b2 sin(w2 t)
N(t) = 0.04 + 0.02 tanh(s(t))
```

where `w1=0.9ω1` and `w2=1.35ω1` are the two existing forcing frequencies.
The coefficient Jacobian is evaluated in five batched columns: one positive
and one negative forward for each column, using the same seed on both sides
(10 batch forwards total for `[8,5]`). No FFT, dynamic top-K mode selection,
or finite difference over network weights is used.

## What is frozen

- 16×2 `QUAD4` cantilever, 32 elements, 102 total DOF and 96 free DOF;
- two independent hard Jenkins contacts on the lower surface;
- exact STICK/SLIP regime projection and slider updates;
- the existing 800-step integration, stochastic forcing and displacement-only loss;
- eight fixed training seeds and one disjoint eight-seed held-out set.

## Results

### Stage H2 — PyTorch Fourier controller

The native Mac CPU/Float64 run passed.

| quantity | result |
| --- | ---: |
| fixed objective `J_fixed` | `0.006642794744366401` |
| zero-initialized MLP `J_initial` | `0.006642794744366401` |
| first accepted Adam learning rate | `0.01` |
| objective after first real hard step | `0.006570569575355878` |
| objective after 20 steps `J_final` | `0.005769207113962147` |
| training reduction | `13.1509%` |
| held-out fixed objective | `0.006385067389503073` |
| held-out trained objective | `0.006199324621437767` |
| held-out reduction | `2.9090%` |
| trained `N(t)` range | `[0.0200217, 0.0597595]` |

The initial zero last layer reproduces the fixed `N=0.04` baseline exactly.
After training, the eight seeds receive genuinely different controls: the
maximum pairwise distance between coefficient rows is `2.62153`, and the
maximum pairwise RMS difference between their preload histories is `0.0215773`.
This is the intended role of the forcing descriptor: the network does not
collapse to one shared preload waveform.

The held-out result is reported once, after training, and is not used for any
model or learning-rate choice.

![Fixed and learned preload and representative displacement](./outputs/stage_h2/fourier_controlled_response.png)

![Training objective over 20 Adam steps](./outputs/stage_h2/training_objective.png)

### Stage H1.5 — two-parameter baseline

The preceding public baseline also passes: all eight seeds show complete
STICK→SLIP→STICK cycles at both contact locations, and the first CRN-FD descent
step reduced `J` by `1.1484%`. Five accepted hard-forward steps gave a total
reduction of `4.5074%`.

![H1.5 mesh, contacts and representative response](./outputs/stage_h15/mesh_and_two_contact_response.png)

![H1.5 objective history](./outputs/stage_h15/objective_history.png)

## Reproduce locally

```bash
uv sync
uv run pytest -q
uv run python scripts/run_stage_h15.py
uv run python scripts/run_stage_h2.py
```

The test suite currently reports 12 passing tests. Re-running the H2 runner
with the fixed Torch seed reproduces the same objective history, coefficients,
accepted learning rate and PASS result. The runs are intentionally local:
there is no Docker image, GPU, server, PETSc solve, background job or push-time
automation required.

## Repository layout

```text
stochastic_stick_slip/model.py                 JAX-FEM, forcing and hard contact
stochastic_stick_slip/controller.py            deterministic PyTorch MLP
tesseracts/stick_slip_fem/tesseract_api.py    physics apply/JVP/VJP
tesseracts/stochastic_objective/tesseract_api.py
                                                mean objective apply/JVP/VJP
scripts/run_stage_h15.py                       H1.5 baseline runner
scripts/run_stage_h2.py                        H2 training and diagnostics
tests/                                          focused physics and composition tests
outputs/stage_h15/                              H1.5 figures
outputs/stage_h2/                               H2 figures
```

## License

This repository is licensed under Apache-2.0. JAX-FEM is used as an external
dependency and is distributed under GPL-3.0; no JAX-FEM source is copied into
this repository.
