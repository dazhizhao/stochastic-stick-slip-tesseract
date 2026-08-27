<p align="center">
  <img src="assets/jumpgrad_logo.png" width="90" alt="JumpGrad logo" align="middle">&nbsp;&nbsp;&nbsp;
  <img src="assets/jumpgrad_title.svg" width="700" alt="JumpGrad: End-to-End Gradient Optimization through Discrete Stochastic Mechanics" align="middle">
</p>

<p align="center">
  <a href="https://pasteurlabs.ai/tesseract-hackathon-2026/">Tesseract Hackathon 2026</a> · <strong>Track 03 — Hybrid ML + Mechanistic Models</strong><br>
  <a href="#1-problem">Problem</a> · <a href="#2-method">Method</a> · <a href="#3-why-tesseract">Why Tesseract</a> · <a href="#4-results">Results</a> · <a href="#5-reproduce">Reproduce</a>
</p>

<p align="center">
  <img src="outputs/jumpgrad_visuals/passive_wu_jumpgrad.gif" width="760" alt="Passive, Wu2019, and JumpGrad beam vibration under the same operating condition">
</p>

## 1. Problem

We study vibration suppression in a finite-element beam with two semi-active friction dampers. Each actuator switches between `LOW` and `HIGH` clamping-force modes, corresponding to two fixed normal preloads. In practice, actuator response delay, sensor noise, and contact-state fluctuations make the realized switching times uncertain, so a Markov jump process represents this aggregate mode-switching uncertainty. The friction law and its coefficients remain fixed; randomness enters through the actuator switching times. Given the forcing amplitude and frequency, the neural controller sets the transition probabilities and learns a policy that suppresses vibration despite the stochastic switching.

With a fixed random tape, a small controller perturbation may leave the sampled switching trajectory unchanged. Ordinary automatic differentiation then returns no useful mechanics gradient along that path.

JumpGrad trains the neural controller end to end and keeps every switching event discrete.

## 2. Method

JumpGrad connects a PyTorch neural controller to JAX-FEM mechanics through Tesseract. The mechanics model uses hard Markov switching, Jenkins friction, and a custom gradient for the discrete path.

```text
Operating condition
        ↓
    PyTorch MLP
        ↓
switching parameters q
        ↓
hard LOW/HIGH Markov switching
        ↓
JAX-FEM + Jenkins friction
        ↓
vibration objective
        ↓
backward through Tesseract
```

The [`jumpgrad_controller`](tesseracts/jumpgrad_controller/tesseract_api.py) Tesseract maps operating conditions to `q=[a2,b2]`. The [`wu_v2_markov_fem`](tesseracts/wu_v2_markov_fem/tesseract_api.py) Tesseract converts `q` and explicit random tapes into hard preload histories, advances the beam response, and returns the vibration objective.

<p align="center">
  <img src="outputs/jumpgrad_visuals/tesseract_pipeline.png?v=f04d3c3" width="760" alt="Two peer Tesseract blocks composing a PyTorch controller with hard stochastic JAX-FEM mechanics">
</p>

The two Tesseracts form the end-to-end system. The controller has 354 parameters, and the registered experiment trains them for 100 Adam updates using the hard stochastic mechanics.

## 3. Why Tesseract

The controller runs in PyTorch, and the stochastic mechanics runs in JAX. Each component needs a different backward rule: the controller uses PyTorch autograd, and the mechanics uses common-random-number finite differences for hard switching.

Tesseract gives each component its own forward implementation and derivative rule, then composes their derivatives into one end-to-end backward pass.

| Component | Forward | Backward |
|---|---|---|
| [`jumpgrad_controller`](tesseracts/jumpgrad_controller/tesseract_api.py) | operating condition → `q` | PyTorch autograd |
| [`wu_v2_markov_fem`](tesseracts/wu_v2_markov_fem/tesseract_api.py) | `q` + random tapes → vibration objective | CRN centered finite differences |

### Hard switching needs a custom gradient

For a fixed random tape, a small change in `q` can leave every sampled `LOW`/`HIGH` decision unchanged. The preload history and mechanical trajectory are then locally constant with respect to `q`. In the registered gradient audit, direct automatic differentiation through this sampled path returns a zero mechanics gradient.

The physics interface has only two dimensions, so a centered finite difference is affordable and keeps the original hard forward model:

$$
g_i =
\frac{
L(q + \varepsilon e_i;\xi) - L(q - \varepsilon e_i;\xi)
}{
2\varepsilon
}.
$$

Each `+eps`/`-eps` pair receives the same explicit `markov_tapes`. This common-random-number pairing reduces stochastic variation unrelated to `q`. The forward pass keeps the original `LOW`/`HIGH` events, and PyTorch autograd propagates the resulting mechanics cotangent through the controller.

### Independent batches in Tesseract Core

The mechanics block uses our [independent-batch extension to Tesseract Core](https://github.com/dazhizhao/tesseract-core/commit/43fe09bd8ef1a96569e8499d022482d1ae4ce1de). For batched `q[B,2]`, the extension perturbs one coefficient across all independent conditions at once. This reduces centered differences from `4B` mechanics evaluations to four and keeps same-tape CRN pairing.

<p align="center">
  <img src="outputs/jumpgrad_visuals/gradient_story.png" width="640" alt="Direct AD, CRN finite-difference, and end-to-end controller gradients with optimization history">
</p>

## 4. Results

On the same numerical JAX-FEM/Jenkins benchmark, the reproduced Wu2019 control method reduces the sampled resonance peak by about 20.2% relative to passive friction. JumpGrad reaches **23.9%** reduction. Both peaks lie inside the sampled local-FRF window.

<p align="center">
  <img src="outputs/jumpgrad_visuals/main_results.png" width="680" alt="Local resonance response and peak reduction for Passive, Wu2019, and JumpGrad">
</p>

We evaluated the frozen controller on 128 unseen random realizations, each aggregated with equal weight across the same eight held-out operating conditions. Training improves the mean aggregate reduction from **2.99% → 21.11%**, and **128/128** realizations improve. The MLP maps each forcing amplitude and frequency pair to its own switching parameters.

<p align="center">
  <img src="outputs/jumpgrad_visuals/held_out.png" width="680" alt="Initial and Trained JumpGrad aggregate reductions with paired fresh-seed improvements">
</p>

The hero uses four contacts for visual clarity; the quantitative results use the original two-contact benchmark.

## 5. Reproduce

With Python 3.12 and [uv](https://docs.astral.sh/uv/) installed, clone the repository and run the JumpGrad demo:

```bash
git clone https://github.com/dazhizhao/stochastic-stick-slip-tesseract.git
cd stochastic-stick-slip-tesseract
uv sync
uv run python scripts/run_jumpgrad_end_to_end.py
```

The demo runs the neural controller through the hard stochastic mechanics and propagates its gradient backward through both Tesseracts.

## 6. References

1. Y. G. Wu et al., “Design of semi-active dry friction dampers for steady-state vibration: sensitivity analysis and experimental studies,” *Journal of Sound and Vibration* 459, 114850 (2019). [doi:10.1016/j.jsv.2019.114850](https://doi.org/10.1016/j.jsv.2019.114850)
2. D. Häfner and A. Lavin, “Tesseract Core: Universal, autodiff-native software components for Simulation Intelligence,” *Journal of Open Source Software* 10(111), 8385 (2025). [doi:10.21105/joss.08385](https://doi.org/10.21105/joss.08385)
3. T. Xue et al., “JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science,” *Computer Physics Communications* 291, 108802 (2023). [doi:10.1016/j.cpc.2023.108802](https://doi.org/10.1016/j.cpc.2023.108802)
4. H. J. Suh, M. Simchowitz, K. Zhang, and R. Tedrake, “Do Differentiable Simulators Give Better Policy Gradients?” *Proceedings of Machine Learning Research* 162, 20668–20696 (2022). [PMLR](https://proceedings.mlr.press/v162/suh22b.html)
5. L. Dai, “Rate of Convergence for Derivative Estimation of Discrete-Time Markov Chains via Finite-Difference Approximation with Common Random Numbers,” *SIAM Journal on Applied Mathematics* 57(3), 731–751 (1997). [doi:10.1137/S003613999427173X](https://doi.org/10.1137/S003613999427173X)

The repository is licensed under [Apache-2.0](LICENSE). JAX-FEM is an external GPL-3.0 dependency; its source is not copied into this repository.
