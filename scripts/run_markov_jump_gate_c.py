"""Run the 10-step Direct-AD versus CRN-FD Markov-jump Gate C."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from stochastic_stick_slip.controller import (
    NUM_CONTROLLER_PARAMETERS,
    build_controller,
    flatten_controller_parameters,
)
from stochastic_stick_slip.engineering_markov import (
    BETA,
    DAMPING,
    FD_EPSILON,
    GATE_A_FORCING_SEEDS,
    LAMBDA_0,
    MARKOV_BASE_SEED,
    MARKOV_ITERATION,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    SIMULATE_MARKOV_BANK,
    evaluate_markov_bank,
    gate_a_forcing,
    markov_uniform_bank,
)
from stochastic_stick_slip.model import forcing_descriptor_batch


CONTROLLER_API = ROOT / "tesseracts/fourier_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/markov_jump_fem/tesseract_api.py"
OUTPUT_PATH = ROOT / "outputs/markov_jump_gate_c/results.json"
TRAINING_STREAM = 5
EVALUATION_STREAM = 6
NUM_REALIZATIONS = 4
LEARNING_RATE = 0.01
NUM_UPDATES = 10
NUMERICAL_ZERO_ATOL = 1e-12
REPORTED_ITERATIONS = (0, 1, 2, 5, 10)


def _raw_hard_objective(coefficients, forcing, uniforms):
    displacement = SIMULATE_MARKOV_BANK(coefficients, forcing, uniforms)[0]
    return jnp.mean(displacement**2)


_DIRECT_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_raw_hard_objective))


def _direct_value_and_grad(coefficients, forcing, uniforms):
    value, gradient = _DIRECT_VALUE_AND_GRAD(
        jnp.asarray(coefficients, dtype=jnp.float64),
        jnp.asarray(forcing, dtype=jnp.float64),
        jnp.asarray(uniforms, dtype=jnp.float64),
    )
    return float(value), np.asarray(gradient)


def _create_tesseracts():
    return (
        Tesseract.from_tesseract_api(CONTROLLER_API),
        Tesseract.from_tesseract_api(PHYSICS_API),
    )


def _controller_coefficients(controller, theta, descriptors):
    return np.asarray(
        controller.apply(
            {
                "theta": np.asarray(theta, dtype=np.float64),
                "descriptors": descriptors,
            }
        )["coeffs"]
    )


def _hard_evaluation(
    controller,
    physics,
    theta,
    descriptors,
    uniforms,
):
    coefficients = _controller_coefficients(
        controller, theta, descriptors
    )
    response = physics.apply(
        {
            "coeffs": coefficients,
            "forcing_seeds": GATE_A_FORCING_SEEDS,
            "markov_uniforms": uniforms,
        }
    )
    seed_losses = np.asarray(response["seed_losses"], dtype=np.float64)
    if not np.all(np.isfinite(seed_losses)):
        raise FloatingPointError("hard evaluation is non-finite")
    return coefficients, response, float(np.mean(seed_losses))


def _objective_entry(iteration, train_objective, evaluation_objective):
    return {
        "iteration": iteration,
        "train_objective": train_objective,
        "evaluation_objective": evaluation_objective,
    }


def _gradient_entry(iteration, coefficient_gradient, theta_gradient, theta, theta0):
    coefficient_gradient = np.asarray(coefficient_gradient, dtype=np.float64)
    theta_gradient = np.asarray(theta_gradient, dtype=np.float64)
    if not np.all(np.isfinite(coefficient_gradient)):
        raise FloatingPointError("coefficient gradient is non-finite")
    if not np.all(np.isfinite(theta_gradient)):
        raise FloatingPointError("theta gradient is non-finite")
    return {
        "iteration": iteration,
        "coefficient_gradient_norm": float(
            np.linalg.norm(coefficient_gradient)
        ),
        "coefficient_gradient_linf": float(
            np.max(np.abs(coefficient_gradient))
        ),
        "theta_gradient_norm": float(np.linalg.norm(theta_gradient)),
        "parameter_displacement": float(
            np.linalg.norm(np.asarray(theta) - theta0)
        ),
    }


def _evaluate_objectives(
    controller,
    physics,
    theta,
    descriptors,
    training_uniforms,
    evaluation_uniforms,
):
    train = _hard_evaluation(
        controller, physics, theta, descriptors, training_uniforms
    )
    evaluation = _hard_evaluation(
        controller, physics, theta, descriptors, evaluation_uniforms
    )
    return train, evaluation


def _direct_backward(
    controller,
    theta,
    descriptors,
    forcing,
    training_uniforms,
    theta0,
    iteration,
):
    coefficients = apply_tesseract(
        controller,
        {"theta": theta, "descriptors": descriptors},
    )["coeffs"]
    _, coefficient_gradient = _direct_value_and_grad(
        coefficients.detach().numpy(), forcing, training_uniforms
    )
    if np.max(np.abs(coefficient_gradient)) > NUMERICAL_ZERO_ATOL:
        raise RuntimeError(
            "Direct AD became non-zero; audit for a continuous shortcut"
        )
    coefficients.backward(
        torch.from_numpy(
            np.array(coefficient_gradient, dtype=np.float64, copy=True)
        )
    )
    return _gradient_entry(
        iteration,
        coefficient_gradient,
        theta.grad.detach().numpy(),
        theta.detach().numpy(),
        theta0,
    )


def _crn_backward(
    controller,
    physics,
    theta,
    descriptors,
    training_uniforms,
    theta0,
    iteration,
):
    coefficients = apply_tesseract(
        controller,
        {"theta": theta, "descriptors": descriptors},
    )["coeffs"]
    coefficients.retain_grad()
    response = apply_tesseract(
        physics,
        {
            "coeffs": coefficients,
            "forcing_seeds": GATE_A_FORCING_SEEDS,
            "markov_uniforms": training_uniforms,
        },
    )
    response["seed_losses"].mean().backward()
    return _gradient_entry(
        iteration,
        coefficients.grad.detach().numpy(),
        theta.grad.detach().numpy(),
        theta.detach().numpy(),
        theta0,
    )


def _train_direct(
    controller,
    physics,
    theta0,
    descriptors,
    forcing,
    training_uniforms,
    evaluation_uniforms,
):
    theta = torch.nn.Parameter(torch.as_tensor(theta0).clone())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    initial_train, initial_eval = _evaluate_objectives(
        controller,
        physics,
        theta.detach().numpy(),
        descriptors,
        training_uniforms,
        evaluation_uniforms,
    )
    objective_history = [
        _objective_entry(0, initial_train[2], initial_eval[2])
    ]
    gradient_history = []

    for iteration in range(NUM_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        gradient_history.append(
            _direct_backward(
                controller,
                theta,
                descriptors,
                forcing,
                training_uniforms,
                theta0,
                iteration,
            )
        )
        optimizer.step()
        train, evaluation = _evaluate_objectives(
            controller,
            physics,
            theta.detach().numpy(),
            descriptors,
            training_uniforms,
            evaluation_uniforms,
        )
        objective_history.append(
            _objective_entry(iteration + 1, train[2], evaluation[2])
        )

    optimizer.zero_grad(set_to_none=True)
    gradient_history.append(
        _direct_backward(
            controller,
            theta,
            descriptors,
            forcing,
            training_uniforms,
            theta0,
            NUM_UPDATES,
        )
    )
    return {
        "objective_history": objective_history,
        "gradient_history": gradient_history,
        "parameter_displacement": float(
            np.linalg.norm(theta.detach().numpy() - theta0)
        ),
    }


def _train_crn(
    controller,
    physics,
    theta0,
    descriptors,
    training_uniforms,
    evaluation_uniforms,
):
    theta = torch.nn.Parameter(torch.as_tensor(theta0).clone())
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    initial_train, initial_eval = _evaluate_objectives(
        controller,
        physics,
        theta.detach().numpy(),
        descriptors,
        training_uniforms,
        evaluation_uniforms,
    )
    objective_history = [
        _objective_entry(0, initial_train[2], initial_eval[2])
    ]
    gradient_history = []

    for iteration in range(NUM_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        gradient_history.append(
            _crn_backward(
                controller,
                physics,
                theta,
                descriptors,
                training_uniforms,
                theta0,
                iteration,
            )
        )
        optimizer.step()
        train, evaluation = _evaluate_objectives(
            controller,
            physics,
            theta.detach().numpy(),
            descriptors,
            training_uniforms,
            evaluation_uniforms,
        )
        objective_history.append(
            _objective_entry(iteration + 1, train[2], evaluation[2])
        )

    optimizer.zero_grad(set_to_none=True)
    gradient_history.append(
        _crn_backward(
            controller,
            physics,
            theta,
            descriptors,
            training_uniforms,
            theta0,
            NUM_UPDATES,
        )
    )
    final_train, final_eval = _evaluate_objectives(
        controller,
        physics,
        theta.detach().numpy(),
        descriptors,
        training_uniforms,
        evaluation_uniforms,
    )
    return (
        theta.detach().numpy(),
        {
            "objective_history": objective_history,
            "gradient_history": gradient_history,
            "parameter_displacement": float(
                np.linalg.norm(theta.detach().numpy() - theta0)
            ),
        },
        initial_train,
        initial_eval,
        final_train,
        final_eval,
    )


def _markov_diagnostics(response):
    transition_counts = np.asarray(response["transition_counts"])
    high_mode_fraction = np.asarray(response["high_mode_fraction"])
    return {
        "transition_totals": np.sum(
            transition_counts, axis=(0, 1)
        ).astype(int).tolist(),
        "mean_high_mode_fraction": np.mean(
            high_mode_fraction, axis=(0, 1)
        ).tolist(),
    }


def _paired_evaluation(theta0_coefficients, theta10_coefficients, forcing, uniforms):
    initial = np.asarray(
        evaluate_markov_bank(
            theta0_coefficients, forcing, uniforms
        ).losses
    )
    final = np.asarray(
        evaluate_markov_bank(
            theta10_coefficients, forcing, uniforms
        ).losses
    )
    return {
        "initial_mean_objective": float(np.mean(initial)),
        "final_mean_objective": float(np.mean(final)),
        "improved_count": int(np.count_nonzero(final < initial)),
        "num_trajectories": int(initial.size),
    }


def _history_entry(result, iteration):
    return next(
        entry
        for entry in result["objective_history"]
        if entry["iteration"] == iteration
    )


def _print_history(direct, crn):
    print("iter | Direct-AD eval | CRN-FD eval")
    for iteration in REPORTED_ITERATIONS:
        direct_entry = _history_entry(direct, iteration)
        crn_entry = _history_entry(crn, iteration)
        print(
            f"{iteration:02d} | "
            f"{direct_entry['evaluation_objective']:.16g} | "
            f"{crn_entry['evaluation_objective']:.16g}"
        )


def main() -> int:
    torch.set_default_dtype(torch.float64)
    controller, physics = _create_tesseracts()
    theta0 = (
        flatten_controller_parameters(build_controller()).detach().numpy()
    )
    if theta0.shape != (NUM_CONTROLLER_PARAMETERS,):
        raise ValueError("unexpected controller parameter shape")
    descriptors = forcing_descriptor_batch(GATE_A_FORCING_SEEDS)
    forcing = gate_a_forcing()
    training_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS, stream_id=TRAINING_STREAM
    )
    evaluation_uniforms = markov_uniform_bank(
        NUM_REALIZATIONS, stream_id=EVALUATION_STREAM
    )

    direct = _train_direct(
        controller,
        physics,
        theta0,
        descriptors,
        forcing,
        training_uniforms,
        evaluation_uniforms,
    )
    (
        theta10,
        crn,
        initial_train,
        initial_eval,
        final_train,
        final_eval,
    ) = _train_crn(
        controller,
        physics,
        theta0,
        descriptors,
        training_uniforms,
        evaluation_uniforms,
    )

    initial_coefficients = _controller_coefficients(
        controller, theta0, descriptors
    )
    final_coefficients = _controller_coefficients(
        controller, theta10, descriptors
    )
    paired = _paired_evaluation(
        initial_coefficients,
        final_coefficients,
        forcing,
        evaluation_uniforms,
    )
    crn_initial_eval = crn["objective_history"][0]["evaluation_objective"]
    crn_final_eval = crn["objective_history"][-1]["evaluation_objective"]
    if not np.isclose(
        paired["initial_mean_objective"], crn_initial_eval, rtol=1e-12, atol=1e-14
    ) or not np.isclose(
        paired["final_mean_objective"], crn_final_eval, rtol=1e-12, atol=1e-14
    ):
        raise AssertionError("paired and Tesseract evaluation objectives differ")

    direct_initial_gradient = direct["gradient_history"][0]
    direct_final_gradient = direct["gradient_history"][-1]
    direct_initial_eval = direct["objective_history"][0]["evaluation_objective"]
    direct_final_eval = direct["objective_history"][-1]["evaluation_objective"]
    direct_gate = bool(
        direct_initial_gradient["coefficient_gradient_linf"]
        <= NUMERICAL_ZERO_ATOL
        and direct_final_gradient["coefficient_gradient_linf"]
        <= NUMERICAL_ZERO_ATOL
        and direct["parameter_displacement"] <= NUMERICAL_ZERO_ATOL
        and np.isclose(
            direct_initial_eval,
            direct_final_eval,
            rtol=0.0,
            atol=NUMERICAL_ZERO_ATOL,
        )
    )
    crn_gate = bool(crn_final_eval < crn_initial_eval)
    passed = bool(direct_gate and crn_gate)

    results = {
        "configuration": {
            "forcing_seeds": GATE_A_FORCING_SEEDS.tolist(),
            "markov_base_seed": MARKOV_BASE_SEED,
            "markov_iteration": MARKOV_ITERATION,
            "training_stream": TRAINING_STREAM,
            "evaluation_stream": EVALUATION_STREAM,
            "num_realizations": NUM_REALIZATIONS,
            "damping": DAMPING,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "lambda_0": LAMBDA_0,
            "beta": BETA,
            "fd_epsilon": FD_EPSILON,
            "learning_rate": LEARNING_RATE,
            "num_updates": NUM_UPDATES,
            "num_controller_parameters": NUM_CONTROLLER_PARAMETERS,
        },
        "pipeline": {
            "controller": "fourier_controller Tesseract / PyTorch autograd VJP",
            "physics": "markov_jump_fem Tesseract / coordinate CRN-FD VJP",
        },
        "direct_ad": direct,
        "crn_fd": crn,
        "markov_response": {
            "training_bank": {
                "iteration_0": _markov_diagnostics(initial_train[1]),
                "iteration_10": _markov_diagnostics(final_train[1]),
            },
            "evaluation_bank": {
                "iteration_0": _markov_diagnostics(initial_eval[1]),
                "iteration_10": _markov_diagnostics(final_eval[1]),
            },
        },
        "paired_evaluation": paired,
        "gate_c": {
            "result": "PASS" if passed else "FAIL",
            "direct_ad_control_pass": direct_gate,
            "crn_fd_evaluation_decreased": crn_gate,
            "reason": (
                "Direct AD remained stationary and CRN-FD reduced the final "
                "independent hard evaluation objective."
                if passed
                else "The frozen 10-step Gate C criteria were not all met."
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("## Tesseract pipeline")
    print(results["pipeline"]["controller"])
    print(results["pipeline"]["physics"])
    print("## Evaluation history")
    _print_history(direct, crn)
    print("## Direct AD")
    print(f"initial_gradient={direct_initial_gradient}")
    print(f"final_gradient={direct_final_gradient}")
    print(f"parameter_displacement={direct['parameter_displacement']:.16g}")
    print("## CRN-FD")
    print(f"initial_gradient={crn['gradient_history'][0]}")
    print(f"final_gradient={crn['gradient_history'][-1]}")
    print(f"parameter_displacement={crn['parameter_displacement']:.16g}")
    print("## Markov response")
    print(json.dumps(results["markov_response"], indent=2))
    print(f"paired_evaluation={paired}")
    print("## Gate C")
    print(f"Gate C: {results['gate_c']['result']}")
    print(f"reason={results['gate_c']['reason']}")
    print(OUTPUT_PATH.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
