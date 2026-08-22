"""Run the final two-Tesseract JumpGrad demo or registered 100-update training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from stochastic_stick_slip.jumpgrad import (
    AUDIT_STREAM,
    FIXED_Q,
    HELD_OUT_CONDITIONS,
    HELD_OUT_STREAM,
    MONITOR_STREAM,
    TRAINING_CONDITIONS,
    TRAINING_STREAM,
    build_jumpgrad_controller,
    condition_descriptors,
    crn_fd_condition_gradient,
    deterministic_condition_objectives,
    direct_ad_physics_gradient,
    flatten_jumpgrad_parameters,
    jumpgrad_uniform_bank,
    q_polar_rows,
)
CONTROLLER_API = ROOT / "tesseracts/jumpgrad_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/wu_v2_markov_fem/tesseract_api.py"

NUM_UPDATES = 100
LEARNING_RATE = 0.01
NUM_TRAINING_REALIZATIONS = 8
NUM_MONITOR_REALIZATIONS = 16
NUM_HELD_OUT_REALIZATIONS = 32
MONITOR_ITERATIONS = np.arange(0, NUM_UPDATES + 1, 10, dtype=np.int64)
ZERO_GRADIENT_TOLERANCE = 1e-12
NONZERO_GRADIENT_TOLERANCE = 1e-12


def create_tesseracts():
    return (
        Tesseract.from_tesseract_api(CONTROLLER_API),
        Tesseract.from_tesseract_api(PHYSICS_API),
    )


def initial_theta() -> np.ndarray:
    return np.asarray(
        flatten_jumpgrad_parameters(build_jumpgrad_controller()).detach(),
        dtype=np.float64,
    )


def controller_q(controller, theta, conditions) -> np.ndarray:
    return np.asarray(
        controller.apply(
            {
                "theta": np.asarray(theta, dtype=np.float64),
                "descriptors": condition_descriptors(conditions),
            }
        )["q"],
        dtype=np.float64,
    )


def physics_evaluation(physics, q, conditions, tapes) -> dict[str, np.ndarray]:
    result = physics.apply(
        {
            "q": np.asarray(q, dtype=np.float64),
            "conditions": np.asarray(conditions, dtype=np.float64),
            "markov_tapes": np.asarray(tapes, dtype=np.float64),
        }
    )
    arrays = {
        "objectives": np.asarray(result["objectives"], dtype=np.float64),
        "transition_counts": np.asarray(
            result["transition_counts"], dtype=np.int64
        ),
        "high_fraction": np.asarray(result["high_fraction"], dtype=np.float64),
    }
    if not np.all(np.isfinite(arrays["objectives"])) or not np.all(
        np.isfinite(arrays["high_fraction"])
    ):
        raise FloatingPointError("Tesseract physics evaluation is non-finite")
    return arrays


def differentiable_loss(controller, physics, theta, conditions, tapes, passive):
    q = apply_tesseract(
        controller,
        {
            "theta": theta,
            "descriptors": condition_descriptors(conditions),
        },
    )["q"]
    result = apply_tesseract(
        physics,
        {"q": q, "conditions": conditions, "markov_tapes": tapes},
    )
    passive_tensor = torch.tensor(
        np.asarray(passive, dtype=np.float64), dtype=torch.float64
    )
    return (result["objectives"] / passive_tensor).mean(), q


def normalized_objective(objectives, passive) -> float:
    return float(
        np.mean(
            np.asarray(objectives, dtype=np.float64)
            / np.asarray(passive, dtype=np.float64)
        )
    )


def gradient_audit(controller, physics, theta0, passive) -> dict:
    conditions = TRAINING_CONDITIONS[:2]
    passive = np.asarray(passive[:2], dtype=np.float64)
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    q0 = controller_q(controller, theta0, conditions)
    cotangent = 1.0 / (len(conditions) * passive)
    _, direct_gradient = direct_ad_physics_gradient(
        q0, conditions, tapes, cotangent
    )
    fd_raw = crn_fd_condition_gradient(q0, conditions, tapes)["gradient"]
    fd_gradient = cotangent[:, None] * fd_raw

    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    loss, _ = differentiable_loss(
        controller, physics, theta, conditions, tapes, passive
    )
    loss.backward()
    theta_gradient = theta.grad.detach().cpu().numpy().copy()
    values = (direct_gradient, fd_gradient, theta_gradient)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError("gradient audit is non-finite")
    return {
        "direct_ad_linf": float(np.max(np.abs(direct_gradient))),
        "crn_fd_l2": float(np.linalg.norm(fd_gradient)),
        "theta_gradient_l2": float(np.linalg.norm(theta_gradient)),
        "normalized_loss": float(loss.detach()),
        "gates": {
            "direct_ad_zero": bool(
                np.max(np.abs(direct_gradient)) <= ZERO_GRADIENT_TOLERANCE
            ),
            "crn_fd_finite_nonzero": bool(
                np.linalg.norm(fd_gradient) > NONZERO_GRADIENT_TOLERANCE
            ),
            "theta_gradient_finite_nonzero": bool(
                np.linalg.norm(theta_gradient) > NONZERO_GRADIENT_TOLERANCE
            ),
        },
    }


def run_demo() -> None:
    """Run the public two-condition mixed-gradient example."""
    controller, physics = create_tesseracts()
    theta0 = initial_theta()
    conditions = TRAINING_CONDITIONS[:2]
    passive = deterministic_condition_objectives(conditions, "passive")
    audit = gradient_audit(controller, physics, theta0, passive)
    if not all(audit["gates"].values()):
        raise AssertionError("JumpGrad gradient demo failed")

    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = differentiable_loss(
        controller, physics, theta, conditions, tapes, passive
    )
    loss.backward()
    optimizer.step()
    updated = theta.detach().cpu().numpy()
    changed = not np.array_equal(updated, theta0)
    if not changed or not np.all(np.isfinite(updated)):
        raise AssertionError("one-update JumpGrad demo failed")

    print("jumpgrad_demo=PASS")
    print(f"direct_ad_linf={audit['direct_ad_linf']:.16g}")
    print(f"crn_fd_l2={audit['crn_fd_l2']:.16g}")
    print(f"theta_gradient_l2={audit['theta_gradient_l2']:.16g}")
    print(f"normalized_loss={float(loss.detach()):.16g}")
    print("one_update=PASS")


def stochastic_summary(physics, q, conditions, tapes, passive) -> dict:
    result = physics_evaluation(physics, q, conditions, tapes)
    normalized = result["objectives"] / passive
    return {
        "mean_normalized_response": float(np.mean(normalized)),
        "mean_reduction_vs_passive_percent": float(
            100.0 * (1.0 - np.mean(normalized))
        ),
        "mean_transition_count_per_trajectory_contact": float(
            np.mean(result["transition_counts"])
        ),
    }


def train_controller(controller, physics, theta0, passive) -> dict:
    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    sampled = np.empty(NUM_UPDATES + 1, dtype=np.float64)
    gradient_norm = np.empty(NUM_UPDATES, dtype=np.float64)
    q_history = np.empty(
        (NUM_UPDATES + 1, len(TRAINING_CONDITIONS), 2), dtype=np.float64
    )
    monitor = np.empty(len(MONITOR_ITERATIONS), dtype=np.float64)

    first_tapes = jumpgrad_uniform_bank(
        len(TRAINING_CONDITIONS),
        NUM_TRAINING_REALIZATIONS,
        TRAINING_STREAM,
        iteration=0,
    )
    q_history[0] = controller_q(controller, theta0, TRAINING_CONDITIONS)
    sampled[0] = normalized_objective(
        physics_evaluation(
            physics, q_history[0], TRAINING_CONDITIONS, first_tapes
        )["objectives"],
        passive,
    )
    monitor_tapes = jumpgrad_uniform_bank(
        len(TRAINING_CONDITIONS),
        NUM_MONITOR_REALIZATIONS,
        MONITOR_STREAM,
        iteration=0,
    )
    monitor[0] = normalized_objective(
        physics_evaluation(
            physics, q_history[0], TRAINING_CONDITIONS, monitor_tapes
        )["objectives"],
        passive,
    )

    started = time.perf_counter()
    monitor_index = 1
    for update in range(1, NUM_UPDATES + 1):
        tapes = jumpgrad_uniform_bank(
            len(TRAINING_CONDITIONS),
            NUM_TRAINING_REALIZATIONS,
            TRAINING_STREAM,
            iteration=update - 1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = differentiable_loss(
            controller, physics, theta, TRAINING_CONDITIONS, tapes, passive
        )
        loss.backward()
        gradient = theta.grad.detach().cpu().numpy()
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("training theta gradient is non-finite")
        gradient_norm[update - 1] = np.linalg.norm(gradient)
        optimizer.step()
        theta_array = theta.detach().cpu().numpy().copy()
        if not np.all(np.isfinite(theta_array)):
            raise FloatingPointError("training theta is non-finite")
        q_history[update] = controller_q(
            controller, theta_array, TRAINING_CONDITIONS
        )
        sampled[update] = normalized_objective(
            physics_evaluation(
                physics, q_history[update], TRAINING_CONDITIONS, tapes
            )["objectives"],
            passive,
        )

        monitor_value = None
        if update % 10 == 0:
            monitor_value = normalized_objective(
                physics_evaluation(
                    physics,
                    q_history[update],
                    TRAINING_CONDITIONS,
                    monitor_tapes,
                )["objectives"],
                passive,
            )
            monitor[monitor_index] = monitor_value
            monitor_index += 1
        message = (
            f"update={update:03d} sampled={sampled[update]:.12g} "
            f"gradient_l2={gradient_norm[update - 1]:.9g}"
        )
        if monitor_value is not None:
            message += f" monitor={monitor_value:.12g}"
        print(message, flush=True)

    if monitor_index != len(MONITOR_ITERATIONS):
        raise AssertionError("monitor history is incomplete")
    magnitude = np.empty(q_history.shape[:2], dtype=np.float64)
    phase = np.empty_like(magnitude)
    for row in range(len(q_history)):
        magnitude[row], phase[row] = q_polar_rows(q_history[row])
    return {
        "sampled_objective": sampled,
        "gradient_l2": gradient_norm,
        "monitor_objective": monitor,
        "q_history": q_history,
        "q_magnitude_history": magnitude,
        "q_phase_history": phase,
        "final_theta": theta.detach().cpu().numpy().copy(),
        "training_seconds": time.perf_counter() - started,
    }


def run_training() -> None:
    """Run the unchanged registered 100-update JumpGrad experiment."""
    controller, physics = create_tesseracts()
    theta0 = initial_theta()
    passive_training = deterministic_condition_objectives(
        TRAINING_CONDITIONS, "passive"
    )
    passive_held = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "passive"
    )
    wu_held = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "wu_continuous_2omega"
    )
    audit = gradient_audit(controller, physics, theta0, passive_training)
    if not all(audit["gates"].values()):
        raise AssertionError("registered gradient audit failed")

    training = train_controller(controller, physics, theta0, passive_training)
    final_theta = training["final_theta"]
    final_training_q = controller_q(
        controller, final_theta, TRAINING_CONDITIONS
    )
    q_spread = float(
        np.max(
            np.linalg.norm(
                final_training_q[:, None, :] - final_training_q[None, :, :],
                axis=-1,
            )
        )
    )
    held_tapes = jumpgrad_uniform_bank(
        len(HELD_OUT_CONDITIONS),
        NUM_HELD_OUT_REALIZATIONS,
        HELD_OUT_STREAM,
        iteration=0,
    )
    final_held_q = controller_q(controller, final_theta, HELD_OUT_CONDITIONS)
    jumpgrad = stochastic_summary(
        physics, final_held_q, HELD_OUT_CONDITIONS, held_tapes, passive_held
    )
    fixed = stochastic_summary(
        physics,
        np.broadcast_to(FIXED_Q, final_held_q.shape),
        HELD_OUT_CONDITIONS,
        held_tapes,
        passive_held,
    )
    wu_reduction = float(100.0 * (1.0 - np.mean(wu_held / passive_held)))
    gates = {
        **audit["gates"],
        "fixed_monitor_improved": bool(
            training["monitor_objective"][-1]
            < training["monitor_objective"][0]
        ),
        "condition_dependent_q": bool(q_spread > 1e-12),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    print(f"jumpgrad_training={status}")
    print(
        f"monitor_initial={training['monitor_objective'][0]:.16g} "
        f"monitor_final={training['monitor_objective'][-1]:.16g}"
    )
    print(f"q_spread={q_spread:.16g}")
    print(f"held_out_wu_reduction_percent={wu_reduction:.12g}")
    print(
        "held_out_jumpgrad_reduction_percent="
        f"{jumpgrad['mean_reduction_vs_passive_percent']:.12g}"
    )
    print(
        "held_out_fixed_q_reduction_percent="
        f"{fixed['mean_reduction_vs_passive_percent']:.12g}"
    )
    print(f"training_seconds={training['training_seconds']:.3f}")
    if status != "PASS":
        raise AssertionError("registered JumpGrad training gates failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final JumpGrad Tesseract demo or full training."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run the registered 100-update training instead of the quick demo",
    )
    arguments = parser.parse_args()
    if arguments.train:
        run_training()
    else:
        run_demo()


if __name__ == "__main__":
    main()
