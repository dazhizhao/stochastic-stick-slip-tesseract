"""Train and evaluate the condition-aware mixed-gradient JumpGrad system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    NUM_CONTROLLER_PARAMETERS,
    OMEGA_R,
    OMEGA_R_RATIO,
    TRAINING_CONDITIONS,
    TRAINING_STREAM,
    WU_AMPLITUDE,
    WU_PHASE,
    build_jumpgrad_controller,
    condition_descriptors,
    crn_fd_condition_gradient,
    deterministic_condition_objectives,
    direct_ad_physics_gradient,
    flatten_jumpgrad_parameters,
    jumpgrad_uniform_bank,
    q_polar_rows,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
)
from stochastic_stick_slip.wu_v2_markov import (
    FD_EPSILON,
    MARKOV_BASE_SEED,
    PRELOAD_HIGH,
    PRELOAD_LOW,
)


W1_PATH = ROOT / "outputs/wu2019_reproduction/scorecard.json"
W2_PATH = ROOT / "outputs/wu_v2_head_to_head/results.json"
CONTROLLER_API = ROOT / "tesseracts/jumpgrad_controller/tesseract_api.py"
PHYSICS_API = ROOT / "tesseracts/wu_v2_markov_fem/tesseract_api.py"
OUTPUT_DIRECTORY = ROOT / "outputs/jumpgrad_end_to_end"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
SUMMARY_PATH = OUTPUT_DIRECTORY / "summary.md"
OPTIMIZATION_PATH = OUTPUT_DIRECTORY / "optimization.png"
HELD_OUT_PATH = OUTPUT_DIRECTORY / "held_out.png"
ARCHITECTURE_PATH = OUTPUT_DIRECTORY / "architecture.png"

NUM_UPDATES = 100
LEARNING_RATE = 0.01
NUM_TRAINING_REALIZATIONS = 8
NUM_MONITOR_REALIZATIONS = 16
NUM_HELD_OUT_REALIZATIONS = 32
MONITOR_ITERATIONS = np.arange(0, NUM_UPDATES + 1, 10, dtype=np.int64)
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12
ZERO_GRADIENT_TOLERANCE = 1e-12
NONZERO_GRADIENT_TOLERANCE = 1e-12
REPRESENTATIVE_TRAINING_INDICES = (0, 3, 7)

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#777D84"
WU_COLOR = "#315F55"
FIXED_COLOR = "#527E99"
JUMPGRAD_COLOR = "#B36A4C"
SAMPLED_COLOR = "#9AA1A8"
CONDITION_COLORS = ("#376A8B", "#7D5687", "#B36A4C")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _allclose(left, right) -> bool:
    return bool(
        np.allclose(
            left,
            right,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
        )
    )


def _load_frozen_references() -> dict:
    w1 = _read_json(W1_PATH)
    w2 = _read_json(W2_PATH)
    config = w1["configuration"]
    harmonic = w1["harmonic_search"]["2"]
    candidate = w2["candidates"]["stochastic_lr1p0"]
    valid = (
        config["mesh"] == "32x4 QUAD4"
        and config["num_free_dofs"] == SYSTEM.num_free_dofs
        and config["num_periods"] == DIAGNOSTIC_NUM_PERIODS
        and config["steps_per_period"] == 100
        and _allclose(config["damping"], DAMPING)
        and _allclose(config["forcing_amplitude"], FORCING_AMPLITUDE)
        and _allclose(config["preload_A0"], REFERENCE_PRELOAD)
        and _allclose(config["omega_r"], OMEGA_R)
        and _allclose(config["omega_r_ratio"], OMEGA_R_RATIO)
        and _allclose(harmonic["best_amplitude"], WU_AMPLITUDE)
        and _allclose(harmonic["best_phase_rad"], WU_PHASE)
        and _allclose(candidate["q"], FIXED_Q)
    )
    if not valid:
        raise RuntimeError("W1/W2 frozen JumpGrad references do not match")
    return {
        "omega_r": float(config["omega_r"]),
        "omega_r_ratio": float(config["omega_r_ratio"]),
        "passive_preload": float(config["preload_A0"]),
        "wu_amplitude": float(harmonic["best_amplitude"]),
        "wu_phase": float(harmonic["best_phase_rad"]),
        "fixed_q": np.asarray(candidate["q"], dtype=np.float64),
        "fixed_q_source": "W2 stochastic lr=1.0, iteration 100",
    }


def create_tesseracts():
    return (
        Tesseract.from_tesseract_api(CONTROLLER_API),
        Tesseract.from_tesseract_api(PHYSICS_API),
    )


def initial_theta() -> np.ndarray:
    return np.asarray(
        flatten_jumpgrad_parameters(
            build_jumpgrad_controller()
        ).detach(),
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
        "high_fraction": np.asarray(
            result["high_fraction"], dtype=np.float64
        ),
    }
    if not np.all(np.isfinite(arrays["objectives"])) or not np.all(
        np.isfinite(arrays["high_fraction"])
    ):
        raise FloatingPointError("Tesseract physics evaluation is non-finite")
    return arrays


def differentiable_loss(
    controller,
    physics,
    theta,
    conditions,
    tapes,
    passive,
):
    q = apply_tesseract(
        controller,
        {
            "theta": theta,
            "descriptors": condition_descriptors(conditions),
        },
    )["q"]
    result = apply_tesseract(
        physics,
        {
            "q": q,
            "conditions": conditions,
            "markov_tapes": tapes,
        },
    )
    passive_tensor = torch.tensor(
        np.asarray(passive, dtype=np.float64), dtype=torch.float64
    )
    return (result["objectives"] / passive_tensor).mean(), q


def normalized_objective(objectives, passive) -> float:
    values = np.asarray(objectives, dtype=np.float64) / np.asarray(
        passive, dtype=np.float64
    )
    return float(np.mean(values))


def _audit_gradients(controller, physics, theta0, passive) -> dict:
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
        "conditions": conditions.tolist(),
        "num_realizations": 2,
        "stream": AUDIT_STREAM,
        "q0": q0.tolist(),
        "direct_ad_physics_gradient": direct_gradient.tolist(),
        "direct_ad_linf": float(np.max(np.abs(direct_gradient))),
        "crn_fd_physics_gradient": fd_gradient.tolist(),
        "crn_fd_l2": float(np.linalg.norm(fd_gradient)),
        "theta_gradient": theta_gradient.tolist(),
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


def run_smoke() -> None:
    controller, physics = create_tesseracts()
    theta0 = initial_theta()
    passive = deterministic_condition_objectives(
        TRAINING_CONDITIONS[:2], "passive"
    )
    audit = _audit_gradients(controller, physics, theta0, passive)
    if not all(audit["gates"].values()):
        raise AssertionError("JumpGrad smoke gradient gates failed")

    theta = torch.nn.Parameter(torch.from_numpy(theta0.copy()))
    optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
    tapes = jumpgrad_uniform_bank(2, 2, AUDIT_STREAM, iteration=0)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = differentiable_loss(
        controller,
        physics,
        theta,
        TRAINING_CONDITIONS[:2],
        tapes,
        passive,
    )
    loss.backward()
    optimizer.step()
    changed = not np.array_equal(theta.detach().numpy(), theta0)
    if not changed or not np.all(np.isfinite(theta.detach().numpy())):
        raise AssertionError("one-step JumpGrad smoke update failed")
    print("jumpgrad_smoke=PASS")
    print(f"loss={float(loss.detach()):.16g}")
    print(f"theta_gradient_l2={audit['theta_gradient_l2']:.16g}")


def _evaluate_stochastic_method(
    physics,
    q,
    conditions,
    tapes,
    passive,
) -> dict:
    result = physics_evaluation(physics, q, conditions, tapes)
    objectives = result["objectives"]
    normalized = objectives / passive
    return {
        "objectives": objectives.tolist(),
        "normalized_response": normalized.tolist(),
        "reduction_vs_passive_percent": (
            100.0 * (1.0 - normalized)
        ).tolist(),
        "mean_normalized_response": float(np.mean(normalized)),
        "mean_transition_count_per_trajectory_contact": float(
            np.mean(result["transition_counts"])
        ),
        "mean_high_fraction_per_contact": np.mean(
            result["high_fraction"], axis=(0, 1)
        ).tolist(),
    }


def _deterministic_method_summary(objectives, passive) -> dict:
    objectives = np.asarray(objectives, dtype=np.float64)
    normalized = objectives / passive
    return {
        "objectives": objectives.tolist(),
        "normalized_response": normalized.tolist(),
        "reduction_vs_passive_percent": (
            100.0 * (1.0 - normalized)
        ).tolist(),
        "mean_normalized_response": float(np.mean(normalized)),
    }


def _evaluate_split(
    controller,
    physics,
    theta,
    conditions,
    tapes,
    passive,
    wu,
) -> tuple[dict, np.ndarray]:
    jumpgrad_q = controller_q(controller, theta, conditions)
    fixed_q = np.broadcast_to(FIXED_Q, jumpgrad_q.shape).copy()
    methods = {
        "passive": _deterministic_method_summary(passive, passive),
        "wu_continuous_2omega": _deterministic_method_summary(wu, passive),
        "fixed_q": _evaluate_stochastic_method(
            physics, fixed_q, conditions, tapes, passive
        ),
        "jumpgrad": _evaluate_stochastic_method(
            physics, jumpgrad_q, conditions, tapes, passive
        ),
    }
    return methods, jumpgrad_q


def _train(controller, physics, theta0, passive) -> dict:
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
            controller,
            physics,
            theta,
            TRAINING_CONDITIONS,
            tapes,
            passive,
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
        post = physics_evaluation(
            physics, q_history[update], TRAINING_CONDITIONS, tapes
        )
        sampled[update] = normalized_objective(
            post["objectives"], passive
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
        "iterations": np.arange(NUM_UPDATES + 1).tolist(),
        "sampled_objective": sampled.tolist(),
        "gradient_l2": gradient_norm.tolist(),
        "monitor_iterations": MONITOR_ITERATIONS.tolist(),
        "monitor_objective": monitor.tolist(),
        "q_history": q_history.tolist(),
        "q_magnitude_history": magnitude.tolist(),
        "q_phase_history": phase.tolist(),
        "initial_theta": theta0.tolist(),
        "final_theta": theta.detach().cpu().numpy().tolist(),
        "training_seconds": time.perf_counter() - started,
    }


def _condition_records(conditions) -> list[dict]:
    return [
        {
            "index": index,
            "forcing_ratio": float(condition[0]),
            "frequency_ratio": float(condition[1]),
            "descriptor": condition_descriptors(
                np.asarray(condition)[None, :]
            )[0].tolist(),
        }
        for index, condition in enumerate(np.asarray(conditions))
    ]


def _controller_output_records(conditions, q) -> list[dict]:
    magnitude, phase = q_polar_rows(q)
    return [
        {
            "condition_index": index,
            "forcing_ratio": float(condition[0]),
            "frequency_ratio": float(condition[1]),
            "q": np.asarray(q[index]).tolist(),
            "magnitude": float(magnitude[index]),
            "phase": float(phase[index]),
            "phase_fraction": float(phase[index] / (2.0 * np.pi)),
        }
        for index, condition in enumerate(np.asarray(conditions))
    ]


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        width=0.9,
        colors=FRAME_COLOR,
    )


def _plot_optimization(results: dict) -> None:
    _configure_plotting()
    training = results["training"]
    iterations = np.asarray(training["iterations"])
    q_magnitude = np.asarray(training["q_magnitude_history"])
    q_phase = np.asarray(training["q_phase_history"]) / (2.0 * np.pi)
    figure = plt.figure(figsize=(7.2, 3.5))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.35, 1.0), hspace=0.16, wspace=0.30
    )
    axis = figure.add_subplot(grid[:, 0])
    axis.plot(
        iterations,
        training["sampled_objective"],
        color=SAMPLED_COLOR,
        linewidth=1.1,
        label="Sampled training",
    )
    axis.plot(
        training["monitor_iterations"],
        training["monitor_objective"],
        color=JUMPGRAD_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        label="Fixed monitor",
    )
    axis.set_xlabel("Update")
    axis.set_ylabel("Normalized response")
    axis.legend(loc="best")
    axis.text(-0.13, 1.03, "a", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)

    labels = []
    for index in REPRESENTATIVE_TRAINING_INDICES:
        condition = TRAINING_CONDITIONS[index]
        labels.append(f"{condition[0]:.1f}, {condition[1]:.2f}")
    magnitude_axis = figure.add_subplot(grid[0, 1])
    phase_axis = figure.add_subplot(grid[1, 1], sharex=magnitude_axis)
    for color, label, index in zip(
        CONDITION_COLORS,
        labels,
        REPRESENTATIVE_TRAINING_INDICES,
        strict=True,
    ):
        magnitude_axis.plot(
            iterations,
            q_magnitude[:, index],
            color=color,
            linewidth=1.5,
            label=label,
        )
        phase_axis.plot(
            iterations,
            q_phase[:, index],
            color=color,
            linewidth=1.5,
        )
    magnitude_axis.set_ylabel("q magnitude")
    magnitude_axis.legend(
        loc="best", title=r"$F/F_0$, $\omega/\omega_r$", title_fontsize=6.8
    )
    magnitude_axis.tick_params(labelbottom=False)
    magnitude_axis.text(
        -0.18, 1.08, "b", transform=magnitude_axis.transAxes, fontweight="bold"
    )
    phase_axis.set_xlabel("Update")
    phase_axis.set_ylabel(r"q phase / $2\pi$")
    _style_axis(magnitude_axis)
    _style_axis(phase_axis)
    figure.savefig(
        OPTIMIZATION_PATH,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_held_out(results: dict) -> None:
    _configure_plotting()
    methods = results["evaluations"]["held_out"]["methods"]
    method_names = ("wu_continuous_2omega", "fixed_q", "jumpgrad")
    colors = (WU_COLOR, FIXED_COLOR, JUMPGRAD_COLOR)
    labels = ("Wu-style 2ω", "Fixed q", "JumpGrad")
    condition_labels = [
        f"{condition[0]:.1f}\n{condition[1]:.2f}"
        for condition in HELD_OUT_CONDITIONS
    ]
    x = np.arange(len(condition_labels), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    for offset, name, color, label in zip(
        (-width, 0.0, width), method_names, colors, labels, strict=True
    ):
        axis.bar(
            x + offset,
            methods[name]["reduction_vs_passive_percent"],
            width=width,
            color=color,
            label=label,
        )
    axis.axhline(0.0, color=FRAME_COLOR, linewidth=0.8)
    axis.set_xticks(x, condition_labels)
    axis.set_xlabel(r"Held-out condition: $F/F_0$, $\omega/\omega_r$")
    axis.set_ylabel("Reduction vs passive (%)")
    axis.legend(loc="best", ncol=3)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(
        HELD_OUT_PATH,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_architecture(results: dict) -> None:
    _configure_plotting()
    figure, axis = plt.subplots(figsize=(7.2, 2.45))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    outer = FancyBboxPatch(
        (0.015, 0.08),
        0.97,
        0.82,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=FRAME_COLOR,
        facecolor="white",
    )
    axis.add_patch(outer)
    boxes = (
        (0.05, 0.34, 0.14, 0.30, "Operating\ncondition", PASSIVE_COLOR),
        (0.24, 0.28, 0.16, 0.42, "PyTorch MLP\n2–16–16–2", JUMPGRAD_COLOR),
        (0.45, 0.34, 0.11, 0.30, r"$q=[a_2,b_2]$", FIXED_COLOR),
        (0.61, 0.28, 0.16, 0.42, "Hard Markov\nLOW / HIGH", FIXED_COLOR),
        (0.82, 0.28, 0.14, 0.42, "JAX-FEM\n+ Jenkins", WU_COLOR),
    )
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=color,
            facecolor=mpl.colors.to_rgba(color, 0.10),
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2.0,
            y + height / 2.0,
            label,
            ha="center",
            va="center",
            color=FRAME_COLOR,
            fontsize=8.0,
        )
    for start, end in ((0.19, 0.24), (0.40, 0.45), (0.56, 0.61), (0.77, 0.82)):
        axis.annotate(
            "",
            xy=(end, 0.49),
            xytext=(start, 0.49),
            arrowprops={"arrowstyle": "-|>", "color": FRAME_COLOR, "lw": 1.1},
        )
    axis.text(0.32, 0.77, "autograd VJP", ha="center", color=JUMPGRAD_COLOR)
    axis.text(0.69, 0.77, "CRN-FD VJP", ha="center", color=FIXED_COLOR)
    axis.text(0.50, 0.14, "Tesseract composition", ha="center", fontweight="bold")
    status = results["status"]["result"]
    axis.text(0.94, 0.14, f"J1 {status}", ha="right", fontweight="bold")
    figure.savefig(
        ARCHITECTURE_PATH,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _write_summary(results: dict) -> None:
    audit = results["gradient_audit"]
    training = results["training"]
    evaluations = results["evaluations"]
    lines = [
        "# JumpGrad end-to-end result",
        "",
        f"**J1 {results['status']['result']}**",
        "",
        "## Mixed-gradient evidence",
        "",
        "| Gradient route | Norm | Gate |",
        "|---|---:|---:|",
        f"| Direct AD physics | {audit['direct_ad_linf']:.12g} (L∞) | "
        f"{audit['gates']['direct_ad_zero']} |",
        f"| CRN-FD physics | {audit['crn_fd_l2']:.12g} (L2) | "
        f"{audit['gates']['crn_fd_finite_nonzero']} |",
        f"| End-to-end theta | {audit['theta_gradient_l2']:.12g} (L2) | "
        f"{audit['gates']['theta_gradient_finite_nonzero']} |",
        "",
        "## Training",
        "",
        f"- Fixed monitor, iteration 0: `{training['monitor_objective'][0]:.12g}`",
        f"- Fixed monitor, iteration 100: `{training['monitor_objective'][-1]:.12g}`",
        f"- Condition-dependent q: `{results['status']['gates']['condition_dependent_q']}`",
        "",
        "## Mean normalized response",
        "",
        "| Split | Passive | Wu-style 2ω | Fixed q | JumpGrad |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("training", "held_out"):
        methods = evaluations[split]["methods"]
        lines.append(
            f"| {split.replace('_', ' ').title()} | "
            f"{methods['passive']['mean_normalized_response']:.9f} | "
            f"{methods['wu_continuous_2omega']['mean_normalized_response']:.9f} | "
            f"{methods['fixed_q']['mean_normalized_response']:.9f} | "
            f"{methods['jumpgrad']['mean_normalized_response']:.9f} |"
        )
    lines.extend(
        [
            "",
            "Held-out rankings are reported as observed and are not J1 gates.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines))


def _plot_all(results: dict) -> None:
    _plot_optimization(results)
    _plot_held_out(results)
    _plot_architecture(results)


def run_formal() -> dict:
    total_started = time.perf_counter()
    frozen = _load_frozen_references()
    controller, physics = create_tesseracts()
    theta0 = initial_theta()
    passive_training = deterministic_condition_objectives(
        TRAINING_CONDITIONS, "passive"
    )
    passive_held = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "passive"
    )
    wu_training = deterministic_condition_objectives(
        TRAINING_CONDITIONS, "wu_continuous_2omega"
    )
    wu_held = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "wu_continuous_2omega"
    )
    audit = _audit_gradients(
        controller, physics, theta0, passive_training
    )
    if not all(audit["gates"].values()):
        raise AssertionError("pre-registered gradient audit failed")

    training = _train(controller, physics, theta0, passive_training)
    final_theta = np.asarray(training["final_theta"], dtype=np.float64)
    monitor_tapes = jumpgrad_uniform_bank(
        len(TRAINING_CONDITIONS),
        NUM_MONITOR_REALIZATIONS,
        MONITOR_STREAM,
        iteration=0,
    )
    held_tapes = jumpgrad_uniform_bank(
        len(HELD_OUT_CONDITIONS),
        NUM_HELD_OUT_REALIZATIONS,
        HELD_OUT_STREAM,
        iteration=0,
    )
    training_methods, final_training_q = _evaluate_split(
        controller,
        physics,
        final_theta,
        TRAINING_CONDITIONS,
        monitor_tapes,
        passive_training,
        wu_training,
    )
    held_methods, final_held_q = _evaluate_split(
        controller,
        physics,
        final_theta,
        HELD_OUT_CONDITIONS,
        held_tapes,
        passive_held,
        wu_held,
    )
    q_spread = float(
        np.max(
            np.linalg.norm(
                final_training_q[:, None, :] - final_training_q[None, :, :],
                axis=-1,
            )
        )
    )
    gates = {
        **audit["gates"],
        "fixed_monitor_improved": bool(
            training["monitor_objective"][-1]
            < training["monitor_objective"][0]
        ),
        "condition_dependent_q": bool(q_spread > 1e-12),
    }
    result = "PASS" if all(gates.values()) else "FAIL"
    results = {
        "configuration": {
            "architecture": [2, 16, 16, 2],
            "activation": "tanh",
            "num_controller_parameters": NUM_CONTROLLER_PARAMETERS,
            "dtype": "float64",
            "num_updates": NUM_UPDATES,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "fd_epsilon": FD_EPSILON,
            "markov_base_seed": MARKOV_BASE_SEED,
            "streams": {
                "audit": AUDIT_STREAM,
                "training": TRAINING_STREAM,
                "monitor": MONITOR_STREAM,
                "held_out": HELD_OUT_STREAM,
            },
            "realizations": {
                "training": NUM_TRAINING_REALIZATIONS,
                "monitor": NUM_MONITOR_REALIZATIONS,
                "held_out": NUM_HELD_OUT_REALIZATIONS,
            },
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": 100,
            "objective_cycles": [21, 22, 23, 24],
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
        },
        "frozen_references": {
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in frozen.items()
            },
            "forcing_amplitude_F0": FORCING_AMPLITUDE,
            "damping": DAMPING,
        },
        "conditions": {
            "training": _condition_records(TRAINING_CONDITIONS),
            "held_out": _condition_records(HELD_OUT_CONDITIONS),
        },
        "passive_references": {
            "training": passive_training.tolist(),
            "held_out": passive_held.tolist(),
        },
        "gradient_audit": audit,
        "training": training,
        "evaluations": {
            "training": {
                "num_realizations": NUM_MONITOR_REALIZATIONS,
                "stream": MONITOR_STREAM,
                "methods": training_methods,
            },
            "held_out": {
                "num_realizations": NUM_HELD_OUT_REALIZATIONS,
                "stream": HELD_OUT_STREAM,
                "methods": held_methods,
            },
        },
        "controller_outputs": {
            "training": _controller_output_records(
                TRAINING_CONDITIONS, final_training_q
            ),
            "held_out": _controller_output_records(
                HELD_OUT_CONDITIONS, final_held_q
            ),
            "training_max_pairwise_q_distance": q_spread,
        },
        "comparisons": {
            "training_jumpgrad_minus_fixed_q": float(
                training_methods["jumpgrad"]["mean_normalized_response"]
                - training_methods["fixed_q"]["mean_normalized_response"]
            ),
            "held_out_jumpgrad_minus_fixed_q": float(
                held_methods["jumpgrad"]["mean_normalized_response"]
                - held_methods["fixed_q"]["mean_normalized_response"]
            ),
            "performance_rankings_are_not_gates": True,
        },
        "status": {"result": result, "gates": gates},
        "runtime": {
            "training_seconds": training["training_seconds"],
        },
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _write_summary(results)
    _plot_all(results)
    results["runtime"]["total_seconds"] = time.perf_counter() - total_started
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"jumpgrad_j1={result}")
    print(
        f"monitor_initial={training['monitor_objective'][0]:.16g} "
        f"monitor_final={training['monitor_objective'][-1]:.16g}"
    )
    print(f"q_spread={q_spread:.16g}")
    print(f"results={RESULTS_PATH}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        run_smoke()
    else:
        run_formal()


if __name__ == "__main__":
    main()
