"""Render the frozen JumpGrad visualization package without retraining."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
import torch

from stochastic_stick_slip.jumpgrad import (
    HELD_OUT_CONDITIONS,
    NUM_CONTROLLER_PARAMETERS,
    OMEGA_R,
    WU_AMPLITUDE,
    WU_PHASE,
    build_jumpgrad_controller,
    condition_descriptors,
    functional_jumpgrad_controller,
)
from stochastic_stick_slip.model import (
    BEAM_HEIGHT,
    BEAM_LENGTH,
    STEPS_PER_PERIOD,
    build_variable_time_step_mechanics_batch_simulator,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    NUM_STEPS,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    evaluate_markov_bank,
    generate_hard_preload_history,
    markov_uniform_bank,
)


J1_PATH = ROOT / "outputs/jumpgrad_end_to_end/results.json"
W1_PATH = ROOT / "outputs/wu2019_reproduction/scorecard.json"
W2_PATH = ROOT / "outputs/wu_v2_head_to_head/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/jumpgrad_visuals"

MAIN_GIF_PATH = OUTPUT_DIRECTORY / "passive_wu_jumpgrad.gif"
CONTROL_GIF_PATH = OUTPUT_DIRECTORY / "wu_vs_jumpgrad_control.gif"
MAIN_RESULTS_PATH = OUTPUT_DIRECTORY / "main_results.png"
GRADIENT_PATH = OUTPUT_DIRECTORY / "gradient_story.png"

EXPECTED_OUTPUTS = (
    MAIN_GIF_PATH,
    CONTROL_GIF_PATH,
    MAIN_RESULTS_PATH,
    GRADIENT_PATH,
)

SELECTED_HELD_OUT_INDEX = 5
SELECTED_STREAM = 12
SELECTED_REALIZATION = 0
CONFIRMATION_STREAMS = (5, 6, 7, 8)
NUM_REALIZATIONS_PER_CONDITION = 8
REALIZATIONS_PER_STREAM = 64
NUM_CONFIRMATION_REALIZATIONS = 256
STABLE_START = 20 * STEPS_PER_PERIOD
STABLE_STOP = 24 * STEPS_PER_PERIOD
FRAME_STRIDE = 4
NUM_GIF_FRAMES = 100
GIF_FPS = 20
TARGET_DEFORMATION = 0.16 * BEAM_LENGTH
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#858B91"
WU_COLOR = "#35695C"
JUMPGRAD_COLOR = "#B86D4B"
MESH_EDGE_COLOR = "#30363D"

METHOD_COLORS = {
    "passive": PASSIVE_COLOR,
    "wu": WU_COLOR,
    "jumpgrad": JUMPGRAD_COLOR,
}
METHOD_LABELS = {
    "passive": "Passive",
    "wu": "Wu2019",
    "jumpgrad": "JumpGrad",
}

FULL_MECHANICS = build_variable_time_step_mechanics_batch_simulator(
    SYSTEM, return_full_displacement=True
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _allclose(left, right) -> bool:
    return bool(
        np.allclose(left, right, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    )


def load_frozen_sources() -> dict:
    """Load and validate the three immutable scientific source artifacts."""
    j1 = _read_json(J1_PATH)
    w1 = _read_json(W1_PATH)
    w2 = _read_json(W2_PATH)
    ratios = np.asarray(w1["configuration"]["local_frf_ratios"])
    held_out = j1["conditions"]["held_out"]
    valid = (
        j1["status"]["result"] == "PASS"
        and len(j1["training"]["final_theta"]) == NUM_CONTROLLER_PARAMETERS
        and len(held_out) == len(HELD_OUT_CONDITIONS) == 8
        and _allclose(
            [[row["forcing_ratio"], row["frequency_ratio"]] for row in held_out],
            HELD_OUT_CONDITIONS,
        )
        and _allclose(ratios, np.linspace(0.90, 1.10, 21))
        and w2["configuration"]["confirmation_streams"]
        == list(CONFIRMATION_STREAMS)
        and w2["configuration"]["total_confirmation_realizations"]
        == NUM_CONFIRMATION_REALIZATIONS
        and _allclose(w1["harmonic_search"]["2"]["best_phase_rad"], WU_PHASE)
        and _allclose(w1["harmonic_search"]["2"]["best_amplitude"], WU_AMPLITUDE)
    )
    if not valid:
        raise RuntimeError("J1/W1/W2 frozen visualization inputs changed")
    return {"j1": j1, "w1": w1, "w2": w2, "ratios": ratios}


def controller_q(final_theta: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    """Evaluate the frozen J1 MLP for one or more operating conditions."""
    controller = build_jumpgrad_controller()
    theta = torch.as_tensor(final_theta, dtype=torch.float64)
    descriptors = torch.as_tensor(
        condition_descriptors(conditions), dtype=torch.float64
    )
    with torch.no_grad():
        q = functional_jumpgrad_controller(controller, theta, descriptors)
    result = q.detach().cpu().numpy()
    if result.shape != (len(conditions), 2) or not np.all(np.isfinite(result)):
        raise FloatingPointError("frozen JumpGrad controller output is invalid")
    return result


def select_representative_condition(j1: dict) -> dict:
    """Select the stable-order held-out condition nearest median improvement."""
    methods = j1["evaluations"]["held_out"]["methods"]
    wu = np.asarray(methods["wu_continuous_2omega"]["objectives"])
    jumpgrad = np.asarray(methods["jumpgrad"]["objectives"])
    improvement = 100.0 * (wu - jumpgrad) / wu
    median = float(np.median(improvement))
    distance = np.abs(improvement - median)
    minimum = float(np.min(distance))
    tied = np.flatnonzero(
        np.isclose(distance, minimum, rtol=1e-12, atol=1e-12)
    )
    index = int(tied[0])
    if index != SELECTED_HELD_OUT_INDEX:
        raise AssertionError("registered representative condition changed")
    condition = HELD_OUT_CONDITIONS[index]
    return {
        "index": index,
        "forcing_ratio": float(condition[0]),
        "frequency_ratio": float(condition[1]),
        "improvement_percent": float(improvement[index]),
        "median_improvement_percent": median,
    }


def confirmation_banks() -> dict[int, np.ndarray]:
    """Recreate only the four W2 confirmation streams authorized for replay."""
    banks = {
        stream: markov_uniform_bank(
            NUM_REALIZATIONS_PER_CONDITION,
            stream_id=stream,
            iteration=0,
        )
        for stream in CONFIRMATION_STREAMS
    }
    for stream, bank in banks.items():
        expected = (8, 8, NUM_STEPS + 1, 2)
        if bank.shape != expected:
            raise AssertionError(f"stream {stream} has shape {bank.shape}, not {expected}")
    return banks


def local_frf_conditions(ratios: np.ndarray) -> np.ndarray:
    """Return deployed-policy conditions for the nominal-amplitude FRF."""
    values = np.asarray(ratios, dtype=np.float64)
    return np.column_stack((np.ones_like(values), values))


def peak_summary(ratios: np.ndarray, amplitudes: np.ndarray) -> dict:
    """Return the stable sampled-window peak summary."""
    ratios = np.asarray(ratios, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.shape != ratios.shape or not np.all(np.isfinite(amplitudes)):
        raise FloatingPointError("FRF curve is invalid")
    index = int(np.argmax(amplitudes))
    boundary = index in (0, len(ratios) - 1)
    return {
        "index": index,
        "ratio": float(ratios[index]),
        "amplitude": float(amplitudes[index]),
        "at_boundary": boundary,
    }


def evaluate_jumpgrad_local_frf(
    sources: dict, banks: dict[int, np.ndarray]
) -> dict:
    """Replay the final J1 policy on the frozen 21-point W2 bank design."""
    ratios = sources["ratios"]
    final_theta = np.asarray(sources["j1"]["training"]["final_theta"])
    q_values = controller_q(final_theta, local_frf_conditions(ratios))
    values = np.empty(
        (len(ratios), len(CONFIRMATION_STREAMS), REALIZATIONS_PER_STREAM),
        dtype=np.float64,
    )
    for frequency_index, ratio in enumerate(ratios):
        omega = float(OMEGA_R * ratio)
        time_step, forcing = single_tone_forcing(
            FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
        )
        times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
        for bank_index, stream in enumerate(CONFIRMATION_STREAMS):
            result = evaluate_markov_bank(
                q_values[frequency_index],
                forcing,
                banks[stream],
                times,
                omega,
                time_step,
            )
            objectives = np.asarray(result["trajectory_objectives"]).reshape(-1)
            if objectives.shape != (REALIZATIONS_PER_STREAM,):
                raise AssertionError("confirmation realization count changed")
            values[frequency_index, bank_index] = objectives
        print(f"jumpgrad_frf={frequency_index + 1:02d}/{len(ratios)}", flush=True)
    aggregate = np.mean(values.reshape((len(ratios), -1)), axis=1)
    if not np.all(np.isfinite(aggregate)):
        raise FloatingPointError("JumpGrad local FRF is non-finite")
    return {
        "amplitudes": aggregate,
        "q": q_values,
        "peak": peak_summary(ratios, aggregate),
        "num_realizations": NUM_CONFIRMATION_REALIZATIONS,
    }


def _full_nodal_field(free_field: np.ndarray) -> np.ndarray:
    full = np.zeros(
        free_field.shape[:-1] + (SYSTEM.num_total_dofs,), dtype=np.float64
    )
    full[..., SYSTEM.free_dofs] = free_field
    return full.reshape(free_field.shape[:-1] + (len(SYSTEM.points), 2))


def simulate_full_field(
    forcing: np.ndarray, preload: np.ndarray, time_step: float
) -> dict:
    """Run one frozen mechanics path and return its full nodal field."""
    forcing = np.asarray(forcing, dtype=np.float64)
    preload = np.asarray(preload, dtype=np.float64)
    outputs = FULL_MECHANICS(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        jnp.asarray(forcing[None, :], dtype=jnp.float64),
        jnp.asarray(preload[None, :, :], dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    tip = np.asarray(outputs[0][0])
    slip = np.asarray(outputs[2][0], dtype=bool)
    free_field = np.asarray(outputs[5][0])
    observed = free_field @ np.asarray(SYSTEM.observation)
    if not np.allclose(observed, tip, rtol=1e-10, atol=1e-12):
        raise AssertionError("full-field replay does not match scalar observation")
    return {
        "tip": tip,
        "slip": slip,
        "free_field": free_field,
        "field": _full_nodal_field(free_field),
        "preload": preload,
    }


def shared_deformation_scale(fields: list[np.ndarray]) -> tuple[float, float]:
    """Return one physical amplification for every compared deformation field."""
    magnitudes = np.concatenate(
        [
            np.linalg.norm(field[STABLE_START:STABLE_STOP], axis=2).ravel()
            for field in fields
        ]
    )
    maximum = float(np.max(magnitudes))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("deformation replay has no finite motion")
    return TARGET_DEFORMATION / maximum, maximum


def replay_selected_condition(sources: dict) -> dict:
    """Replay Passive, Wu2019, and final JumpGrad on the fixed J2 example."""
    representative = select_representative_condition(sources["j1"])
    condition = np.asarray(
        [[representative["forcing_ratio"], representative["frequency_ratio"]]],
        dtype=np.float64,
    )
    final_theta = np.asarray(sources["j1"]["training"]["final_theta"])
    q = controller_q(final_theta, condition)[0]
    saved_q = np.asarray(
        sources["j1"]["controller_outputs"]["held_out"]
        [SELECTED_HELD_OUT_INDEX]["q"]
    )
    if not _allclose(q, saved_q):
        raise AssertionError("replayed MLP q does not match J1 output")

    omega = float(OMEGA_R * representative["frequency_ratio"])
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE * representative["forcing_ratio"],
        omega,
        DIAGNOSTIC_NUM_PERIODS,
    )
    times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    passive_preload = np.full((NUM_STEPS, 2), REFERENCE_PRELOAD)
    wu_scalar = REFERENCE_PRELOAD + WU_AMPLITUDE * np.sin(
        2.0 * omega * times + WU_PHASE
    )
    wu_preload = np.repeat(wu_scalar[:, None], 2, axis=1)
    tape = markov_uniform_bank(
        32, stream_id=SELECTED_STREAM, iteration=0
    )[SELECTED_HELD_OUT_INDEX, SELECTED_REALIZATION]
    modes, jumpgrad_preload, _, _ = generate_hard_preload_history(
        jnp.asarray(q),
        jnp.asarray(times),
        jnp.asarray(tape),
        jnp.asarray(omega),
        jnp.asarray(time_step),
    )
    methods = {
        "passive": simulate_full_field(forcing, passive_preload, time_step),
        "wu": simulate_full_field(forcing, wu_preload, time_step),
        "jumpgrad": simulate_full_field(
            forcing, np.asarray(jumpgrad_preload), time_step
        ),
    }
    methods["jumpgrad"]["modes"] = np.asarray(modes, dtype=bool)
    scale, maximum = shared_deformation_scale(
        [method["field"] for method in methods.values()]
    )
    frame_indices = np.arange(STABLE_START, STABLE_STOP, FRAME_STRIDE)
    if len(frame_indices) != NUM_GIF_FRAMES:
        raise AssertionError("registered GIF frame count changed")
    return {
        "representative": representative,
        "q": q,
        "omega": omega,
        "time_step": time_step,
        "times": times,
        "methods": methods,
        "deformation_scale": scale,
        "maximum_nodal_displacement": maximum,
        "frame_indices": frame_indices,
    }


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.88,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _style_axis(axis, *, hide_ticks: bool = False) -> None:
    axis.set_facecolor("white")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.88)
    axis.tick_params(
        direction="in",
        top=False,
        right=False,
        colors=FRAME_COLOR,
        width=0.8,
        length=3.0,
    )
    if hide_ticks:
        axis.set_xticks([])
        axis.set_yticks([])


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.13,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _add_beam_panel(axis, method: str):
    collection = PolyCollection(
        SYSTEM.points[SYSTEM.cells],
        facecolor=mpl.colors.to_rgba(METHOD_COLORS[method], 0.30),
        edgecolor=MESH_EDGE_COLOR,
        linewidth=0.28,
    )
    axis.add_collection(collection)
    contacts = axis.scatter(
        [], [], s=24, facecolor="white", edgecolor=METHOD_COLORS[method],
        linewidth=1.2, zorder=5,
    )
    axis.set_xlim(-0.20, BEAM_LENGTH + 0.20)
    axis.set_ylim(-0.20, BEAM_HEIGHT + 0.20)
    axis.set_aspect("equal")
    _style_axis(axis, hide_ticks=True)
    axis.text(
        0.5, 0.94, METHOD_LABELS[method], transform=axis.transAxes,
        ha="center", va="top", fontweight="bold",
    )
    return collection, contacts


def _deformed_vertices(replay: dict, method: str, step: int) -> np.ndarray:
    field = replay["methods"][method]["field"][step]
    return SYSTEM.points + replay["deformation_scale"] * field


def render_main_gif(replay: dict) -> None:
    _configure_plotting()
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.35), dpi=100)
    collections = {}
    contacts = {}
    for axis, method in zip(axes, ("passive", "wu", "jumpgrad"), strict=True):
        collections[method], contacts[method] = _add_beam_panel(axis, method)

    def update(step):
        artists = []
        for method in ("passive", "wu", "jumpgrad"):
            deformed = _deformed_vertices(replay, method, int(step))
            collections[method].set_verts(deformed[SYSTEM.cells])
            contacts[method].set_offsets(deformed[SYSTEM.contact_nodes])
            artists.extend((collections[method], contacts[method]))
        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=replay["frame_indices"],
        interval=1000 / GIF_FPS,
        blit=False,
        repeat=True,
    )
    figure.tight_layout(pad=1.0)
    animation.save(MAIN_GIF_PATH, writer=PillowWriter(fps=GIF_FPS), dpi=100)
    plt.close(figure)


def render_control_gif(replay: dict) -> None:
    _configure_plotting()
    figure = plt.figure(figsize=(7.2, 4.2), dpi=100)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.1, 1.0), hspace=0.27, wspace=0.22)
    beam_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    signal_axes = (figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]))
    collections = {}
    contacts = {}
    for axis, method in zip(beam_axes, ("wu", "jumpgrad"), strict=True):
        collections[method], contacts[method] = _add_beam_panel(axis, method)

    periods = (
        np.arange(NUM_STEPS, dtype=np.float64) - STABLE_START
    ) / STEPS_PER_PERIOD
    commands = {
        "wu": replay["methods"]["wu"]["preload"][:, 0],
        "jumpgrad": replay["methods"]["jumpgrad"]["preload"][:, 0],
    }
    markers = {}
    verticals = {}
    for axis, method in zip(signal_axes, ("wu", "jumpgrad"), strict=True):
        axis.plot(
            periods[STABLE_START:STABLE_STOP],
            commands[method][STABLE_START:STABLE_STOP],
            color=METHOD_COLORS[method],
            linewidth=1.35,
            drawstyle="steps-post" if method == "jumpgrad" else "default",
        )
        markers[method], = axis.plot(
            [], [], "o", color=METHOD_COLORS[method], markersize=4.5,
        )
        verticals[method] = axis.axvline(0.0, color=FRAME_COLOR, linewidth=0.7, alpha=0.65)
        axis.set_xlim(0.0, 4.0)
        axis.set_ylim(PRELOAD_LOW - 0.003, PRELOAD_HIGH + 0.003)
        axis.set_xlabel("Steady-cycle time (periods)")
        _style_axis(axis)
    signal_axes[0].set_ylabel("Preload")
    signal_axes[1].set_yticklabels([])

    def update(step):
        artists = []
        current_period = (int(step) - STABLE_START) / STEPS_PER_PERIOD
        for method in ("wu", "jumpgrad"):
            deformed = _deformed_vertices(replay, method, int(step))
            collections[method].set_verts(deformed[SYSTEM.cells])
            contacts[method].set_offsets(deformed[SYSTEM.contact_nodes])
            markers[method].set_data([current_period], [commands[method][int(step)]])
            verticals[method].set_xdata([current_period, current_period])
            artists.extend(
                (collections[method], contacts[method], markers[method], verticals[method])
            )
        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=replay["frame_indices"],
        interval=1000 / GIF_FPS,
        blit=False,
        repeat=True,
    )
    animation.save(CONTROL_GIF_PATH, writer=PillowWriter(fps=GIF_FPS), dpi=100)
    plt.close(figure)


def _main_result_data(sources: dict, jumpgrad_frf: dict) -> dict:
    w1_methods = sources["w1"]["local_frf"]["methods"]
    curves = {
        "passive": np.asarray(w1_methods["passive"]["steady_amplitudes"]),
        "wu": np.asarray(w1_methods["2omega"]["steady_amplitudes"]),
        "jumpgrad": np.asarray(jumpgrad_frf["amplitudes"]),
    }
    peaks = {name: peak_summary(sources["ratios"], curve) for name, curve in curves.items()}
    passive_peak = peaks["passive"]["amplitude"]
    reductions = {
        name: 100.0 * (passive_peak - peaks[name]["amplitude"]) / passive_peak
        for name in ("wu", "jumpgrad")
    }
    outperforms = (
        not peaks["jumpgrad"]["at_boundary"]
        and peaks["jumpgrad"]["amplitude"] < peaks["wu"]["amplitude"]
    )
    return {
        "curves": curves,
        "peaks": peaks,
        "reductions": reductions,
        "outperforms_wu": outperforms,
    }


def render_main_results(sources: dict, jumpgrad_frf: dict) -> dict:
    _configure_plotting()
    data = _main_result_data(sources, jumpgrad_frf)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": (1.65, 1.0)})
    ratios = sources["ratios"]
    for method in ("passive", "wu", "jumpgrad"):
        axes[0].plot(
            ratios,
            data["curves"][method],
            color=METHOD_COLORS[method],
            linewidth=1.7,
            label=METHOD_LABELS[method],
        )
        peak = data["peaks"][method]
        axes[0].plot(
            peak["ratio"], peak["amplitude"], "o",
            color=METHOD_COLORS[method], markersize=4.5,
            markerfacecolor="white" if peak["at_boundary"] else METHOD_COLORS[method],
        )
    axes[0].set_xlabel(r"Frequency ratio, $\omega/\omega_r$")
    axes[0].set_ylabel("Response amplitude")
    axes[0].legend(loc="best")
    _style_axis(axes[0])
    _panel_label(axes[0], "a")

    names = ("wu", "jumpgrad")
    bars = axes[1].bar(
        np.arange(2),
        [data["reductions"][name] for name in names],
        color=[METHOD_COLORS[name] for name in names],
        width=0.62,
    )
    axes[1].set_xticks(np.arange(2), [METHOD_LABELS[name] for name in names])
    axes[1].set_ylabel("Peak reduction vs passive (%)")
    axes[1].set_ylim(0.0, max(bar.get_height() for bar in bars) * 1.22)
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.45,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    _style_axis(axes[1])
    _panel_label(axes[1], "b")
    figure.tight_layout(w_pad=2.1)
    figure.savefig(MAIN_RESULTS_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return data


def render_gradient_story(j1: dict) -> None:
    _configure_plotting()
    audit = j1["gradient_audit"]
    values = np.asarray(
        [audit["direct_ad_linf"], audit["crn_fd_l2"], audit["theta_gradient_l2"]]
    )
    labels = ("Direct AD\nphysics", "CRN-FD\nphysics", "End-to-end\nMLP")
    colors = (PASSIVE_COLOR, WU_COLOR, JUMPGRAD_COLOR)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": (1.0, 1.65)})
    bars = axes[0].bar(np.arange(3), values, color=colors, width=0.62)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_ylim(0.0, 1.55)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            max(value, 0.015) + 0.045,
            "0" if value == 0.0 else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    _style_axis(axes[0])
    _panel_label(axes[0], "a")

    iterations = np.asarray(j1["training"]["monitor_iterations"])
    monitor = np.asarray(j1["training"]["monitor_objective"])
    axes[1].plot(
        iterations, monitor, "o-", color=JUMPGRAD_COLOR,
        linewidth=1.8, markersize=4.5,
    )
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Fixed-monitor objective")
    axes[1].set_xlim(0, 100)
    _style_axis(axes[1])
    _panel_label(axes[1], "b")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(GRADIENT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate_gif(path: Path, expected_frames: int = NUM_GIF_FRAMES) -> tuple[int, tuple[int, int]]:
    """Validate GitHub-compatible animation metadata."""
    with Image.open(path) as image:
        if not getattr(image, "is_animated", False):
            raise AssertionError(f"{path.name} is not animated")
        frames = int(image.n_frames)
        size = tuple(image.size)
        loop = int(image.info.get("loop", -1))
    if frames != expected_frames or loop != 0:
        raise AssertionError(
            f"{path.name} metadata mismatch: frames={frames}, loop={loop}"
        )
    return frames, size


def _validate_outputs(replay: dict) -> None:
    actual = tuple(sorted(path.name for path in OUTPUT_DIRECTORY.iterdir() if path.is_file()))
    expected = tuple(sorted(path.name for path in EXPECTED_OUTPUTS))
    if actual != expected:
        raise AssertionError(f"visual output contract mismatch: {actual}")
    main_metadata = validate_gif(MAIN_GIF_PATH)
    control_metadata = validate_gif(CONTROL_GIF_PATH)
    if main_metadata[0] != control_metadata[0]:
        raise AssertionError("GIF frame counts differ")
    if not np.isfinite(replay["deformation_scale"]):
        raise FloatingPointError("shared deformation scale is non-finite")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sources = load_frozen_sources()
    representative = select_representative_condition(sources["j1"])
    print(
        "selected_condition="
        f"{representative['index']} "
        f"F_ratio={representative['forcing_ratio']:.1f} "
        f"omega_ratio={representative['frequency_ratio']:.2f}",
        flush=True,
    )
    banks = confirmation_banks()
    jumpgrad_frf = evaluate_jumpgrad_local_frf(sources, banks)
    replay = replay_selected_condition(sources)
    render_main_gif(replay)
    render_control_gif(replay)
    main_results = render_main_results(sources, jumpgrad_frf)
    render_gradient_story(sources["j1"])
    _validate_outputs(replay)
    print(
        f"jumpgrad_peak={main_results['peaks']['jumpgrad']['amplitude']:.12g} "
        f"wu_peak={main_results['peaks']['wu']['amplitude']:.12g} "
        f"outperforms_wu={main_results['outperforms_wu']}",
        flush=True,
    )
    print(f"deformation_scale={replay['deformation_scale']:.12g}", flush=True)
    print(f"outputs={OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    main()
