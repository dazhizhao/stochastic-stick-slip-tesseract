"""Render the frozen JumpGrad README visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


LIGHTWEIGHT_RENDER_REQUESTED = any(
    option in sys.argv[1:]
    for option in ("--held-out-only", "--pipeline-only")
)
if not LIGHTWEIGHT_RENDER_REQUESTED:
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import torch

    from stochastic_stick_slip.jumpgrad import (
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
        _assemble_system,
        build_variable_time_step_mechanics_batch_simulator,
    )
    from stochastic_stick_slip.wu_v2 import (
        DAMPING,
        DIAGNOSTIC_NUM_PERIODS,
        FORCING_AMPLITUDE,
        REFERENCE_PRELOAD,
        single_tone_forcing,
    )
    from stochastic_stick_slip.wu_v2_markov import (
        NUM_STEPS,
        PRELOAD_HIGH,
        PRELOAD_LOW,
        generate_hard_preload_history,
        markov_uniform_bank,
    )


GENERALIZATION_PATH = ROOT / "outputs/jumpgrad_generalization/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/jumpgrad_visuals"
HERO_PATH = OUTPUT_DIRECTORY / "passive_wu_jumpgrad.gif"
HELD_OUT_PATH = OUTPUT_DIRECTORY / "held_out.png"
PIPELINE_PATH = OUTPUT_DIRECTORY / "tesseract_pipeline.png"
EXPECTED_OUTPUTS = (
    "gradient_story.png",
    "held_out.png",
    "main_results.png",
    "optimization.gif",
    "passive_wu_jumpgrad.gif",
    "tesseract_pipeline.png",
)

SELECTED_CONDITION_INDEX = 5
SELECTED_STREAM = 12
SELECTED_REALIZATION = 0
HERO_NUM_REALIZATIONS = 16
HERO_CONTACT_COLUMNS = (16, 21, 25, 30)
HERO_CONTACT_NODES = (80, 105, 125, 150)
HERO_OPTIMIZED_Q = np.asarray(
    [-5.548885147350078, -1.6692728220424704], dtype=np.float64
)
FRAME_STRIDE = 4
NUM_GIF_FRAMES = 100
GIF_FPS = 20
FROZEN_HELD_OUT_CONDITIONS = np.asarray(
    [
        (amplitude, frequency)
        for amplitude in (0.9, 1.1, 1.3, 1.5)
        for frequency in (0.98, 1.06)
    ],
    dtype=np.float64,
)
if not LIGHTWEIGHT_RENDER_REQUESTED:
    STABLE_START = 20 * STEPS_PER_PERIOD
    STABLE_STOP = 24 * STEPS_PER_PERIOD
    TARGET_DEFORMATION = 0.22 * BEAM_LENGTH

FRAME_COLOR = "#28323A"
MESH_EDGE_COLOR = "#3B4650"
PASSIVE_PAD_COLOR = "#8A939B"
LOW_PAD_COLOR = "#13B8A6"
MID_PAD_COLOR = "#E7ECEF"
HIGH_PAD_COLOR = "#F59E0B"
WU_COLOR = "#5C8FBA"
INITIAL_COLOR = "#AFC2D4"
TRAINED_COLOR = "#1F7894"
HERO_COLORMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "jumpgrad_blues",
    mpl.colormaps["Blues"](np.linspace(0.28, 0.95, 256)),
)
PAD_CONTINUOUS_COLORMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "preload_state", (LOW_PAD_COLOR, MID_PAD_COLOR, HIGH_PAD_COLOR)
)

METHOD_LABELS = {
    "passive": "Passive",
    "wu": "Wu2019",
    "jumpgrad": "JumpGrad",
}

if not LIGHTWEIGHT_RENDER_REQUESTED:
    HERO_SYSTEM = _assemble_system(
        num_elements_x=32,
        num_elements_y=4,
        contact_columns=HERO_CONTACT_COLUMNS,
    )
    HERO_MECHANICS = build_variable_time_step_mechanics_batch_simulator(
        HERO_SYSTEM, return_full_displacement=True
    )


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 11.0,
            "axes.linewidth": 1.0,
        }
    )


def _style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.0)
    axis.tick_params(
        direction="in",
        top=False,
        right=False,
        colors=FRAME_COLOR,
        width=0.9,
        length=4.0,
    )


def load_generalization() -> dict:
    result = json.loads(GENERALIZATION_PATH.read_text())
    initial = np.asarray(
        result["normalized_responses"]["initial"], dtype=np.float64
    )
    trained = np.asarray(
        result["normalized_responses"]["trained"], dtype=np.float64
    )
    conditions = np.asarray(
        [
            [row["forcing_ratio"], row["frequency_ratio"]]
            for row in result["conditions"]
        ],
        dtype=np.float64,
    )
    valid = (
        result["configuration"]["stream_id"] == 13
        and result["configuration"]["iteration"] == 0
        and initial.shape == trained.shape == (8, 128)
        and np.allclose(conditions, FROZEN_HELD_OUT_CONDITIONS)
        and len(result["controller"]["frozen_final_theta"]) == 354
        and result["finite"] is True
        and np.all(np.isfinite(initial))
        and np.all(np.isfinite(trained))
    )
    if not valid:
        raise RuntimeError("generalization result contract changed")
    return result


def controller_q(theta: np.ndarray, condition: np.ndarray) -> np.ndarray:
    controller = build_jumpgrad_controller()
    descriptors = torch.from_numpy(condition_descriptors(condition))
    with torch.no_grad():
        q = functional_jumpgrad_controller(
            controller, torch.from_numpy(theta), descriptors
        )
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (len(condition), 2) or not np.all(np.isfinite(value)):
        raise FloatingPointError("frozen controller output is invalid")
    return value


def _full_nodal_field(free_field: np.ndarray) -> np.ndarray:
    full = np.zeros(
        free_field.shape[:-1] + (HERO_SYSTEM.num_total_dofs,),
        dtype=np.float64,
    )
    full[..., HERO_SYSTEM.free_dofs] = free_field
    return full.reshape(
        free_field.shape[:-1] + (len(HERO_SYSTEM.points), 2)
    )


def simulate_full_field(
    forcing: np.ndarray, preload: np.ndarray, time_step: float
) -> dict[str, np.ndarray]:
    outputs = HERO_MECHANICS(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        jnp.asarray(forcing[None, :], dtype=jnp.float64),
        jnp.asarray(preload[None, :, :], dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    tip = np.asarray(outputs[0][0])
    free_field = np.asarray(outputs[5][0])
    if not np.all(np.isfinite(tip)) or not np.all(np.isfinite(free_field)):
        raise FloatingPointError("four-contact full-field replay is not finite")
    observed = free_field @ np.asarray(HERO_SYSTEM.observation)
    if not np.allclose(observed, tip, rtol=1e-10, atol=1e-12):
        raise AssertionError("full-field replay disagrees with scalar mechanics")
    return {"tip": tip, "field": _full_nodal_field(free_field)}


def replay_hero(generalization: dict) -> dict:
    condition = FROZEN_HELD_OUT_CONDITIONS[[SELECTED_CONDITION_INDEX]]
    theta = np.asarray(
        generalization["controller"]["frozen_final_theta"], dtype=np.float64
    )
    initial_q = controller_q(theta, condition)[0]
    expected_initial_q = np.asarray(
        [-5.048300073899528, -2.2099613642207494], dtype=np.float64
    )
    if not np.allclose(initial_q, expected_initial_q, rtol=1e-13, atol=1e-13):
        raise RuntimeError("registered hero controller output changed")
    if not np.array_equal(
        HERO_SYSTEM.contact_nodes, np.asarray(HERO_CONTACT_NODES)
    ):
        raise RuntimeError("registered four-contact hero layout changed")
    expected_positions = np.column_stack(
        (
            np.asarray(HERO_CONTACT_COLUMNS, dtype=np.float64) / 32.0,
            np.zeros(len(HERO_CONTACT_COLUMNS), dtype=np.float64),
        )
    )
    if not np.allclose(
        HERO_SYSTEM.contact_coordinates,
        expected_positions,
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("registered four-contact positions changed")

    q = HERO_OPTIMIZED_Q.copy()
    forcing_ratio, frequency_ratio = condition[0]
    omega = float(OMEGA_R * frequency_ratio)
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE * forcing_ratio,
        omega,
        DIAGNOSTIC_NUM_PERIODS,
    )
    times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    num_contacts = len(HERO_CONTACT_COLUMNS)
    passive_preload = np.full(
        (NUM_STEPS, num_contacts), REFERENCE_PRELOAD
    )
    wu_scalar = REFERENCE_PRELOAD + WU_AMPLITUDE * np.sin(
        2.0 * omega * times + WU_PHASE
    )
    wu_preload = np.repeat(wu_scalar[:, None], num_contacts, axis=1)
    tape = markov_uniform_bank(
        HERO_NUM_REALIZATIONS,
        stream_id=SELECTED_STREAM,
        iteration=0,
        num_contacts=num_contacts,
    )[SELECTED_CONDITION_INDEX, SELECTED_REALIZATION]
    if any(
        np.array_equal(tape[:, first], tape[:, second])
        for first in range(num_contacts)
        for second in range(first + 1, num_contacts)
    ):
        raise RuntimeError("registered four-contact tapes are not independent")
    modes, jumpgrad_preload, _, _ = generate_hard_preload_history(
        jnp.asarray(q),
        jnp.asarray(times),
        jnp.asarray(tape),
        jnp.asarray(omega),
        jnp.asarray(time_step),
    )
    modes = np.asarray(modes)
    jumpgrad_preload = np.asarray(jumpgrad_preload)
    if (
        jumpgrad_preload.shape != (NUM_STEPS, num_contacts)
        or modes.shape != (NUM_STEPS, num_contacts)
        or any(
            set(np.unique(jumpgrad_preload[:, contact]))
            != {PRELOAD_LOW, PRELOAD_HIGH}
            for contact in range(num_contacts)
        )
    ):
        raise RuntimeError("registered hero does not use both hard preloads")
    methods = {
        "passive": simulate_full_field(forcing, passive_preload, time_step),
        "wu": simulate_full_field(forcing, wu_preload, time_step),
        "jumpgrad": simulate_full_field(
            forcing, jumpgrad_preload, time_step
        ),
    }
    methods["passive"]["preload"] = passive_preload
    methods["wu"]["preload"] = wu_preload
    methods["jumpgrad"]["preload"] = jumpgrad_preload
    methods["jumpgrad"]["modes"] = modes
    frame_indices = np.arange(STABLE_START, STABLE_STOP, FRAME_STRIDE)
    if frame_indices.shape != (NUM_GIF_FRAMES,):
        raise AssertionError("hero frame count changed")

    fields = [value["field"] for value in methods.values()]
    displayed = [field[frame_indices] for field in fields]
    nodal_values = np.concatenate(
        [np.linalg.norm(field, axis=2).ravel() for field in displayed]
    )
    maximum = float(np.max(nodal_values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("hero replay has no finite deformation")
    deformation_scale = TARGET_DEFORMATION / maximum

    cell_values = []
    for field in displayed:
        magnitude = np.linalg.norm(field, axis=2)
        cell_values.append(
            np.mean(magnitude[:, HERO_SYSTEM.cells], axis=2)
        )
    cell_values = np.concatenate([value.ravel() for value in cell_values])
    color_limits = (float(np.min(cell_values)), float(np.max(cell_values)))

    vertices = np.concatenate(
        [
            HERO_SYSTEM.points[None, :, :] + deformation_scale * field
            for field in displayed
        ],
        axis=0,
    )
    minimum = np.min(vertices, axis=(0, 1))
    maximum_vertex = np.max(vertices, axis=(0, 1))
    span = np.maximum(
        maximum_vertex - minimum, np.asarray([BEAM_LENGTH, BEAM_HEIGHT])
    )
    padding = 0.07 * span
    axis_limits = (
        (float(minimum[0] - padding[0]), float(maximum_vertex[0] + padding[0])),
        (float(minimum[1] - padding[1]), float(maximum_vertex[1] + padding[1])),
    )
    return {
        "q": q,
        "methods": methods,
        "frame_indices": frame_indices,
        "deformation_scale": deformation_scale,
        "color_limits": color_limits,
        "axis_limits": axis_limits,
    }


def _add_beam_panel(axis, method: str, normalization, limits):
    collection = PolyCollection(
        HERO_SYSTEM.points[HERO_SYSTEM.cells],
        cmap=HERO_COLORMAP,
        norm=normalization,
        edgecolor=MESH_EDGE_COLOR,
        linewidth=0.25,
    )
    axis.add_collection(collection)
    contacts = axis.scatter(
        [],
        [],
        s=28,
        marker="o",
        facecolor=PASSIVE_PAD_COLOR,
        edgecolor=FRAME_COLOR,
        linewidth=0.75,
        zorder=5,
    )
    axis.plot(
        [0.0, 0.0],
        [-0.045, BEAM_HEIGHT + 0.045],
        color=FRAME_COLOR,
        linewidth=2.2,
    )
    for offset in np.linspace(-0.04, 0.04, 5):
        axis.plot(
            [-0.035, 0.0],
            [offset - 0.025, offset],
            color=FRAME_COLOR,
            linewidth=0.75,
        )
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.text(
        0.5,
        0.99,
        METHOD_LABELS[method],
        transform=axis.transAxes,
        ha="center",
        va="top",
        color=FRAME_COLOR,
        fontsize=15.0,
        fontweight="bold",
    )
    return collection, contacts


def _pad_colors(method: str, data: dict, step: int) -> np.ndarray:
    if method == "passive":
        return np.tile(
            mpl.colors.to_rgba(PASSIVE_PAD_COLOR),
            (len(HERO_CONTACT_COLUMNS), 1),
        )
    if method == "wu":
        normalized = np.clip(
            (data["preload"][step] - PRELOAD_LOW)
            / (PRELOAD_HIGH - PRELOAD_LOW),
            0.0,
            1.0,
        )
        return PAD_CONTINUOUS_COLORMAP(normalized)

    low = np.asarray(mpl.colors.to_rgba(LOW_PAD_COLOR))
    high = np.asarray(mpl.colors.to_rgba(HIGH_PAD_COLOR))
    return np.where(data["modes"][step, :, None], high, low)


def render_hero(replay: dict) -> None:
    _configure_plotting()
    figure = plt.figure(figsize=(12.0, 4.30), dpi=100, facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        left=0.018,
        right=0.982,
        bottom=0.245,
        top=0.91,
        wspace=0.025,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    normalization = mpl.colors.Normalize(*replay["color_limits"])
    collections = {}
    contacts = {}
    for axis, method in zip(
        axes, ("passive", "wu", "jumpgrad"), strict=True
    ):
        collections[method], contacts[method] = _add_beam_panel(
            axis, method, normalization, replay["axis_limits"]
        )

    colorbar_axis = figure.add_axes([0.15, 0.180, 0.70, 0.050])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=normalization, cmap=HERO_COLORMAP),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Displacement magnitude", fontsize=13.0, labelpad=7.0)
    colorbar.outline.set_linewidth(0.9)
    colorbar.ax.tick_params(direction="in", labelsize=10.5, length=4.0)

    def update(step):
        artists = []
        for method in ("passive", "wu", "jumpgrad"):
            data = replay["methods"][method]
            field = data["field"][int(step)]
            deformed = (
                HERO_SYSTEM.points + replay["deformation_scale"] * field
            )
            collections[method].set_verts(deformed[HERO_SYSTEM.cells])
            magnitude = np.linalg.norm(field, axis=1)
            collections[method].set_array(
                np.mean(magnitude[HERO_SYSTEM.cells], axis=1)
            )
            contacts[method].set_offsets(
                deformed[HERO_SYSTEM.contact_nodes]
            )
            contacts[method].set_facecolors(
                _pad_colors(method, data, int(step))
            )
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
    animation.save(HERO_PATH, writer=PillowWriter(fps=GIF_FPS), dpi=100)
    plt.close(figure)


def render_held_out(result: dict) -> tuple[float, float, float, int]:
    _configure_plotting()
    initial_aggregate = np.asarray(
        result["aggregate_reduction_percent"]["initial"], dtype=np.float64
    )
    trained_aggregate = np.asarray(
        result["aggregate_reduction_percent"]["trained"], dtype=np.float64
    )
    stored_improvement = np.asarray(
        result["paired_trained_minus_initial_percent"]["values"],
        dtype=np.float64,
    )
    improvement = trained_aggregate - initial_aggregate
    if (
        initial_aggregate.shape != (128,)
        or trained_aggregate.shape != (128,)
        or stored_improvement.shape != (128,)
        or not np.all(np.isfinite(initial_aggregate))
        or not np.all(np.isfinite(trained_aggregate))
        or not np.all(np.isfinite(stored_improvement))
        or not np.allclose(
            improvement, stored_improvement, rtol=1e-12, atol=1e-12
        )
    ):
        raise RuntimeError("held-out plotting data contract changed")
    initial_mean = float(np.mean(initial_aggregate))
    trained_mean = float(np.mean(trained_aggregate))
    improvement_mean = float(np.mean(improvement))
    improved_count = int(np.count_nonzero(improvement > 0.0))

    figure = plt.figure(figsize=(8.6, 3.35))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.25),
        height_ratios=(0.82, 1.18),
        hspace=0.075,
        wspace=0.32,
    )
    upper_axis = figure.add_subplot(grid[0, 0])
    lower_axis = figure.add_subplot(grid[1, 0], sharex=upper_axis)
    histogram_axis = figure.add_subplot(grid[:, 1])
    positions = np.asarray([0.0, 1.0])
    distributions = (initial_aggregate, trained_aggregate)
    distribution_colors = (INITIAL_COLOR, TRAINED_COLOR)
    distribution_means = (initial_mean, trained_mean)
    for axis in (upper_axis, lower_axis):
        violin = axis.violinplot(
            distributions,
            positions=positions,
            widths=0.70,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(
            violin["bodies"], distribution_colors, strict=True
        ):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.38)
            body.set_linewidth(1.35)
        boxes = axis.boxplot(
            distributions,
            positions=positions,
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": FRAME_COLOR, "linewidth": 1.8},
            whiskerprops={"color": FRAME_COLOR, "linewidth": 1.1},
            capprops={"color": FRAME_COLOR, "linewidth": 1.1},
            boxprops={"edgecolor": FRAME_COLOR, "linewidth": 1.1},
        )
        for box, color in zip(
            boxes["boxes"], distribution_colors, strict=True
        ):
            box.set_facecolor(color)
            box.set_alpha(0.78)
        axis.scatter(
            positions,
            distribution_means,
            marker="o",
            color=distribution_colors,
            edgecolor="white",
            linewidth=1.1,
            s=62,
            zorder=4,
        )
        _style_axis(axis)

    initial_span = float(np.ptp(initial_aggregate))
    trained_span = float(np.ptp(trained_aggregate))
    lower_axis.set_ylim(
        float(np.min(initial_aggregate)) - 0.12 * initial_span,
        float(np.max(initial_aggregate)) + 0.24 * initial_span,
    )
    upper_axis.set_ylim(
        float(np.min(trained_aggregate)) - 0.25 * trained_span,
        float(np.max(trained_aggregate)) + 0.42 * trained_span,
    )
    upper_axis.spines["bottom"].set_visible(False)
    lower_axis.spines["top"].set_visible(False)
    upper_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    lower_axis.set_xticks(positions, ["Initial", "Trained\nJumpGrad"])
    upper_axis.text(
        0.94,
        0.93,
        f"Trained mean = {trained_mean:.2f}%",
        transform=upper_axis.transAxes,
        color=TRAINED_COLOR,
        fontsize=10.5,
        fontweight="bold",
        ha="right",
        va="top",
    )
    lower_axis.text(
        0.06,
        0.94,
        f"Initial mean = {initial_mean:.2f}%",
        transform=lower_axis.transAxes,
        color=FRAME_COLOR,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    upper_axis.text(
        -0.075,
        1.06,
        "a",
        transform=upper_axis.transAxes,
        color=FRAME_COLOR,
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    break_size = 0.014
    break_style = {
        "color": FRAME_COLOR,
        "clip_on": False,
        "linewidth": 1.15,
    }
    upper_axis.plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=upper_axis.transAxes,
        **break_style,
    )
    upper_axis.plot(
        (1.0 - break_size, 1.0 + break_size),
        (-break_size, break_size),
        transform=upper_axis.transAxes,
        **break_style,
    )
    lower_axis.plot(
        (-break_size, break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=lower_axis.transAxes,
        **break_style,
    )
    lower_axis.plot(
        (1.0 - break_size, 1.0 + break_size),
        (1.0 - break_size, 1.0 + break_size),
        transform=lower_axis.transAxes,
        **break_style,
    )

    bin_edges = np.histogram_bin_edges(improvement, bins="fd")
    counts, _, _ = histogram_axis.hist(
        improvement,
        bins=bin_edges,
        color=TRAINED_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.9,
    )
    improvement_min = float(np.min(improvement))
    improvement_max = float(np.max(improvement))
    improvement_span = improvement_max - improvement_min
    histogram_axis.set_xlim(
        improvement_min - 0.06 * improvement_span,
        improvement_max + 0.06 * improvement_span,
    )
    histogram_axis.set_ylim(0.0, float(np.max(counts)) * 1.18)
    histogram_axis.set_xlabel("Improvement in aggregate reduction (pts)")
    histogram_axis.set_ylabel("Fresh realizations")
    histogram_axis.text(
        -0.075,
        1.06,
        "b",
        transform=histogram_axis.transAxes,
        color=FRAME_COLOR,
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    _style_axis(histogram_axis)

    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.97)
    figure.text(
        0.025,
        0.58,
        "Aggregate reduction vs passive (%)",
        color=FRAME_COLOR,
        fontsize=14.0,
        rotation=90,
        ha="center",
        va="center",
    )
    figure.savefig(
        HELD_OUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return initial_mean, trained_mean, improvement_mean, improved_count


def render_tesseract_pipeline(path: Path = PIPELINE_PATH) -> None:
    """Render the two Tesseract blocks with separate forward/backward lanes."""
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "text.color": FRAME_COLOR,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.0, 4.0), dpi=100)
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")

        box_y = 0.15
        box_width = 0.29
        box_height = 0.70
        boxes = (
            {
                "x": 0.15,
                "accent": "#4F86B4",
                "header": "TESSERACT BLOCK 01",
                "name": "JumpGrad Controller",
                "lines": (
                    ("PyTorch MLP", 0.50),
                    ("VJP: PyTorch autograd", 0.38),
                ),
            },
            {
                "x": 0.56,
                "accent": "#16879A",
                "header": "TESSERACT BLOCK 02",
                "name": "Stochastic Mechanics",
                "lines": (
                    ("Hard Markov LOW / HIGH", 0.52),
                    ("JAX-FEM + Jenkins friction", 0.42),
                    ("VJP: CRN centered finite diff.", 0.32),
                ),
            },
        )

        for spec in boxes:
            x = spec["x"]
            axis.add_patch(
                FancyBboxPatch(
                    (x, box_y),
                    box_width,
                    box_height,
                    boxstyle="round,pad=0.008,rounding_size=0.012",
                    facecolor="white",
                    edgecolor=FRAME_COLOR,
                    linewidth=1.2,
                )
            )
            axis.add_patch(
                Rectangle(
                    (x + 0.012, box_y + box_height - 0.085),
                    box_width - 0.024,
                    0.067,
                    facecolor=spec["accent"],
                    edgecolor="none",
                )
            )
            axis.text(
                x + box_width / 2.0,
                box_y + box_height - 0.0515,
                spec["header"],
                color="white",
                fontsize=11.0,
                fontweight="bold",
                ha="center",
                va="center",
            )
            axis.text(
                x + box_width / 2.0,
                0.67,
                spec["name"],
                fontsize=16.0,
                fontweight="bold",
                ha="center",
                va="center",
            )
            for line, line_y in spec["lines"]:
                axis.text(
                    x + box_width / 2.0,
                    line_y,
                    line,
                    fontsize=12.0,
                    ha="center",
                    va="center",
                )

        arrow = {
            "arrowstyle": "-|>",
            "color": FRAME_COLOR,
            "linewidth": 1.4,
            "mutation_scale": 14,
            "shrinkA": 0,
            "shrinkB": 0,
        }
        forward_y = 0.60
        backward_y = 0.25

        axis.text(
            0.065,
            forward_y,
            "Operating\ncondition",
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.25,
        )
        axis.annotate(
            "", xy=(0.142, forward_y), xytext=(0.108, forward_y), arrowprops=arrow
        )

        axis.annotate(
            "", xy=(0.552, forward_y), xytext=(0.448, forward_y), arrowprops=arrow
        )
        axis.text(
            0.50,
            0.655,
            "q = [a2, b2]",
            fontsize=11.5,
            ha="center",
            va="bottom",
        )
        axis.annotate(
            "", xy=(0.448, backward_y), xytext=(0.552, backward_y), arrowprops=arrow
        )
        axis.text(
            0.50,
            0.205,
            "q cotangent",
            fontsize=11.0,
            ha="center",
            va="top",
        )

        axis.annotate(
            "", xy=(0.890, forward_y), xytext=(0.858, forward_y), arrowprops=arrow
        )
        axis.text(
            0.94,
            forward_y,
            "Vibration\nobjective",
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.25,
        )
        axis.annotate(
            "", xy=(0.858, backward_y), xytext=(0.898, backward_y), arrowprops=arrow
        )
        axis.text(
            0.92,
            0.205,
            "loss cotangent",
            fontsize=11.0,
            ha="center",
            va="top",
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, facecolor="white")
        plt.close(figure)


def validate_outputs(*, validate_hero: bool = True) -> None:
    actual = tuple(
        sorted(path.name for path in OUTPUT_DIRECTORY.iterdir() if path.is_file())
    )
    if actual != EXPECTED_OUTPUTS:
        raise AssertionError(f"visual output contract changed: {actual}")
    if validate_hero:
        with Image.open(HERO_PATH) as image:
            durations = []
            for frame in range(image.n_frames):
                image.seek(frame)
                durations.append(image.info.get("duration"))
            if (
                image.n_frames != NUM_GIF_FRAMES
                or image.info.get("loop") != 0
                or image.size != (1200, 430)
                or set(durations) != {1000 // GIF_FPS}
            ):
                raise AssertionError("hero GIF metadata changed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frozen JumpGrad README visualizations."
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--held-out-only",
        action="store_true",
        help="render only the fresh-seed held-out comparison",
    )
    output_mode.add_argument(
        "--hero-only",
        action="store_true",
        help="render only the frozen three-method hero animation",
    )
    output_mode.add_argument(
        "--pipeline-only",
        action="store_true",
        help="render only the two-Tesseract pipeline figure",
    )
    arguments = parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.pipeline_only:
        render_tesseract_pipeline()
        print(f"output={PIPELINE_PATH.relative_to(ROOT)}")
        return
    result = load_generalization()
    if arguments.held_out_only:
        initial_mean, trained_mean, improvement_mean, improved_count = (
            render_held_out(result)
        )
        validate_outputs(validate_hero=False)
        print(f"initial_mean_reduction_percent={initial_mean:.12g}")
        print(f"trained_mean_reduction_percent={trained_mean:.12g}")
        print(f"mean_paired_improvement_points={improvement_mean:.12g}")
        print(f"improved={improved_count}/128")
        print(f"output={HELD_OUT_PATH.relative_to(ROOT)}")
        return
    replay = replay_hero(result)
    render_hero(replay)
    if arguments.hero_only:
        validate_outputs()
        print(
            f"hero_frames={NUM_GIF_FRAMES} deformation_scale="
            f"{replay['deformation_scale']:.12g}"
        )
        print(f"output={HERO_PATH.relative_to(ROOT)}")
        return
    initial_mean, trained_mean, improvement_mean, improved_count = (
        render_held_out(result)
    )
    validate_outputs()
    print(
        f"hero_frames={NUM_GIF_FRAMES} deformation_scale="
        f"{replay['deformation_scale']:.12g}"
    )
    print(f"initial_mean_reduction_percent={initial_mean:.12g}")
    print(f"trained_mean_reduction_percent={trained_mean:.12g}")
    print(f"mean_paired_improvement_points={improvement_mean:.12g}")
    print(f"improved={improved_count}/128")
    print(f"outputs={HERO_PATH.relative_to(ROOT)},{HELD_OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
