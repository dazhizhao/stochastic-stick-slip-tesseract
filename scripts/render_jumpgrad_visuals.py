"""Render the fresh-seed held-out figure and frozen JumpGrad hero GIF."""

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
import numpy as np
from PIL import Image


HELD_OUT_ONLY_REQUESTED = "--held-out-only" in sys.argv[1:]
if not HELD_OUT_ONLY_REQUESTED:
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
        generate_hard_preload_history,
        markov_uniform_bank,
    )


GENERALIZATION_PATH = ROOT / "outputs/jumpgrad_generalization/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/jumpgrad_visuals"
HERO_PATH = OUTPUT_DIRECTORY / "passive_wu_jumpgrad.gif"
HELD_OUT_PATH = OUTPUT_DIRECTORY / "held_out.png"
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
if not HELD_OUT_ONLY_REQUESTED:
    STABLE_START = 20 * STEPS_PER_PERIOD
    STABLE_STOP = 24 * STEPS_PER_PERIOD
    TARGET_DEFORMATION = 0.22 * BEAM_LENGTH

FRAME_COLOR = "#28323A"
MESH_EDGE_COLOR = "#3B4650"
WU_COLOR = "#5C8FBA"
TRAINED_COLOR = "#1F7894"
PLASMA = mpl.colormaps["plasma"]

METHOD_LABELS = {
    "passive": "Passive",
    "wu": "Wu2019",
    "jumpgrad": "JumpGrad",
}

if not HELD_OUT_ONLY_REQUESTED:
    FULL_MECHANICS = build_variable_time_step_mechanics_batch_simulator(
        SYSTEM, return_full_displacement=True
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
        free_field.shape[:-1] + (SYSTEM.num_total_dofs,), dtype=np.float64
    )
    full[..., SYSTEM.free_dofs] = free_field
    return full.reshape(free_field.shape[:-1] + (len(SYSTEM.points), 2))


def simulate_full_field(
    forcing: np.ndarray, preload: np.ndarray, time_step: float
) -> dict[str, np.ndarray]:
    outputs = FULL_MECHANICS(
        jnp.asarray(DAMPING, dtype=jnp.float64),
        jnp.asarray(forcing[None, :], dtype=jnp.float64),
        jnp.asarray(preload[None, :, :], dtype=jnp.float64),
        jnp.asarray(time_step, dtype=jnp.float64),
    )
    tip = np.asarray(outputs[0][0])
    free_field = np.asarray(outputs[5][0])
    observed = free_field @ np.asarray(SYSTEM.observation)
    if not np.allclose(observed, tip, rtol=1e-10, atol=1e-12):
        raise AssertionError("full-field replay disagrees with scalar mechanics")
    return {"tip": tip, "field": _full_nodal_field(free_field)}


def replay_hero(generalization: dict) -> dict:
    condition = FROZEN_HELD_OUT_CONDITIONS[[SELECTED_CONDITION_INDEX]]
    theta = np.asarray(
        generalization["controller"]["frozen_final_theta"], dtype=np.float64
    )
    q = controller_q(theta, condition)[0]
    forcing_ratio, frequency_ratio = condition[0]
    omega = float(OMEGA_R * frequency_ratio)
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE * forcing_ratio,
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
    )[SELECTED_CONDITION_INDEX, SELECTED_REALIZATION]
    _, jumpgrad_preload, _, _ = generate_hard_preload_history(
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
        cell_values.append(np.mean(magnitude[:, SYSTEM.cells], axis=2))
    cell_values = np.concatenate([value.ravel() for value in cell_values])
    color_limits = (float(np.min(cell_values)), float(np.max(cell_values)))

    vertices = np.concatenate(
        [
            SYSTEM.points[None, :, :] + deformation_scale * field
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
        "methods": methods,
        "frame_indices": frame_indices,
        "deformation_scale": deformation_scale,
        "color_limits": color_limits,
        "axis_limits": axis_limits,
    }


def _add_beam_panel(axis, method: str, normalization, limits):
    collection = PolyCollection(
        SYSTEM.points[SYSTEM.cells],
        cmap=PLASMA,
        norm=normalization,
        edgecolor=MESH_EDGE_COLOR,
        linewidth=0.25,
    )
    axis.add_collection(collection)
    contacts = axis.scatter(
        [],
        [],
        s=28,
        facecolor="white",
        edgecolor=FRAME_COLOR,
        linewidth=1.25,
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


def render_hero(replay: dict) -> None:
    _configure_plotting()
    figure = plt.figure(figsize=(12.0, 4.30), dpi=100, facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        left=0.018,
        right=0.982,
        bottom=0.27,
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

    colorbar_axis = figure.add_axes([0.15, 0.115, 0.70, 0.050])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=normalization, cmap=PLASMA),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Displacement magnitude", fontsize=13.0, labelpad=7.0)
    colorbar.outline.set_linewidth(0.9)
    colorbar.ax.tick_params(direction="in", labelsize=10.5, length=4.0)

    def update(step):
        artists = []
        for method in ("passive", "wu", "jumpgrad"):
            field = replay["methods"][method]["field"][int(step)]
            deformed = SYSTEM.points + replay["deformation_scale"] * field
            collections[method].set_verts(deformed[SYSTEM.cells])
            magnitude = np.linalg.norm(field, axis=1)
            collections[method].set_array(
                np.mean(magnitude[SYSTEM.cells], axis=1)
            )
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
    animation.save(HERO_PATH, writer=PillowWriter(fps=GIF_FPS), dpi=100)
    plt.close(figure)


def render_held_out(result: dict) -> tuple[float, int]:
    _configure_plotting()
    trained_normalized = np.asarray(
        result["normalized_responses"]["trained"], dtype=np.float64
    )
    trained_reduction = 100.0 * (1.0 - trained_normalized)
    trained_aggregate = np.mean(trained_reduction, axis=0)
    wu = np.asarray(
        result["references"]["wu2019_reduction_percent"], dtype=np.float64
    )
    stored_aggregate = np.asarray(
        result["aggregate_reduction_percent"]["trained"], dtype=np.float64
    )
    if (
        trained_reduction.shape != (8, 128)
        or wu.shape != (8,)
        or stored_aggregate.shape != (128,)
        or not np.all(np.isfinite(trained_reduction))
        or not np.all(np.isfinite(wu))
        or not np.allclose(
            trained_aggregate, stored_aggregate, rtol=1e-12, atol=1e-12
        )
    ):
        raise RuntimeError("held-out plotting data contract changed")
    wu_aggregate = float(np.mean(wu))
    above_wu_count = int(np.count_nonzero(trained_aggregate > wu_aggregate))
    aggregate_mean = float(np.mean(trained_aggregate))

    figure, axis = plt.subplots(figsize=(8.6, 2.65))
    violin = axis.violinplot(
        [trained_aggregate],
        positions=[0.0],
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        orientation="horizontal",
    )
    for body in violin["bodies"]:
        body.set_facecolor(TRAINED_COLOR)
        body.set_edgecolor(TRAINED_COLOR)
        body.set_alpha(0.38)
        body.set_linewidth(1.35)
    boxes = axis.boxplot(
        [trained_aggregate],
        positions=[0.0],
        widths=0.17,
        showfliers=False,
        patch_artist=True,
        orientation="horizontal",
        medianprops={"color": FRAME_COLOR, "linewidth": 1.8},
        whiskerprops={"color": FRAME_COLOR, "linewidth": 1.1},
        capprops={"color": FRAME_COLOR, "linewidth": 1.1},
        boxprops={"edgecolor": FRAME_COLOR, "linewidth": 1.1},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(TRAINED_COLOR)
        box.set_alpha(0.72)
    axis.scatter(
        [aggregate_mean],
        [0.0],
        marker="o",
        color=TRAINED_COLOR,
        edgecolor="white",
        linewidth=1.1,
        s=70,
        zorder=4,
    )
    axis.axvline(
        wu_aggregate,
        color=WU_COLOR,
        linestyle="--",
        linewidth=1.9,
        zorder=1,
    )
    axis.text(
        wu_aggregate + 0.08,
        0.34,
        f"Wu2019 = {wu_aggregate:.1f}%",
        color=WU_COLOR,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    axis.text(
        aggregate_mean,
        0.20,
        f"Mean = {aggregate_mean:.2f}%",
        color=TRAINED_COLOR,
        fontsize=11.0,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axis.text(
        0.975,
        0.90,
        f"{above_wu_count}/128 above Wu2019",
        transform=axis.transAxes,
        color=FRAME_COLOR,
        fontsize=11.5,
        fontweight="bold",
        ha="right",
        va="top",
    )
    lower = min(wu_aggregate, float(np.min(trained_aggregate))) - 0.45
    upper = max(wu_aggregate, float(np.max(trained_aggregate))) + 0.45
    axis.set_xlim(lower, upper)
    axis.set_ylim(-0.53, 0.53)
    axis.set_yticks([0.0], ["JumpGrad\nfresh seeds"])
    axis.set_xlabel("Aggregate reduction vs passive (%)")
    _style_axis(axis)

    figure.tight_layout()
    figure.savefig(
        HELD_OUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return wu_aggregate, above_wu_count


def validate_outputs(*, validate_hero: bool = True) -> None:
    actual = tuple(
        sorted(path.name for path in OUTPUT_DIRECTORY.iterdir() if path.is_file())
    )
    if actual != EXPECTED_OUTPUTS:
        raise AssertionError(f"visual output contract changed: {actual}")
    if validate_hero:
        with Image.open(HERO_PATH) as image:
            if image.n_frames != NUM_GIF_FRAMES or image.info.get("loop") != 0:
                raise AssertionError("hero GIF metadata changed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frozen JumpGrad README visualizations."
    )
    parser.add_argument(
        "--held-out-only",
        action="store_true",
        help="render only the fresh-seed held-out comparison",
    )
    arguments = parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    result = load_generalization()
    if arguments.held_out_only:
        wu_aggregate, above_wu_count = render_held_out(result)
        validate_outputs(validate_hero=False)
        print(f"wu2019_aggregate_reduction_percent={wu_aggregate:.12g}")
        print(f"above_wu2019={above_wu_count}/128")
        print(f"output={HELD_OUT_PATH.relative_to(ROOT)}")
        return
    replay = replay_hero(result)
    render_hero(replay)
    wu_aggregate, above_wu_count = render_held_out(result)
    validate_outputs()
    print(
        f"hero_frames={NUM_GIF_FRAMES} deformation_scale="
        f"{replay['deformation_scale']:.12g}"
    )
    print(f"wu2019_aggregate_reduction_percent={wu_aggregate:.12g}")
    print(f"above_wu2019={above_wu_count}/128")
    print(f"outputs={HERO_PATH.relative_to(ROOT)},{HELD_OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
