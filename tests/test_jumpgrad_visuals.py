from pathlib import Path

import numpy as np
from PIL import Image

from scripts import render_jumpgrad_visuals as visuals
from stochastic_stick_slip.jumpgrad import (
    HELD_OUT_CONDITIONS,
    TRAINING_CONDITIONS,
)
from stochastic_stick_slip.wu_v2 import (
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import NUM_STEPS


def test_representative_condition_and_frozen_controller_outputs():
    sources = visuals.load_frozen_sources()
    selected = visuals.select_representative_condition(sources["j1"])
    assert selected["index"] == 5
    assert selected["forcing_ratio"] == 1.3
    assert selected["frequency_ratio"] == 1.06

    theta = np.asarray(sources["j1"]["training"]["final_theta"])
    conditions = np.vstack((TRAINING_CONDITIONS, HELD_OUT_CONDITIONS))
    q = visuals.controller_q(theta, conditions)
    saved = np.asarray(
        [row["q"] for row in sources["j1"]["controller_outputs"]["training"]]
        + [row["q"] for row in sources["j1"]["controller_outputs"]["held_out"]]
    )
    np.testing.assert_allclose(q, saved, rtol=1e-10, atol=1e-12)
    assert np.max(np.linalg.norm(q[:, None] - q[None, :], axis=2)) > 0.0


def test_registered_streams_are_fixed_and_distinct():
    assert visuals.CONFIRMATION_STREAMS == (5, 6, 7, 8)
    assert visuals.SELECTED_STREAM == 12
    assert visuals.SELECTED_REALIZATION == 0
    banks = visuals.confirmation_banks()
    assert tuple(banks) == visuals.CONFIRMATION_STREAMS
    assert all(bank.shape == (8, 8, NUM_STEPS + 1, 2) for bank in banks.values())
    repeated = visuals.markov_uniform_bank(8, stream_id=5, iteration=0)
    np.testing.assert_array_equal(banks[5], repeated)
    for left, right in zip(visuals.CONFIRMATION_STREAMS[:-1], visuals.CONFIRMATION_STREAMS[1:]):
        assert not np.array_equal(banks[left], banks[right])


def test_local_frf_grid_and_deployed_q_are_condition_aware():
    sources = visuals.load_frozen_sources()
    ratios = sources["ratios"]
    conditions = visuals.local_frf_conditions(ratios)
    np.testing.assert_allclose(ratios, np.linspace(0.90, 1.10, 21))
    np.testing.assert_allclose(conditions[:, 0], 1.0)
    np.testing.assert_allclose(conditions[:, 1], ratios)
    theta = np.asarray(sources["j1"]["training"]["final_theta"])
    q = visuals.controller_q(theta, conditions)
    assert q.shape == (21, 2)
    assert np.max(np.linalg.norm(q - q[10], axis=1)) > 0.0


def test_full_field_replay_matches_scalar_observation():
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, visuals.OMEGA_R, num_periods=1
    )
    preload = np.full((len(forcing), 2), REFERENCE_PRELOAD)
    replay = visuals.simulate_full_field(forcing, preload, time_step)
    assert replay["tip"].shape == (100,)
    assert replay["free_field"].shape == (100, visuals.SYSTEM.num_free_dofs)
    assert replay["field"].shape == (100, len(visuals.SYSTEM.points), 2)
    assert np.all(np.isfinite(replay["field"]))


def test_shared_deformation_scale_uses_all_methods():
    fields = []
    for maximum in (1.0, 2.0, 4.0):
        field = np.zeros((NUM_STEPS, 2, 2), dtype=np.float64)
        field[visuals.STABLE_START : visuals.STABLE_STOP, 0, 1] = maximum
        fields.append(field)
    scale, maximum = visuals.shared_deformation_scale(fields)
    assert maximum == 4.0
    assert scale == visuals.TARGET_DEFORMATION / 4.0


def test_shared_hero_color_limits_use_every_method_and_frame():
    indices = np.asarray([visuals.STABLE_START, visuals.STABLE_START + 1])
    fields = []
    for value in (1.0, 2.0, 5.0):
        field = np.zeros((NUM_STEPS, len(visuals.SYSTEM.points), 2))
        field[indices, :, 0] = value
        fields.append(field)
    lower, upper = visuals.shared_cell_color_limits(fields, indices)
    assert lower == 1.0
    assert upper == 5.0


def test_optimization_replay_registration_and_saved_history():
    j1 = visuals.load_frozen_sources()["j1"]
    q_history = np.asarray(j1["training"]["q_history"])
    assert visuals.OPTIMIZATION_CONDITION_INDEX == 2
    assert visuals.MONITOR_STREAM == 11
    assert visuals.OPTIMIZATION_REALIZATION == 0
    assert q_history.shape == (101, 8, 2)
    np.testing.assert_allclose(
        visuals.TRAINING_CONDITIONS[visuals.OPTIMIZATION_CONDITION_INDEX],
        [1.0, 1.0],
    )
    assert len(j1["training"]["monitor_iterations"]) == 11


def test_held_out_visual_uses_all_eight_frozen_conditions():
    values = visuals.held_out_reductions(visuals.load_frozen_sources()["j1"])
    assert set(values) == {"wu", "jumpgrad"}
    assert all(value.shape == (8,) for value in values.values())


def test_gif_contract_and_visual_asset_order(tmp_path: Path):
    path = tmp_path / "test.gif"
    frames = [
        Image.new("RGB", (4, 3), color=(index, 2 * index, 255 - index))
        for index in range(100)
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    assert visuals.validate_gif(path) == (100, (4, 3))
    expected = (
        "passive_wu_jumpgrad.gif",
        "optimization.gif",
        "tesseract_pipeline.png",
        "main_results.png",
        "gradient_story.png",
        "held_out.png",
    )
    assert tuple(path.name for path in visuals.EXPECTED_OUTPUTS) == expected
    actual = tuple(
        path.name
        for path in visuals.OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    )
    assert set(actual) == set(expected)
    assert visuals.validate_gif(visuals.MAIN_GIF_PATH)[0] == 100
    assert visuals.validate_gif(
        visuals.OPTIMIZATION_GIF_PATH,
        expected_frames=visuals.NUM_OPTIMIZATION_FRAMES,
    )[0] == 101
    assert set(visuals.METHOD_LABELS.values()) == {"Passive", "Wu2019", "JumpGrad"}
    assert "wu_vs_jumpgrad_control.gif" not in actual


def test_pipeline_only_cli_skips_scientific_replay(tmp_path: Path, monkeypatch):
    output = tmp_path / "tesseract_pipeline.png"
    monkeypatch.setattr(visuals, "OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(visuals, "PIPELINE_PATH", output)

    def fail_if_called():
        raise AssertionError("pipeline-only mode must not load scientific artifacts")

    monkeypatch.setattr(visuals, "load_frozen_sources", fail_if_called)
    visuals.main(["--pipeline-only"])

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 2000
        assert image.height >= 800


def test_readme_is_tesseract_first_without_numbered_captions():
    readme = (visuals.ROOT / "README.md").read_text()
    visual_paths = [
        f"outputs/jumpgrad_visuals/{path.name}" for path in visuals.EXPECTED_OUTPUTS
    ]
    required = (
        "Track 03 — Hybrid ML + mechanistic models",
        "jumpgrad_controller",
        "wu_v2_markov_fem",
        "Framework",
        "Derivative strategy",
        "Physics",
        "outputs/jumpgrad_visuals/tesseract_pipeline.png",
        "outputs/jumpgrad_visuals/passive_wu_jumpgrad.gif",
        "outputs/jumpgrad_visuals/optimization.gif",
        "outputs/jumpgrad_visuals/main_results.png",
        "outputs/jumpgrad_visuals/gradient_story.png",
        "outputs/jumpgrad_visuals/held_out.png",
        "10.21105/joss.08385",
        "10.1016/j.cpc.2023.108802",
        "10.1016/S0927-0507(06)13019-4",
        "10.1287/mnsc.45.11.1570",
        "10.1287/mnsc.38.6.884",
    )
    assert all(text in readme for text in required)
    assert "Figure 1" not in readme
    assert "Figure 2" not in readme
    assert "Figure 3" not in readme
    assert "Figure 4" not in readme
    assert "wu_vs_jumpgrad_control.gif" not in readme
    assert all(readme.count(path) == 1 for path in visual_paths)
    quick_commands = (
        "git clone https://github.com/dazhizhao/stochastic-stick-slip-tesseract.git",
        "cd stochastic-stick-slip-tesseract",
        "uv run python scripts/reproduce.py",
    )
    assert all(command in readme for command in quick_commands)
