from types import SimpleNamespace

import numpy as np
import torch

from scripts import run_markov_jump_ablation as ablation
from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.engineering_markov import markov_uniform_bank


def _fake_result(coefficients):
    losses = np.repeat(
        np.sum(np.asarray(coefficients), axis=1)[:, None], 4, axis=1
    )
    return SimpleNamespace(losses=losses)


def test_crn_and_independent_fd_receive_the_locked_explicit_tapes(
    monkeypatch,
) -> None:
    seeds = ablation.GATE_A_FORCING_SEEDS
    plus_uniforms = markov_uniform_bank(
        4, stream_id=ablation.COUPLING_PLUS_STREAM, forcing_seeds=seeds
    )
    minus_uniforms = markov_uniform_bank(
        4, stream_id=ablation.COUPLING_MINUS_STREAM, forcing_seeds=seeds
    )
    captured = []

    def fake_evaluate(coefficients, forcing, uniforms):
        del forcing
        captured.append(np.asarray(uniforms).copy())
        return _fake_result(coefficients)

    monkeypatch.setattr(ablation, "evaluate_markov_bank", fake_evaluate)
    coefficients = np.zeros((8, 5), dtype=np.float64)
    ablation._per_seed_coordinate_fd(
        coefficients, seeds, plus_uniforms, plus_uniforms
    )
    assert len(captured) == 10
    assert all(np.array_equal(tape, plus_uniforms) for tape in captured)

    captured.clear()
    ablation._per_seed_coordinate_fd(
        coefficients, seeds, plus_uniforms, minus_uniforms
    )
    assert len(captured) == 10
    assert all(
        np.array_equal(tape, plus_uniforms)
        for tape in captured[0::2]
    )
    assert all(
        np.array_equal(tape, minus_uniforms)
        for tape in captured[1::2]
    )
    assert not np.array_equal(plus_uniforms, minus_uniforms)


def test_shared_zero_matches_neutral_and_has_finite_gradient() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    uniforms = markov_uniform_bank(
        4,
        stream_id=ablation.TRAINING_STREAM,
        forcing_seeds=seeds,
        iteration=3,
    )
    zero = np.zeros(5, dtype=np.float64)
    neutral = ablation._evaluate_coefficients(
        np.zeros((8, 5), dtype=np.float64), seeds, uniforms
    )
    shared = ablation._shared_bank(zero, seeds, uniforms)
    gradient = ablation._shared_coordinate_gradient(
        zero, seeds, uniforms, uniforms
    )

    assert np.array_equal(neutral.seed_losses, shared.seed_losses)
    assert np.all(np.isfinite(shared.seed_losses))
    assert gradient.shape == (5,)
    assert np.all(np.isfinite(gradient))


def test_shared_backward_and_held_out_bank_are_explicit() -> None:
    physics = ablation.Tesseract.from_tesseract_api(ablation.PHYSICS_API)
    seeds = H4_TRAINING_SEEDS[:8]
    uniforms = markov_uniform_bank(
        4,
        stream_id=ablation.TRAINING_STREAM,
        forcing_seeds=seeds,
        iteration=4,
    )
    shared = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    loss = ablation._shared_training_loss(physics, shared, seeds, uniforms)
    loss.backward()

    assert np.isfinite(float(loss.detach()))
    assert shared.grad.shape == (5,)
    assert np.all(np.isfinite(shared.grad.detach().numpy()))

    held_out = markov_uniform_bank(
        4,
        stream_id=ablation.HELD_OUT_STREAM,
        forcing_seeds=H4_TEST_SEEDS[:8],
        iteration=0,
    )
    neutral = ablation._shared_bank(
        np.zeros(5), H4_TEST_SEEDS[:8], held_out
    )
    repeated = ablation._shared_bank(
        np.zeros(5), H4_TEST_SEEDS[:8], held_out
    )
    assert np.array_equal(neutral.losses, repeated.losses)


def test_all_controller_evaluations_receive_the_same_held_out_bank(
    monkeypatch,
) -> None:
    seeds = H4_TEST_SEEDS[:8]
    held_out = markov_uniform_bank(
        4,
        stream_id=ablation.HELD_OUT_STREAM,
        forcing_seeds=seeds,
        iteration=0,
    )
    captured = []

    def fake_shared(shared, received_seeds, uniforms):
        del shared, received_seeds
        captured.append(np.asarray(uniforms).copy())
        return len(captured)

    def fake_mlp(controller, theta, received_seeds, uniforms):
        del controller, theta, received_seeds
        captured.append(np.asarray(uniforms).copy())
        return len(captured)

    monkeypatch.setattr(ablation, "_shared_bank", fake_shared)
    monkeypatch.setattr(ablation, "_evaluate_bank", fake_mlp)
    results = ablation._evaluate_controller_ablation(
        object(), np.zeros(469), np.zeros(5), seeds, held_out
    )

    assert results == (1, 2, 3)
    assert len(captured) == 3
    assert all(np.array_equal(tape, held_out) for tape in captured)


def test_m4_history_matches_frozen_ablation_contract() -> None:
    history = ablation._read_m4_history()

    assert history["theta_history"].shape == (201, 469)
    assert history["train_objective"].shape == (201,)
    assert history["gradient_norm"].shape == (200,)
    assert history["monitor_iterations"].shape == (21,)
    assert history["monitor_objective"].shape == (21,)
