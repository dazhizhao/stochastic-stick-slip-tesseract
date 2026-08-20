from types import SimpleNamespace

import numpy as np

from scripts.run_markov_jump_long_training import (
    HELD_OUT_STREAM,
    MONITOR_STREAM,
    _seed_batches,
)
from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.engineering_markov import (
    DAMPING,
    FULL_FIELD_MECHANICS_SIMULATOR,
    GATE_A_FORCING_SEEDS,
    evaluate_markov_bank,
    markov_uniform_bank,
)
from stochastic_stick_slip.engineering_showcase import forcing_batch
from tesseracts.markov_jump_fem import tesseract_api


def test_m4_training_seeds_split_into_four_fixed_batches() -> None:
    batches = list(_seed_batches(H4_TRAINING_SEEDS))

    assert len(batches) == 4
    assert all(batch.shape == (8,) for batch in batches)
    assert np.array_equal(np.concatenate(batches), H4_TRAINING_SEEDS)
    assert set(H4_TRAINING_SEEDS).isdisjoint(H4_TEST_SEEDS)


def test_markov_banks_support_reproducible_iterations_and_custom_seeds() -> None:
    default = markov_uniform_bank(4, stream_id=5)
    explicit = markov_uniform_bank(
        4,
        stream_id=5,
        forcing_seeds=GATE_A_FORCING_SEEDS,
        iteration=0,
    )
    first = markov_uniform_bank(
        4,
        stream_id=7,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=1,
    )
    repeated = markov_uniform_bank(
        4,
        stream_id=7,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=1,
    )
    second = markov_uniform_bank(
        4,
        stream_id=7,
        forcing_seeds=H4_TRAINING_SEEDS,
        iteration=2,
    )

    assert np.array_equal(default, explicit)
    assert first.shape == (32, 4, 801, 2)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_markov_vjp_reuses_explicit_tape_for_all_fd_sides(
    monkeypatch,
) -> None:
    uniforms = markov_uniform_bank(4, stream_id=7)
    captured = []

    def fake_evaluate(coefficients, forcing_seeds, markov_uniforms):
        del forcing_seeds
        captured.append(np.asarray(markov_uniforms).copy())
        losses = np.repeat(
            np.sum(np.asarray(coefficients), axis=1)[:, None], 4, axis=1
        )
        return SimpleNamespace(losses=losses)

    monkeypatch.setattr(tesseract_api, "_evaluate", fake_evaluate)
    inputs = SimpleNamespace(
        coeffs=np.zeros((8, 5), dtype=np.float64),
        forcing_seeds=GATE_A_FORCING_SEEDS,
        markov_uniforms=uniforms,
    )
    jacobian = tesseract_api._fd_coefficients(inputs)

    assert jacobian.shape == (8, 5)
    assert len(captured) == 10
    assert all(np.array_equal(tape, uniforms) for tape in captured)


def test_fixed_monitor_objective_is_reproducible() -> None:
    seeds = H4_TRAINING_SEEDS[:8]
    coefficients = np.zeros((8, 5), dtype=np.float64)
    uniforms = markov_uniform_bank(
        4,
        stream_id=MONITOR_STREAM,
        forcing_seeds=seeds,
        iteration=0,
    )
    forcing = forcing_batch(seeds)
    first = evaluate_markov_bank(coefficients, forcing, uniforms)
    second = evaluate_markov_bank(coefficients, forcing, uniforms)

    assert np.array_equal(np.asarray(first.losses), np.asarray(second.losses))


def test_full_field_replay_matches_markov_batch_forward() -> None:
    seeds = H4_TEST_SEEDS[:1]
    coefficients = np.zeros((1, 5), dtype=np.float64)
    uniforms = markov_uniform_bank(
        1,
        stream_id=HELD_OUT_STREAM,
        forcing_seeds=seeds,
        iteration=0,
    )
    forcing = forcing_batch(seeds)
    batch = evaluate_markov_bank(coefficients, forcing, uniforms)
    full = FULL_FIELD_MECHANICS_SIMULATOR(
        DAMPING,
        forcing,
        np.asarray(batch.preload[0]),
    )

    assert np.asarray(full[5]).shape == (1, 800, 320)
    assert np.allclose(
        np.asarray(full[0][0]),
        np.asarray(batch.displacement[0, 0]),
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.array_equal(
        np.asarray(full[2][0]), np.asarray(batch.slip[0, 0])
    )
