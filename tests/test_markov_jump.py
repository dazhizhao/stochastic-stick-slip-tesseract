import jax.numpy as jnp
import numpy as np

from scripts.run_markov_jump_gate_a import _centered_fd
from stochastic_stick_slip.engineering_markov import (
    DAMPING,
    GATE_A_FORCING_SEEDS,
    MECHANICS_SIMULATOR,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    direct_ad_objective_and_gradient,
    gate_a_forcing,
    markov_uniform_bank,
)
from stochastic_stick_slip.engineering_showcase import _SIMULATE_BATCH
from stochastic_stick_slip.markov_jump import generate_markov_preload_history
from stochastic_stick_slip.model import NUM_FOURIER_COEFFICIENTS, NUM_STEPS


def test_markov_preloads_are_hard_and_contact_tapes_are_independent() -> None:
    basis = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    coefficients = jnp.zeros((1, 2), dtype=jnp.float64)
    uniforms = jnp.array(
        [[[[0.25, 0.75], [0.90, 0.90], [0.90, 0.90]]]],
        dtype=jnp.float64,
    )
    modes, preload, transition_counts, _ = generate_markov_preload_history(
        coefficients,
        basis,
        uniforms,
        time_step=1.0,
        lambda_0=1.0,
        beta=1.0,
        preload_low=PRELOAD_LOW,
        preload_high=PRELOAD_HIGH,
    )

    assert np.array_equal(np.asarray(modes[0, 0, :, 0]), [True, True])
    assert np.array_equal(np.asarray(modes[0, 0, :, 1]), [False, False])
    assert set(np.unique(np.asarray(preload))) == {PRELOAD_LOW, PRELOAD_HIGH}
    assert np.array_equal(np.asarray(transition_counts), np.zeros((1, 1, 2)))


def test_fixed_preload_core_matches_legacy_and_supports_mixed_contacts() -> None:
    forcing = gate_a_forcing()
    zero_coefficients = jnp.zeros(
        (len(GATE_A_FORCING_SEEDS), NUM_FOURIER_COEFFICIENTS),
        dtype=jnp.float64,
    )
    for contact_preloads in (
        (PRELOAD_LOW, PRELOAD_LOW),
        (PRELOAD_LOW, PRELOAD_HIGH),
        (PRELOAD_HIGH, PRELOAD_LOW),
        (PRELOAD_HIGH, PRELOAD_HIGH),
    ):
        preload = jnp.broadcast_to(
            jnp.asarray(contact_preloads),
            (len(GATE_A_FORCING_SEEDS), NUM_STEPS, 2),
        )
        outputs = MECHANICS_SIMULATOR(DAMPING, forcing, preload)
        assert np.all(np.isfinite(np.asarray(outputs[0])))
        assert np.all(np.isfinite(np.asarray(outputs[1])))
        if contact_preloads[0] == contact_preloads[1]:
            legacy = _SIMULATE_BATCH(
                jnp.asarray([DAMPING, contact_preloads[0]]),
                zero_coefficients,
                forcing,
            )
            for candidate, reference in zip(outputs, legacy, strict=True):
                assert np.allclose(
                    np.asarray(candidate),
                    np.asarray(reference),
                    rtol=1e-12,
                    atol=1e-14,
                )


def test_hard_markov_forward_has_zero_raw_direct_ad() -> None:
    forcing = gate_a_forcing()
    uniforms = markov_uniform_bank(1)
    objective, gradient = direct_ad_objective_and_gradient(
        np.zeros(NUM_FOURIER_COEFFICIENTS), forcing, uniforms
    )

    assert np.isfinite(objective)
    assert gradient.shape == (NUM_FOURIER_COEFFICIENTS,)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) <= 1e-12


def test_independent_markov_banks_are_reproducible_and_nested() -> None:
    bank_a_r4 = markov_uniform_bank(4, stream_id=5)
    bank_a_repeat = markov_uniform_bank(4, stream_id=5)
    bank_a_r8 = markov_uniform_bank(8, stream_id=5)
    bank_b_r4 = markov_uniform_bank(4, stream_id=6)
    bank_b_r8 = markov_uniform_bank(8, stream_id=6)

    assert np.array_equal(bank_a_r4, bank_a_repeat)
    assert not np.array_equal(bank_a_r4, bank_b_r4)
    assert np.array_equal(bank_a_r8[:, :4], bank_a_r4)
    assert np.array_equal(bank_b_r8[:, :4], bank_b_r4)


def test_independent_bank_crn_fd_gradients_are_finite_and_nonzero() -> None:
    forcing = gate_a_forcing()
    shared_coefficients = np.zeros(NUM_FOURIER_COEFFICIENTS, dtype=np.float64)
    for stream_id in (5, 6):
        result = _centered_fd(
            shared_coefficients,
            forcing,
            markov_uniform_bank(4, stream_id=stream_id),
        )
        gradient = np.asarray(result["gradient"])
        assert result["finite"]
        assert not result["numerical_zero"]
        assert np.all(np.isfinite(gradient))
