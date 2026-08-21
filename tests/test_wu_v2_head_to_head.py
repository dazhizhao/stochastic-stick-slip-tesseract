import numpy as np

from scripts.run_wu_v2_head_to_head import (
    CONFIRMATION_STREAMS,
    REALIZATIONS_PER_BANK,
    W1_LOCAL_FRF_RATIOS,
    expected_high_probability,
    stochastic_frf_summary,
)
from stochastic_stick_slip.wu_v2_markov import (
    markov_uniform_bank,
    two_omega_signal,
)


def test_confirmation_streams_are_new_independent_and_reproducible() -> None:
    assert CONFIRMATION_STREAMS == (5, 6, 7, 8)
    confirmation = [
        markov_uniform_bank(1, stream_id=stream, iteration=0)
        for stream in CONFIRMATION_STREAMS
    ]
    repeated = markov_uniform_bank(1, stream_id=5, iteration=0)
    training = markov_uniform_bank(1, stream_id=2, iteration=99)
    old_evaluation = markov_uniform_bank(1, stream_id=3, iteration=0)
    old_confirmation = markov_uniform_bank(1, stream_id=4, iteration=0)
    assert np.array_equal(confirmation[0], repeated)
    assert not np.array_equal(confirmation[0], training)
    assert not np.array_equal(confirmation[0], old_evaluation)
    assert not np.array_equal(confirmation[0], old_confirmation)
    for left, right in zip(confirmation[:-1], confirmation[1:], strict=True):
        assert not np.array_equal(left, right)


def test_stochastic_signal_tracks_current_frequency_without_changing_q() -> None:
    q = np.asarray([-3.5, -0.5])
    frozen = q.copy()
    time = np.asarray([0.13, 0.29])
    first = np.asarray(two_omega_signal(q, time, 3.0))
    second = np.asarray(two_omega_signal(q, time, 4.0))
    assert not np.allclose(first, second)
    assert np.array_equal(q, frozen)


def test_head_to_head_grid_is_the_registered_21_point_w1_grid() -> None:
    expected = np.linspace(0.90, 1.10, 21)
    assert np.array_equal(W1_LOCAL_FRF_RATIOS, expected)


def test_aggregate_and_bank_peaks_use_frequency_of_the_mean() -> None:
    ratios = np.linspace(0.90, 1.10, 21)
    values = np.zeros((21, 4, REALIZATIONS_PER_BANK), dtype=np.float64)
    values[7, :, :] = 2.0
    values[9, 0, :] = 3.0
    values[10, 1, :] = 4.0
    values[11, 2, :] = 5.0
    values[12, 3, :] = 6.0
    summary, flattened = stochastic_frf_summary(ratios, values)
    assert flattened.shape == (21, 256)
    assert summary["aggregate_peak"]["peak_index"] == 7
    assert [entry["peak_index"] for entry in summary["bank_peaks"]] == [9, 10, 11, 12]


def test_expected_high_probability_uses_two_state_recurrence() -> None:
    p_lh = np.asarray([0.25, 0.25, 0.25])
    p_hl = np.asarray([0.10, 0.10, 0.10])
    expected = []
    high = 0.5
    for _ in range(3):
        high = high * 0.90 + (1.0 - high) * 0.25
        expected.append(high)
    assert np.allclose(expected_high_probability(p_lh, p_hl), expected)
