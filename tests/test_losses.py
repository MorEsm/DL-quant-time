import numpy as np
import pytest

from dlquanttime.losses import inverse_amplitude_weights, weighted_mse_numpy


def test_inverse_amplitude_weights_low_amplitude_gets_higher_weight():
    amplitudes = np.array(
        [
            [10.0, 1.0],
            [12.0, 1.2],
            [8.0, 0.8],
        ]
    )
    weights = inverse_amplitude_weights(amplitudes)
    assert weights.shape == (2,)
    # Metabolite 1 (low mean amplitude) should receive a larger weight.
    assert weights[1] > weights[0]


def test_inverse_amplitude_weights_normalized_to_n_metabolites():
    amplitudes = np.array([[1.0, 5.0, 10.0], [2.0, 4.0, 9.0]])
    weights = inverse_amplitude_weights(amplitudes)
    assert weights.sum() == pytest.approx(amplitudes.shape[1], rel=1e-6)


def test_weighted_mse_numpy_zero_for_perfect_predictions():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = y_true.copy()
    assert weighted_mse_numpy(y_true, y_pred) == 0.0


def test_weighted_mse_numpy_penalizes_low_amplitude_errors_more():
    y_true = np.array([[10.0, 1.0], [10.0, 1.0]])
    # Equal absolute error on both channels.
    y_pred_high_error = np.array([[9.0, 1.0], [9.0, 1.0]])
    y_pred_low_error = np.array([[10.0, 0.0], [10.0, 0.0]])
    loss_high_amp_error = weighted_mse_numpy(y_true, y_pred_high_error)
    loss_low_amp_error = weighted_mse_numpy(y_true, y_pred_low_error)
    # An error of the same magnitude on the low-amplitude metabolite
    # should be penalized more heavily.
    assert loss_low_amp_error > loss_high_amp_error
