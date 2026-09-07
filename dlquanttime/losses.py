"""Metabolite-specific, inverse-amplitude-weighted loss function.

Metabolites present at low physiological concentration (e.g. GABA, Gln)
are systematically harder to quantify than high-concentration metabolites
(e.g. Cr, NAA). A plain mean-squared-error loss is dominated by the
high-amplitude metabolites, so we instead weight each metabolite's squared
error by the inverse of its (batch-average) ground-truth amplitude. This
gives every metabolite channel comparable influence on the training
gradient regardless of its natural concentration scale.

Both a pure NumPy reference implementation (used for unit testing without
requiring a deep learning framework) and a ``torch.nn.Module`` wrapper
(used during training) are provided.
"""

from typing import Optional

import numpy as np

try:  # pragma: no cover - exercised only when torch is installed.
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore


def inverse_amplitude_weights(
    amplitudes: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Compute per-metabolite inverse-amplitude weights.

    Parameters
    ----------
    amplitudes:
        Array of shape ``(n_samples, n_metabolites)`` of ground-truth
        amplitudes.
    eps:
        Small constant added to the mean amplitude to avoid division by
        (near) zero.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_metabolites,)`` with weights normalized so
        that they sum to ``n_metabolites`` (i.e. an unweighted average of
        the weights leaves the overall loss scale unchanged).
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    mean_amp = amplitudes.mean(axis=0) + eps
    weights = 1.0 / mean_amp
    weights = weights * (weights.shape[0] / weights.sum())
    return weights


def weighted_mse_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Reference NumPy implementation of the weighted MSE loss.

    Parameters
    ----------
    y_true, y_pred:
        Arrays of shape ``(n_samples, n_metabolites)``.
    weights:
        Optional pre-computed per-metabolite weights (shape
        ``(n_metabolites,)``); if omitted, computed from ``y_true`` via
        :func:`inverse_amplitude_weights`.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if weights is None:
        weights = inverse_amplitude_weights(y_true)
    sq_err = (y_true - y_pred) ** 2
    return float(np.mean(sq_err * weights))


if _HAS_TORCH:

    class InverseAmplitudeWeightedMSELoss(nn.Module):
        """Torch loss module implementing the inverse-amplitude weighting.

        Weights are (re-)estimated from each batch's ground-truth
        amplitudes, so no dataset-wide statistics need to be tracked.
        """

        def __init__(self, eps: float = 1e-3):
            super().__init__()
            self.eps = eps

        def forward(self, y_pred: "torch.Tensor", y_true: "torch.Tensor") -> "torch.Tensor":
            mean_amp = y_true.mean(dim=0) + self.eps
            weights = 1.0 / mean_amp
            weights = weights * (weights.shape[0] / weights.sum())
            sq_err = (y_true - y_pred) ** 2
            return torch.mean(sq_err * weights)
