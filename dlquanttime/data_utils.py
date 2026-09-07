"""Helpers to convert simulated complex FIDs into model-ready tensors."""

from typing import Sequence, Tuple

import numpy as np

from .ar_features import extract_ggg_ar_features
from .constants import GGG_METABOLITES


def fid_to_channels(fid: np.ndarray) -> np.ndarray:
    """Stack the real/imaginary parts of a (batch of) complex FID(s).

    Parameters
    ----------
    fid:
        Complex array of shape ``(n_points,)`` or ``(n_samples, n_points)``.

    Returns
    -------
    np.ndarray
        Real array of shape ``(2, n_points)`` or ``(n_samples, 2, n_points)``.
    """
    fid = np.asarray(fid)
    real = fid.real
    imag = fid.imag
    return np.stack([real, imag], axis=-2)


def ggg_indices(metabolites: Sequence[str]) -> Tuple[int, ...]:
    """Indices of the GGG metabolites (Glu, Gln, GABA) within ``metabolites``."""
    return tuple(metabolites.index(name) for name in GGG_METABOLITES)


def build_ar_feature_batch(
    fids: np.ndarray,
    dwell_time: float,
    field_strength_mhz: float,
    order: int = 8,
) -> np.ndarray:
    """Compute GGG-region AR features for a batch of complex FIDs.

    Parameters
    ----------
    fids:
        Complex array of shape ``(n_samples, n_points)``.
    dwell_time, field_strength_mhz, order:
        See :func:`dlquanttime.ar_features.extract_ggg_ar_features`.

    Returns
    -------
    np.ndarray
        Real array of shape ``(n_samples, 2 * order)``.
    """
    fids = np.asarray(fids)
    features = [
        extract_ggg_ar_features(fid, dwell_time, field_strength_mhz, order=order)
        for fid in fids
    ]
    return np.stack(features, axis=0)
