"""Autoregressive (AR) feature extraction from band-limited FID signals.

As described in the accompanying publication, the MC-DenseNet-GGG model is
additionally conditioned on autoregressive coefficients derived from the
FID after band-limiting to the glutamate/glutamine/GABA ("GGG") spectral
region (2.1-2.6 ppm). This module implements:

* :func:`bandlimit_fid` -- isolate a ppm range of a time-domain FID via an
  FFT domain band-pass filter.
* :func:`burg_ar_coefficients` -- estimate AR model coefficients from a
  (band-limited) complex signal using the Burg method.
* :func:`extract_ggg_ar_features` -- convenience wrapper combining the two
  steps above for the GGG region.
"""

from typing import Tuple

import numpy as np

from .constants import GGG_PPM_RANGE


def _frequency_axis_hz(n_points: int, dwell_time: float) -> np.ndarray:
    """FFT frequency axis (Hz) matching ``np.fft.fft`` bin ordering."""
    return np.fft.fftfreq(n_points, d=dwell_time)


def bandlimit_fid(
    fid: np.ndarray,
    dwell_time: float,
    field_strength_mhz: float,
    ppm_range: Tuple[float, float] = GGG_PPM_RANGE,
) -> np.ndarray:
    """Band-limit a complex FID to a given ppm range.

    The FID is Fourier transformed, all frequency bins outside the
    requested ppm range are zeroed, and the signal is transformed back to
    the time domain.

    Parameters
    ----------
    fid:
        Complex time-domain signal, shape ``(n_points,)`` (or
        ``(n_samples, n_points)`` for a batch).
    dwell_time:
        Sample spacing in seconds.
    field_strength_mhz:
        ¹H Larmor frequency (MHz), used to convert ppm to Hz.
    ppm_range:
        ``(low, high)`` ppm bounds of the pass-band.

    Returns
    -------
    np.ndarray
        Band-limited complex time-domain signal, same shape as ``fid``.
    """
    fid = np.asarray(fid)
    n_points = fid.shape[-1]
    freq_hz = _frequency_axis_hz(n_points, dwell_time)
    ppm = freq_hz / field_strength_mhz
    low, high = min(ppm_range), max(ppm_range)
    mask = (ppm >= low) & (ppm <= high)
    spectrum = np.fft.fft(fid, axis=-1)
    spectrum = spectrum * mask
    return np.fft.ifft(spectrum, axis=-1)


def burg_ar_coefficients(signal: np.ndarray, order: int) -> np.ndarray:
    """Estimate AR coefficients of a complex signal using the Burg method.

    Parameters
    ----------
    signal:
        1-D complex (or real) signal.
    order:
        AR model order (number of coefficients to estimate).

    Returns
    -------
    np.ndarray
        Complex array of ``order`` AR coefficients ``a_1 .. a_order`` such
        that ``x[n] ~= -sum_k a_k * x[n-k]``.
    """
    x = np.asarray(signal).astype(np.complex128)
    n = x.shape[0]
    if order <= 0:
        return np.zeros(0, dtype=np.complex128)
    if order >= n:
        raise ValueError("AR order must be smaller than the signal length")

    # Standard complex Burg recursion (e.g. Marple, "Digital Spectral
    # Analysis"). ``a`` holds the coefficients of the order-0 model
    # (a[0] == 1); after the loop ``a[1:]`` are the desired AR
    # coefficients such that ``x[n] + sum_k a[k] * x[n-k] ~= e[n]``.
    ef = x.copy()
    eb = x.copy()
    a = np.array([1.0 + 0.0j])

    for k in range(order):
        efp = ef[k + 1 :]
        ebp = eb[k:-1]
        num = -2.0 * np.dot(np.conj(ebp), efp)
        den = np.dot(np.conj(efp), efp) + np.dot(np.conj(ebp), ebp)
        reflection = num / den if np.abs(den) > 1e-12 else 0.0 + 0.0j

        a_padded = np.concatenate((a, [0.0 + 0.0j]))
        a_reversed_conj = np.concatenate(([0.0 + 0.0j], np.conj(a[::-1])))
        a = a_padded + reflection * a_reversed_conj

        ef = efp + reflection * ebp
        eb = ebp + np.conj(reflection) * efp

    return a[1:]


def extract_ggg_ar_features(
    fid: np.ndarray,
    dwell_time: float,
    field_strength_mhz: float,
    order: int = 8,
    ppm_range: Tuple[float, float] = GGG_PPM_RANGE,
) -> np.ndarray:
    """Extract real-valued AR features from the GGG-band-limited FID.

    The complex AR coefficients are returned as a real feature vector of
    length ``2 * order`` (concatenated real and imaginary parts), suitable
    for direct use as an auxiliary neural network input.
    """
    limited = bandlimit_fid(fid, dwell_time, field_strength_mhz, ppm_range)
    coeffs = burg_ar_coefficients(limited, order)
    return np.concatenate([coeffs.real, coeffs.imag]).astype(np.float64)
