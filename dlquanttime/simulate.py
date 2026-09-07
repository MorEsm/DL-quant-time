"""Synthetic time-domain (FID) MR spectroscopy data generator.

This module produces synthetic complex free induction decays (FIDs) that
simulate physiologically plausible metabolite concentrations, linewidths
and noise levels, following the description in the accompanying
publication. Each metabolite is modelled as a single decaying complex
exponential (Lorentzian lineshape) centered at its dominant chemical
shift; the full FID is the sum of all metabolite components plus complex
Gaussian noise.

The generator is intentionally simplified (it does not model full
metabolite basis spectra with J-coupling) but is sufficient to produce a
labelled dataset (FID -> ground-truth amplitudes) for training and
validating the quantification networks in :mod:`dlquanttime.models`.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import METABOLITES


@dataclass
class SimulationConfig:
    """Configuration for the synthetic FID generator."""

    n_points: int = 2048
    dwell_time: float = 5e-4  # seconds
    field_strength_mhz: float = 127.74  # ¹H Larmor frequency (3T), MHz
    amplitude_range: Tuple[float, float] = (0.1, 1.0)
    linewidth_range_hz: Tuple[float, float] = (2.0, 12.0)
    noise_std_range: Tuple[float, float] = (0.0, 0.05)
    metabolites: Tuple[str, ...] = tuple(METABOLITES.keys())
    seed: Optional[int] = None


def _time_axis(cfg: SimulationConfig) -> np.ndarray:
    return np.arange(cfg.n_points) * cfg.dwell_time


def simulate_fid(
    amplitudes: Dict[str, float],
    linewidths_hz: Dict[str, float],
    cfg: SimulationConfig,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a single complex FID given metabolite amplitudes/linewidths.

    Parameters
    ----------
    amplitudes:
        Mapping of metabolite name -> (non-negative) amplitude.
    linewidths_hz:
        Mapping of metabolite name -> Lorentzian linewidth (Hz).
    cfg:
        Simulation configuration (defines sampling, ppm shifts, etc).
    noise_std:
        Standard deviation of the (real and imaginary) additive Gaussian
        noise, expressed relative to the same units as ``amplitudes``.
    rng:
        Optional random generator used for the additive noise.

    Returns
    -------
    np.ndarray
        Complex-valued FID of shape ``(cfg.n_points,)``.
    """
    rng = rng or np.random.default_rng()
    t = _time_axis(cfg)
    fid = np.zeros(cfg.n_points, dtype=np.complex128)
    for name in cfg.metabolites:
        amp = amplitudes.get(name, 0.0)
        if amp == 0.0:
            continue
        ppm = METABOLITES[name]
        # Convert chemical shift (ppm, relative to an arbitrary 0 ppm
        # reference) to an angular frequency in rad/s.
        freq_hz = ppm * cfg.field_strength_mhz
        omega = 2.0 * np.pi * freq_hz
        lw = max(linewidths_hz.get(name, 5.0), 1e-6)
        decay_rate = np.pi * lw  # Lorentzian T2* decay -> exp(-t/T2*)
        fid += amp * np.exp(1j * omega * t) * np.exp(-decay_rate * t)
    if noise_std > 0:
        noise = rng.normal(0, noise_std, cfg.n_points) + 1j * rng.normal(
            0, noise_std, cfg.n_points
        )
        fid = fid + noise
    return fid


def sample_parameters(
    cfg: SimulationConfig, rng: np.random.Generator
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Draw a random, physiologically plausible set of simulation parameters."""
    amplitudes = {
        name: rng.uniform(*cfg.amplitude_range) for name in cfg.metabolites
    }
    linewidths = {
        name: rng.uniform(*cfg.linewidth_range_hz) for name in cfg.metabolites
    }
    noise_std = rng.uniform(*cfg.noise_std_range)
    return amplitudes, linewidths, noise_std


def generate_dataset(
    n_samples: int, cfg: Optional[SimulationConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic labelled dataset of FIDs and metabolite amplitudes.

    Parameters
    ----------
    n_samples:
        Number of FIDs to simulate.
    cfg:
        Simulation configuration. Defaults to :class:`SimulationConfig`.

    Returns
    -------
    fids: np.ndarray
        Complex array of shape ``(n_samples, cfg.n_points)``.
    amplitudes: np.ndarray
        Real array of shape ``(n_samples, len(cfg.metabolites))`` holding
        the ground-truth amplitude for each metabolite (columns ordered
        according to ``cfg.metabolites``).
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    fids = np.zeros((n_samples, cfg.n_points), dtype=np.complex128)
    amplitudes = np.zeros((n_samples, len(cfg.metabolites)), dtype=np.float64)
    for i in range(n_samples):
        amps, lws, noise_std = sample_parameters(cfg, rng)
        fids[i] = simulate_fid(amps, lws, cfg, noise_std=noise_std, rng=rng)
        amplitudes[i] = [amps[name] for name in cfg.metabolites]
    return fids, amplitudes
