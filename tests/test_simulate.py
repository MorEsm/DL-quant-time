import numpy as np
import pytest

from dlquanttime.simulate import SimulationConfig, generate_dataset, simulate_fid


def test_simulate_fid_shape_and_dtype():
    cfg = SimulationConfig(n_points=256)
    amps = {name: 0.5 for name in cfg.metabolites}
    lws = {name: 5.0 for name in cfg.metabolites}
    fid = simulate_fid(amps, lws, cfg, noise_std=0.0)
    assert fid.shape == (256,)
    assert np.iscomplexobj(fid)


def test_simulate_fid_zero_amplitude_gives_zero_signal():
    cfg = SimulationConfig(n_points=128)
    amps = {name: 0.0 for name in cfg.metabolites}
    lws = {name: 5.0 for name in cfg.metabolites}
    fid = simulate_fid(amps, lws, cfg, noise_std=0.0)
    assert np.allclose(fid, 0.0)


def test_simulate_fid_decays_over_time():
    cfg = SimulationConfig(n_points=2048)
    amps = {name: 1.0 for name in cfg.metabolites}
    lws = {name: 10.0 for name in cfg.metabolites}
    fid = simulate_fid(amps, lws, cfg, noise_std=0.0)
    # Amplitude envelope should decrease (T2* decay) over the FID.
    assert np.abs(fid[0]) > np.abs(fid[-1])


def test_generate_dataset_shapes():
    cfg = SimulationConfig(n_points=64, seed=42)
    fids, amplitudes = generate_dataset(10, cfg)
    assert fids.shape == (10, 64)
    assert amplitudes.shape == (10, len(cfg.metabolites))
    assert np.all(amplitudes >= cfg.amplitude_range[0] - 1e-9)
    assert np.all(amplitudes <= cfg.amplitude_range[1] + 1e-9)


def test_generate_dataset_reproducible_with_seed():
    cfg1 = SimulationConfig(n_points=32, seed=7)
    cfg2 = SimulationConfig(n_points=32, seed=7)
    fids1, amps1 = generate_dataset(5, cfg1)
    fids2, amps2 = generate_dataset(5, cfg2)
    assert np.allclose(fids1, fids2)
    assert np.allclose(amps1, amps2)
