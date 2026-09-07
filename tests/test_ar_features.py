import numpy as np
import pytest

from dlquanttime.ar_features import (
    bandlimit_fid,
    burg_ar_coefficients,
    extract_ggg_ar_features,
)
from dlquanttime.simulate import SimulationConfig, simulate_fid


def test_burg_ar_coefficients_recovers_known_ar1_process():
    rng = np.random.default_rng(0)
    n = 3000
    x = np.zeros(n)
    true_coeff = 0.5
    for i in range(1, n):
        x[i] = true_coeff * x[i - 1] + rng.normal(0, 0.05)
    coeffs = burg_ar_coefficients(x.astype(complex), order=1)
    assert coeffs.shape == (1,)
    # x[n] ~= -a_1 * x[n-1], so a_1 should be close to -true_coeff.
    assert np.isclose(coeffs[0].real, -true_coeff, atol=0.05)


def test_burg_ar_coefficients_order_validation():
    x = np.zeros(4, dtype=complex)
    with pytest.raises(ValueError):
        burg_ar_coefficients(x, order=4)


def test_burg_ar_coefficients_zero_order():
    x = np.zeros(4, dtype=complex)
    coeffs = burg_ar_coefficients(x, order=0)
    assert coeffs.shape == (0,)


def test_bandlimit_fid_preserves_length():
    cfg = SimulationConfig(n_points=512)
    amps = {name: 0.5 for name in cfg.metabolites}
    lws = {name: 5.0 for name in cfg.metabolites}
    fid = simulate_fid(amps, lws, cfg)
    limited = bandlimit_fid(fid, cfg.dwell_time, cfg.field_strength_mhz)
    assert limited.shape == fid.shape
    assert np.iscomplexobj(limited)


def test_extract_ggg_ar_features_shape_and_finite():
    cfg = SimulationConfig(n_points=512)
    amps = {name: 0.5 for name in cfg.metabolites}
    lws = {name: 5.0 for name in cfg.metabolites}
    fid = simulate_fid(amps, lws, cfg, noise_std=0.01)
    feats = extract_ggg_ar_features(
        fid, cfg.dwell_time, cfg.field_strength_mhz, order=6
    )
    assert feats.shape == (12,)
    assert np.all(np.isfinite(feats))
