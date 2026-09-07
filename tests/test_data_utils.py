import numpy as np

from dlquanttime.data_utils import build_ar_feature_batch, fid_to_channels, ggg_indices
from dlquanttime.simulate import SimulationConfig, generate_dataset


def test_fid_to_channels_single_fid():
    fid = np.array([1 + 2j, 3 - 4j], dtype=complex)
    channels = fid_to_channels(fid)
    assert channels.shape == (2, 2)
    assert np.allclose(channels[0], [1, 3])
    assert np.allclose(channels[1], [2, -4])


def test_fid_to_channels_batch():
    cfg = SimulationConfig(n_points=32, seed=0)
    fids, _ = generate_dataset(4, cfg)
    channels = fid_to_channels(fids)
    assert channels.shape == (4, 2, 32)


def test_ggg_indices():
    metabolites = ("NAA", "Cr", "PCr", "GPC", "Glu", "Gln", "GABA", "mI")
    idx = ggg_indices(metabolites)
    assert idx == (4, 5, 6)


def test_build_ar_feature_batch_shape():
    cfg = SimulationConfig(n_points=256, seed=0)
    fids, _ = generate_dataset(3, cfg)
    features = build_ar_feature_batch(
        fids, cfg.dwell_time, cfg.field_strength_mhz, order=4
    )
    assert features.shape == (3, 8)
    assert np.all(np.isfinite(features))
