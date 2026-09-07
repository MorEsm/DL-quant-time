import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlquanttime.data_utils import fid_to_channels, ggg_indices  # noqa: E402
from dlquanttime.losses import InverseAmplitudeWeightedMSELoss  # noqa: E402
from dlquanttime.models import (  # noqa: E402
    ConvAutoencoder,
    DCDenseNet,
    MCDenseNetGGG,
    SequentialCNN,
)
from dlquanttime.simulate import SimulationConfig, generate_dataset  # noqa: E402

METABOLITES = ("NAA", "Cr", "PCr", "GPC", "Glu", "Gln", "GABA", "mI")


def _make_batch(n=4, n_points=256):
    cfg = SimulationConfig(n_points=n_points, metabolites=METABOLITES, seed=0)
    fids, amps = generate_dataset(n, cfg)
    channels = fid_to_channels(fids)
    return torch.tensor(channels, dtype=torch.float32), torch.tensor(
        amps, dtype=torch.float32
    ), cfg


def test_dc_densenet_output_shape():
    x, y, cfg = _make_batch()
    model = DCDenseNet(len(METABOLITES), n_points=cfg.n_points)
    out = model(x)
    assert out.shape == y.shape


def test_mc_densenet_ggg_output_shape_and_constrained_pathway():
    x, y, cfg = _make_batch()
    idx = ggg_indices(METABOLITES)
    n_ar = 16
    model = MCDenseNetGGG(len(METABOLITES), idx, n_ar, n_points=cfg.n_points)
    ar = torch.zeros(x.shape[0], n_ar)
    out = model(x, ar)
    assert out.shape == y.shape

    # Changing the AR features should only change the GGG output channels.
    ar_nonzero = torch.randn(x.shape[0], n_ar)
    out2 = model(x, ar_nonzero)
    diff = (out2 - out).abs()
    non_ggg = [i for i in range(len(METABOLITES)) if i not in idx]
    assert torch.all(diff[:, non_ggg] < 1e-5)
    assert torch.any(diff[:, list(idx)] > 1e-5)


def test_conv_autoencoder_output_shapes():
    x, y, _ = _make_batch()
    model = ConvAutoencoder(len(METABOLITES), n_points=x.shape[-1])
    amps = model(x)
    assert amps.shape == y.shape
    amps2, recon = model(x, return_reconstruction=True)
    assert recon.shape == x.shape


def test_sequential_cnn_output_shape():
    x, y, _ = _make_batch()
    model = SequentialCNN(len(METABOLITES), n_points=x.shape[-1])
    out = model(x)
    assert out.shape == y.shape


def test_inverse_amplitude_weighted_loss_is_scalar_and_backprops():
    x, y, cfg = _make_batch()
    model = DCDenseNet(len(METABOLITES), n_points=cfg.n_points)
    criterion = InverseAmplitudeWeightedMSELoss()
    preds = model(x)
    loss = criterion(preds, y)
    assert loss.dim() == 0
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)
