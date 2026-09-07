"""1D DenseNet architectures for direct time-domain MRS quantification.

Implements a modified one-dimensional DenseNet (dense connectivity as in
Huang et al., 2017, adapted to 1-D convolutions over the real/imaginary
channels of a time-domain FID) in two configurations:

* :class:`DCDenseNet` -- the dual-input (real, imaginary) configuration
  that uses only the complex time-domain FID.
* :class:`MCDenseNetGGG` -- the multi-input configuration that
  additionally injects autoregressive (AR) coefficients derived from the
  FID band-limited to the glutamate/glutamine/GABA ("GGG") spectral
  region. The AR pathway is structurally constrained (via a small
  auxiliary head added only to the GGG output channels) so that it can
  only influence the Glu, Gln and GABA predictions.
"""

from typing import Sequence

import torch
import torch.nn as nn


class _DenseLayer1D(nn.Module):
    """A single densely-connected 1-D conv layer (BN-ReLU-Conv)."""

    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int = 5):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class DenseBlock1D(nn.Module):
    """A stack of densely-connected 1-D conv layers.

    Each layer receives the concatenation of the outputs of all preceding
    layers (and the block input) as its input, giving the block
    ``in_channels + num_layers * growth_rate`` output channels.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 8,
        num_layers: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer1D(channels, growth_rate, kernel_size))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _TransitionDown1D(nn.Module):
    """1x1 conv + average pooling to compress and downsample a dense block."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.bn(x))))


class _DenseNetEncoder1D(nn.Module):
    """Shared densely-connected 1-D encoder used by both model variants."""

    def __init__(
        self,
        in_channels: int = 2,
        growth_rate: int = 8,
        block_layers: Sequence[int] = (4, 4, 4),
        compression: float = 0.5,
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, 2 * growth_rate, kernel_size=7, padding=3)

        blocks = []
        transitions = []
        channels = 2 * growth_rate
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock1D(channels, growth_rate, num_layers)
            blocks.append(block)
            channels = block.out_channels
            if i != len(block_layers) - 1:
                out_channels = max(int(channels * compression), growth_rate)
                transitions.append(_TransitionDown1D(channels, out_channels))
                channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.final_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_features = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.relu(self.final_bn(x))
        x = self.pool(x).squeeze(-1)
        return x


class DCDenseNet(nn.Module):
    """Dual-input (real + imaginary channel) DenseNet metabolite quantifier.

    Parameters
    ----------
    n_metabolites:
        Number of metabolite output channels.
    n_points:
        Length of the input FID (used only for documentation/validation;
        the network itself is fully convolutional).
    growth_rate, block_layers:
        DenseNet hyper-parameters, see :class:`_DenseNetEncoder1D`.
    """

    def __init__(
        self,
        n_metabolites: int,
        n_points: int = 2048,
        growth_rate: int = 8,
        block_layers: Sequence[int] = (4, 4, 4),
    ):
        super().__init__()
        self.n_points = n_points
        self.encoder = _DenseNetEncoder1D(
            in_channels=2, growth_rate=growth_rate, block_layers=block_layers
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_metabolites),
        )

    def forward(self, fid: torch.Tensor) -> torch.Tensor:
        """Predict metabolite amplitudes from a complex time-domain FID.

        Parameters
        ----------
        fid:
            Real tensor of shape ``(batch, 2, n_points)`` holding the
            stacked real (channel 0) and imaginary (channel 1) parts of
            the FID.
        """
        features = self.encoder(fid)
        return self.head(features)


class MCDenseNetGGG(nn.Module):
    """Multi-input DenseNet with an AR-conditioned, GGG-constrained pathway.

    In addition to the dual-input time-domain encoder used by
    :class:`DCDenseNet`, this model accepts a vector of autoregressive
    (AR) coefficients estimated from the FID after band-limiting to the
    glutamate/glutamine/GABA (GGG) spectral region (see
    :mod:`dlquanttime.ar_features`). The AR pathway is a small MLP whose
    output is added only to the three GGG metabolite channels (assumed to
    be the last ``len(ggg_indices)`` -- or explicitly given -- output
    indices), structurally preventing it from influencing any other
    metabolite's prediction.
    """

    def __init__(
        self,
        n_metabolites: int,
        ggg_indices: Sequence[int],
        n_ar_features: int,
        n_points: int = 2048,
        growth_rate: int = 8,
        block_layers: Sequence[int] = (4, 4, 4),
        ar_hidden_dim: int = 16,
    ):
        super().__init__()
        if len(set(ggg_indices)) != len(ggg_indices):
            raise ValueError("ggg_indices must not contain duplicates")
        if any(idx < 0 or idx >= n_metabolites for idx in ggg_indices):
            raise ValueError("ggg_indices must be valid metabolite indices")

        self.n_points = n_points
        self.n_metabolites = n_metabolites
        self.register_buffer(
            "ggg_indices", torch.as_tensor(list(ggg_indices), dtype=torch.long)
        )

        self.encoder = _DenseNetEncoder1D(
            in_channels=2, growth_rate=growth_rate, block_layers=block_layers
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_metabolites),
        )

        # AR pathway: constrained to only ever produce len(ggg_indices)
        # outputs, which are scattered into the corresponding GGG channels.
        self.ar_pathway = nn.Sequential(
            nn.Linear(n_ar_features, ar_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ar_hidden_dim, len(ggg_indices)),
        )

    def forward(self, fid: torch.Tensor, ar_features: torch.Tensor) -> torch.Tensor:
        """Predict metabolite amplitudes from a time-domain FID and AR features.

        Parameters
        ----------
        fid:
            Real tensor of shape ``(batch, 2, n_points)``.
        ar_features:
            Real tensor of shape ``(batch, n_ar_features)`` of AR
            coefficients extracted from the GGG-band-limited FID.
        """
        features = self.encoder(fid)
        out = self.head(features)

        ar_contribution = self.ar_pathway(ar_features)
        ggg_update = torch.zeros_like(out)
        ggg_update.index_add_(1, self.ggg_indices, ar_contribution)
        return out + ggg_update
