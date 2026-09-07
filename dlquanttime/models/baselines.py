"""Baseline architectures lacking dense connectivity.

These serve as comparison points for the densely-connected DenseNet1D
models (:mod:`dlquanttime.models.densenet1d`):

* :class:`ConvAutoencoder` -- a convolutional autoencoder-style regressor
  (encoder-decoder bottleneck, no skip connections).
* :class:`SequentialCNN` -- a plain sequential (feed-forward) 1-D
  convolutional network with no skip or dense connections.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional-autoencoder-style metabolite quantification baseline.

    A purely sequential encoder compresses the (real, imaginary) FID down
    to a bottleneck vector; a symmetric decoder is trained jointly (via an
    auxiliary reconstruction loss, if desired) while a small regression
    head predicts metabolite amplitudes from the bottleneck. Unlike the
    DenseNet models, there is no dense/skip connectivity between layers.
    """

    def __init__(self, n_metabolites: int, n_points: int = 2048, bottleneck_dim: int = 64):
        super().__init__()
        self.n_points = n_points

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.bottleneck = nn.Linear(64, bottleneck_dim)

        self.decoder_fc = nn.Linear(bottleneck_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 2, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_metabolites),
        )

    def encode(self, fid: torch.Tensor) -> torch.Tensor:
        features = self.encoder(fid).squeeze(-1)
        return self.bottleneck(features)

    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        x = self.decoder_fc(latent).unsqueeze(-1)
        x = x.expand(-1, -1, max(target_length // 8, 1))
        recon = self.decoder(x)
        if recon.shape[-1] != target_length:
            recon = nn.functional.interpolate(recon, size=target_length)
        return recon

    def forward(self, fid: torch.Tensor, return_reconstruction: bool = False):
        latent = self.encode(fid)
        amplitudes = self.regression_head(latent)
        if return_reconstruction:
            recon = self.decode(latent, fid.shape[-1])
            return amplitudes, recon
        return amplitudes


class SequentialCNN(nn.Module):
    """Plain sequential 1-D CNN metabolite quantification baseline.

    A stack of ordinary (non-densely-connected) 1-D convolutions followed
    by global average pooling and a regression head.
    """

    def __init__(self, n_metabolites: int, n_points: int = 2048):
        super().__init__()
        self.n_points = n_points
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_metabolites),
        )

    def forward(self, fid: torch.Tensor) -> torch.Tensor:
        features = self.features(fid).squeeze(-1)
        return self.regression_head(features)
