"""Command-line training script for the time-domain MRS quantification models.

Example
-------
    python -m dlquanttime.train --model dc_densenet --n-train 30000 \
        --n-val 1000 --epochs 20

Requires PyTorch to actually run training; ``--help`` works without it.
"""

import argparse
from typing import Optional

import numpy as np

from .constants import METABOLITES
from .data_utils import build_ar_feature_batch, fid_to_channels, ggg_indices
from .simulate import SimulationConfig, generate_dataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["dc_densenet", "mc_densenet_ggg", "conv_autoencoder", "sequential_cnn"],
        default="dc_densenet",
    )
    parser.add_argument("--n-train", type=int, default=30000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ar-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to save the trained model's state and configuration to.",
    )
    return parser


def _build_model(model_name: str, n_metabolites: int, n_ar_features: int, ggg_idx):
    from .models import ConvAutoencoder, DCDenseNet, MCDenseNetGGG, SequentialCNN

    if model_name == "dc_densenet":
        return DCDenseNet(n_metabolites)
    if model_name == "mc_densenet_ggg":
        return MCDenseNetGGG(n_metabolites, ggg_idx, n_ar_features)
    if model_name == "conv_autoencoder":
        return ConvAutoencoder(n_metabolites)
    if model_name == "sequential_cnn":
        return SequentialCNN(n_metabolites)
    raise ValueError(f"Unknown model: {model_name}")


def train(args: Optional[argparse.Namespace] = None) -> "torch.nn.Module":
    """Train a metabolite quantification model on synthetic data.

    Returns the trained model. If ``args.checkpoint`` is set, the model's
    state dict and the configuration needed to reconstruct it are also
    saved to that path (see :func:`load_checkpoint`).
    """
    args = args or build_argparser().parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .losses import InverseAmplitudeWeightedMSELoss

    metabolites = tuple(METABOLITES.keys())
    idx = ggg_indices(metabolites)

    cfg = SimulationConfig(metabolites=metabolites, seed=args.seed)
    train_fids, train_amps = generate_dataset(args.n_train, cfg)
    val_fids, val_amps = generate_dataset(args.n_val, cfg)

    train_channels = fid_to_channels(train_fids)
    val_channels = fid_to_channels(val_fids)

    model = _build_model(args.model, len(metabolites), 2 * args.ar_order, idx)
    criterion = InverseAmplitudeWeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_train = torch.tensor(train_channels, dtype=torch.float32)
    y_train = torch.tensor(train_amps, dtype=torch.float32)
    x_val = torch.tensor(val_channels, dtype=torch.float32)
    y_val = torch.tensor(val_amps, dtype=torch.float32)

    if args.model == "mc_densenet_ggg":
        train_ar = build_ar_feature_batch(
            train_fids, cfg.dwell_time, cfg.field_strength_mhz, order=args.ar_order
        )
        val_ar = build_ar_feature_batch(
            val_fids, cfg.dwell_time, cfg.field_strength_mhz, order=args.ar_order
        )
        ar_train = torch.tensor(train_ar, dtype=torch.float32)
        ar_val = torch.tensor(val_ar, dtype=torch.float32)
        dataset = TensorDataset(x_train, ar_train, y_train)
    else:
        dataset = TensorDataset(x_train, y_train)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            if args.model == "mc_densenet_ggg":
                x_batch, ar_batch, y_batch = batch
                preds = model(x_batch, ar_batch)
            else:
                x_batch, y_batch = batch
                preds = model(x_batch)
                if isinstance(preds, tuple):
                    preds = preds[0]
            loss = criterion(y_batch, preds)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.shape[0]
        epoch_loss /= len(dataset)

        model.eval()
        with torch.no_grad():
            if args.model == "mc_densenet_ggg":
                val_preds = model(x_val, ar_val)
            else:
                val_preds = model(x_val)
                if isinstance(val_preds, tuple):
                    val_preds = val_preds[0]
            val_loss = criterion(y_val, val_preds).item()
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

    if args.checkpoint:
        torch.save(
            {
                "model_name": args.model,
                "n_metabolites": len(metabolites),
                "n_ar_features": 2 * args.ar_order,
                "ggg_indices": idx,
                "state_dict": model.state_dict(),
            },
            args.checkpoint,
        )
        print(f"Saved checkpoint to {args.checkpoint}")

    return model


def load_checkpoint(path: str) -> "torch.nn.Module":
    """Load a model previously saved by :func:`train` via ``--checkpoint``.

    .. warning::
        Checkpoints are deserialized with :func:`torch.load` using
        ``weights_only=True`` to avoid executing arbitrary code from
        untrusted pickle payloads. Nonetheless, checkpoints should only be
        loaded from sources you trust.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = _build_model(
        checkpoint["model_name"],
        checkpoint["n_metabolites"],
        checkpoint["n_ar_features"],
        checkpoint["ggg_indices"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


if __name__ == "__main__":
    train()
