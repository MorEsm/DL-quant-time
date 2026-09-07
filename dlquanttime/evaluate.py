"""Evaluation script: computes Pearson r and ICC(2,1) per metabolite.

Example
-------
    python -m dlquanttime.evaluate --model dc_densenet --n-test 1000
"""

import argparse
from typing import Dict

import numpy as np

from .constants import METABOLITES
from .data_utils import build_ar_feature_batch, fid_to_channels, ggg_indices
from .metrics import icc, pearson_r
from .simulate import SimulationConfig, generate_dataset


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["dc_densenet", "mc_densenet_ggg", "conv_autoencoder", "sequential_cnn"],
        default="dc_densenet",
    )
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--ar-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, metabolites
) -> Dict[str, Dict[str, float]]:
    """Compute per-metabolite Pearson r and ICC(2,1) between two amplitude arrays."""
    results = {}
    for i, name in enumerate(metabolites):
        results[name] = {
            "pearson_r": pearson_r(y_true[:, i], y_pred[:, i]),
            "icc": icc(y_true[:, i], y_pred[:, i]),
        }
    return results


def evaluate(args=None) -> Dict[str, Dict[str, float]]:
    """Evaluate a (randomly-initialized, for smoke-testing) model on synthetic data."""
    args = args or build_argparser().parse_args()

    import torch

    from .train import _build_model

    metabolites = tuple(METABOLITES.keys())
    idx = ggg_indices(metabolites)

    cfg = SimulationConfig(metabolites=metabolites, seed=args.seed)
    fids, amps = generate_dataset(args.n_test, cfg)
    channels = fid_to_channels(fids)

    model = _build_model(args.model, len(metabolites), 2 * args.ar_order, idx)
    model.eval()
    x = torch.tensor(channels, dtype=torch.float32)

    with torch.no_grad():
        if args.model == "mc_densenet_ggg":
            ar = build_ar_feature_batch(
                fids, cfg.dwell_time, cfg.field_strength_mhz, order=args.ar_order
            )
            preds = model(x, torch.tensor(ar, dtype=torch.float32))
        else:
            preds = model(x)
            if isinstance(preds, tuple):
                preds = preds[0]
    preds = preds.numpy()

    results = evaluate_predictions(amps, preds, metabolites)
    for name, metrics in results.items():
        print(f"{name}: r={metrics['pearson_r']:.3f} ICC={metrics['icc']:.3f}")
    return results


if __name__ == "__main__":
    evaluate()
