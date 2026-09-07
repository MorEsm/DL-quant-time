"""DL-quant-time: deep learning framework for direct time-domain quantification
of brain metabolites from ¹H MR spectroscopy free induction decays (FIDs).

The package provides:

* :mod:`dlquanttime.simulate` -- a synthetic time-domain (FID) data generator.
* :mod:`dlquanttime.ar_features` -- autoregressive (AR) coefficient extraction
  from a band-limited signal (used for the Glu/Gln/GABA "GGG" spectral
  region, 2.1-2.6 ppm).
* :mod:`dlquanttime.metrics` -- Pearson correlation and ICC(2,1) agreement
  metrics used to validate quantification accuracy.
* :mod:`dlquanttime.losses` -- the metabolite-specific, inverse-amplitude
  weighted loss function.
* :mod:`dlquanttime.models` -- the DenseNet1D based architectures
  (DC-DenseNet and MC-DenseNet-GGG) as well as the baseline architectures
  (convolutional autoencoder and sequential convolutional network).
"""

from .constants import METABOLITES, GGG_METABOLITES, GGG_PPM_RANGE

__all__ = ["METABOLITES", "GGG_METABOLITES", "GGG_PPM_RANGE"]

__version__ = "0.1.0"
