# DL-quant-time
Deep Learning for Direct Time-Domain Quantification of Brain Metabolites in ¹H MR Spectroscopy

This repository implements a deep learning framework for direct quantification of
brain metabolites from raw time-domain proton (¹H) MR spectroscopy (MRS) free
induction decays (FIDs), as described in the accompanying study. It includes:

* A synthetic time-domain FID data generator (`dlquanttime.simulate`) that
  simulates physiologically plausible metabolite concentrations, linewidths
  and noise levels for eight major brain metabolites (NAA, Cr, PCr, GPC, Glu,
  Gln, GABA, mI).
* Autoregressive (AR) coefficient extraction (`dlquanttime.ar_features`) from
  the FID after band-limiting to the glutamate/glutamine/GABA ("GGG")
  spectral region (2.1-2.6 ppm), using the Burg method.
* A modified one-dimensional DenseNet architecture (`dlquanttime.models`) in
  two configurations:
  * **DC-DenseNet** -- dual-input (real/imaginary FID channels) model.
  * **MC-DenseNet-GGG** -- multi-input model additionally conditioned on the
    band-limited AR coefficients, injected via a pathway structurally
    constrained to influence only the Glu, Gln and GABA output channels.
* Two baseline architectures lacking dense connectivity for comparison: a
  convolutional autoencoder (`ConvAutoencoder`) and a sequential
  convolutional network (`SequentialCNN`).
* A metabolite-specific, inverse-amplitude-weighted loss function
  (`dlquanttime.losses`) that gives low-concentration metabolites comparable
  influence on training regardless of their natural amplitude scale.
* Evaluation metrics (`dlquanttime.metrics`) -- Pearson correlation and the
  intraclass correlation coefficient ICC(2,1) -- used to validate
  quantification accuracy against ground-truth or reference values.

## Installation

```bash
pip install -r requirements.txt
```

(`torch` is only required for the model/training code; the data simulation,
AR feature extraction, loss and metric modules only depend on `numpy`/`scipy`
and can be used/tested without it.)

## Usage

Train a model on synthetic data:

```bash
python -m dlquanttime.train --model dc_densenet --n-train 30000 --n-val 1000 --epochs 20
python -m dlquanttime.train --model mc_densenet_ggg --n-train 30000 --n-val 1000 --epochs 20
```

Evaluate per-metabolite agreement (Pearson r, ICC(2,1)) on synthetic data:

```bash
python -m dlquanttime.evaluate --model dc_densenet --n-test 1000
```

## Tests

```bash
pip install pytest
pytest tests/
```
