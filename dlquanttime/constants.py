"""Shared constants describing the metabolites modelled by this package.

The chemical shift values are approximate singlet/dominant-peak positions
(in ppm) commonly used for simplified simulation of a ¹H brain MRS
spectrum. They are only used to build a physiologically plausible
synthetic dataset and are not intended to be spectroscopically exact.
"""

from collections import OrderedDict

#: The eight major brain metabolites quantified by the models in this
#: package, together with an approximate dominant chemical shift (ppm) and
#: typical linewidth-independent relative visibility used purely for
#: generating plausible synthetic amplitude ranges.
METABOLITES = OrderedDict(
    [
        ("NAA", 2.02),
        ("Cr", 3.03),
        ("PCr", 3.03),
        ("GPC", 3.21),
        ("Glu", 2.35),
        ("Gln", 2.45),
        ("GABA", 2.28),
        ("mI", 3.56),
    ]
)

#: Names of the metabolites used in the multi-input MC-DenseNet-GGG
#: configuration, i.e. those whose resonances overlap the 2.1-2.6 ppm
#: glutamate/glutamine/GABA ("GGG") spectral region.
GGG_METABOLITES = ("Glu", "Gln", "GABA")

#: The band-limiting frequency range (in ppm) used to isolate the
#: glutamate/glutamine/GABA spectral region before autoregressive (AR)
#: modelling, as described in the problem statement.
GGG_PPM_RANGE = (2.1, 2.6)
