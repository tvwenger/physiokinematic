# `physiokinematic`  <!-- omit in toc -->
Physio-kinematic Distance Calculator

![publish](https://github.com/tvwenger/physiokinematic/actions/workflows/publish.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/physiokinematic/badge/?version=latest)](https://physiokinematic.readthedocs.io/en/latest/?badge=latest)

- [Installation](#installation)
- [Usage: `distance_model`](#usage-distance_model)
- [Usage: `hii_region_model`](#usage-hii_region_model)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)

`physiokinematic` implements a Bayesian model to predict the distances to objects in the Milky Way. It is essentially a fully-Bayesian kinematic distance calculator. Furthermore, for Galactic HII regions, `physiokinematic` also predicts various physical properties of the star forming regions, including the ionizing photon rate of the ionization source and the electron density. Given informative priors on these parameters, `physiokinematic` can probabilistically resolve the kinematic distance ambiguity.

# Installation

```python
mamba env create -f environment.yml
mamba activate physiokinematic
pip install -e .
```

# Usage: `distance_model`

The `distance_model` is a Bayesian kinematic distance model. The free parameters of the model are: the parameters defining the Persic (1996) Galactic rotation model (see also: Reid et al. 2019) and the Galactocentric radius of the object. The prior on the rotation curve parameters is a multivariate normal distribution, and the prior on the Galactocentric radius is a chi-squared distribution offset by the geometric minimum Galactocentric radius along the given line-of-sight. The likelihood is set by the observed LSR velocity with the assumption of systematic deviations from Galactic rotation.

# Usage: `hii_region_model`

The `hii_region_model` is a Bayesian kinematic distance model that also constrains the physical properties of a Galactic HII region. The additional free parameters are the slope and offset of the electron temperature vs. Galactocentric radius gradient, the ionizing photon rate of the ionizing source, the electron density, and the kinematic distance ambiguity resolution. The priors on the electron temperature gradient slope and offset are normal distributions. For the ionizing photon rate and electron density, the priors are log-normal distributions. The kinematic distance ambiguity resolution uses a Dirichlet prior. The likelihood includes observations of the LSR velocity, infrared angular size, and radio recombination line brightness. The later two are defined as log-normal distributions with user-defined widths to account for effects not predicted by the model, such as missing flux due to interferometric observations and the unknown difference between the infrared size and the Stromgren radius.

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/physiokinematic).

# License and Copyright

Copyright(C) 2024-2026 by Trey V. Wenger

This code is licensed under MIT license (see LICENSE for details)