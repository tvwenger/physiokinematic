"""
distance_model.py
Bayesian kinematic distance model.

Copyright(C) 2023-2025 by
Trey V. Wenger; tvwenger@gmail.com
Ryan Bakko
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from physiokinematic import utils


def distance_model(
    glong: float,
    glat: float,
    vlsr: float,
    e_vlsr: float,
    prior_reid19_mean: Iterable[float],
    prior_reid19_cov: Iterable[float],
    prior_Rgal: Optional[float] = 5.0,
    prior_kdar: Optional[Iterable[float]] = [0.5, 0.5],
    vlsr_sigma: Optional[float] = 10.0,
):
    """Generates an instance of the kinematic distance model.

    Parameters
    ----------
    glong : float
        Galactic longitude (degrees)
    glat : float
        Galactic latitude (degrees)
    vlsr : float
        LSR velocity (km/s)
    e_vlsr : float
        LSR velocity uncertainty (km/s)
    prior_reid19_mean : Iterable[float]
        Prior mean rotation curve R0, Usun, Vsun, Wsun, a2, a3
    prior_reid19_cov : Iterable[float]
        Prior covariance R0, Usun, Vsun, Wsun, a2, a3
    prior_Rgal : Optional[float], optional
        Prior width for Galactocentric radius (kpc), by default 2.0, where
        Rgal ~ Rmin + ChiSquared(nu=1) * prior_Rgal
    prior_kdar : Optional[Iterable[float]], optional
        Prior for near and far kinematic distance ambiguity resolution, by default [0.5, 0.5]
    vlsr_sigma : Optional[float], optional
        Assumed systematic velocity uncertainty (km/s), by default 10.0

    Returns
    -------
    pm.Model
        Model instance
    """
    with pm.Model(
        coords={
            "kdar": ["N", "F"],
            "rotparam": ["R0", "Usun", "Vsun", "Wsun", "a2", "a3"],
        }
    ) as model:
        # Rotation curve parameters
        rotcurve = pm.MvNormal(
            "rotcurve", mu=prior_reid19_mean, cov=prior_reid19_cov, dims="rotparam"
        )
        R0, Usun, Vsun, Wsun, a2, a3 = rotcurve

        # Galactocentric radius (kpc)
        Rtan = R0 * np.abs(np.sin(np.deg2rad(glong)))
        if np.cos(np.deg2rad(glong)) < 0.0:
            Rmin = R0
        else:
            Rmin = Rtan
        Rgal_norm = pm.ChiSquared("Rgal_norm", nu=1.0)
        Rgal = pm.Deterministic("Rgal", Rmin + prior_Rgal * Rgal_norm)

        # LSR velocity (km/s)
        vlsr_mu = utils.reid19_vlsr(
            glong,
            glat,
            Rgal,
            R0=R0,
            a2=a2,
            a3=a3,
            Usun=Usun,
            Vsun=Vsun,
            Wsun=Wsun,
        )

        # Observed LSR velocity (km/s)
        _ = pm.Normal(
            "vlsr_obs",
            mu=vlsr_mu,
            sigma=pt.sqrt(e_vlsr**2.0 + vlsr_sigma**2.0),
            observed=vlsr,
        )

        # Distance (kpc)
        dist_near, dist_far = utils.distance(glong, glat, Rgal, R0=R0)
        distance = pt.stack([dist_near, dist_far])
        distance = pm.Deterministic("distance", distance, dims="kdar")

        # KDAR weight
        _ = pm.Categorical("kdar_w", prior_kdar)

    return model
