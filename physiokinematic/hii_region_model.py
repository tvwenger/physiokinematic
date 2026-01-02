"""
hii_region_model.py
Bayesian physio-kinematic distance model for Galactic HII regions.

Copyright(C) 2023-2025 by
Trey V. Wenger; tvwenger@gmail.com
Ryan Bakko
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np
import astropy.constants as c

from physiokinematic import utils


def hii_region_model(
    data: Iterable[float],
    prior_reid19_mean: Iterable[float],
    prior_reid19_cov: Iterable[float],
    prior_Rgal: Optional[float] = 5.0,
    prior_te_offset: Optional[Iterable[float]] = [4500.0, 200.0],
    prior_te_slope: Optional[Iterable[float]] = [360.0, 25.0],
    prior_log10_q: Optional[Iterable[float]] = [47.0, 2.0],
    prior_log10_ne: Optional[Iterable[float]] = [1.0, 2.0],
    prior_kdar: Optional[Iterable[float]] = [0.5, 0.5],
    log10_radius_sigma: Optional[float] = 0.1,
    log10_line_sigma: Optional[float] = 0.2,
    vlsr_sigma: Optional[float] = 10.0,
    te_sigma: Optional[float] = 500.0,
):
    """Generates an instance of the kinematic distance model.

    Parameters
    ----------
    data : Iterable[float]
        Data for a single HII region. Keys must include: 'glong', 'glat', 'vlsr', 'e_vlsr',
        'line_freq', 'fwhm', 'beam_area', 'line', 'radius', 'te', 'e_te'
    prior_reid19_mean : Iterable[float]
        Prior mean rotation curve R0, Usun, Vsun, Wsun, a2, a3
    prior_reid19_cov : Iterable[float]
        Prior covariance R0, Usun, Vsun, Wsun, a2, a3
    prior_Rgal : Optional[float], optional
        Prior width for Galactocentric radius (kpc), by default 2.0, where
        Rgal ~ Rmin + ChiSquared(nu=1) * prior_Rgal
    prior_te_offset : Optional[Iterable[float]], optional
        Prior mean and width for electron temperature offset (K), by default [4500.0, 200.0], where
        te_offset ~ Normal(prior_te_offset[0], prior_te_offset[1])
    prior_te_slope : Optional[Iterable[float]], optional
        Prior mean and width for electron temperature slope (K/kpc), by default [360.0, 25.0], where
        te_slope ~ Normal(prior_te_slope[0], prior_te_slope[1])
    prior_log10_q : Optional[Iterable[float]], optional
        Prior mean and width for log10 ionizing photon rate (s-1), by default [47.0, 2.0], where
        log10_q ~ Normal(prior_log10_q[0], prior_log10_q[1])
    prior_log10_ne : Optional[Iterable[float]], optional
        Prior mean and width for log10 electron density (cm-3), by default [1.0, 2.0], where
        log10_ne ~ Normal(prior_log10_ne[0], prior_log10_ne[1])
    prior_kdar : Optional[Iterable[float]], optional
        Prior for near and far kinematic distance ambiguity resolution, by default [0.5, 0.5]
    log10_radius_sigma : Optional[float], optional
        Apparent radius likelihood width, by default 0.1
    log10_line_sigma : Optional[float], optional
        Line brightness likelihood width, by default 0.2
    vlsr_sigma : Optional[float], optional
        Assumed systematic velocity uncertainty (km/s), by default 10.0
    te_sigma : Optional[float], optional
        Assumed systematic electron temperature uncertainty (K), by default 500.0

    Returns
    -------
    pm.Model
        Model instance
    """
    # line width in kHz
    data["fwhm_kHz"] = 1000.0 * data["line_freq"] * data["fwhm"] / c.c.to("km/s").value

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
        Rtan = R0 * np.abs(np.sin(np.deg2rad(data["glong"])))
        if np.cos(np.deg2rad(data["glong"])) < 0.0:
            Rmin = R0
        else:
            Rmin = Rtan
        Rgal_norm = pm.ChiSquared("Rgal_norm", nu=1.0)
        Rgal = pm.Deterministic("Rgal", Rmin + prior_Rgal * Rgal_norm)

        # LSR velocity (km/s)
        vlsr_mu = utils.reid19_vlsr(
            data["glong"],
            data["glat"],
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
            sigma=pt.sqrt(data["e_vlsr"] ** 2.0 + vlsr_sigma**2.0),
            observed=data["vlsr"],
        )

        # Electron temperature (K)
        te_offset_norm = pm.Normal("te_offset_norm")
        te_offset = pm.Deterministic(
            "te_offset", prior_te_offset[0] + prior_te_offset[1] * te_offset_norm
        )
        te_slope_norm = pm.Normal("te_slope_norm")
        te_slope = pm.Deterministic(
            "te_slope", prior_te_slope[0] + prior_te_slope[1] * te_slope_norm
        )
        te_mu = te_offset + Rgal * te_slope
        if ~np.isnan(data["te"]):
            te = pm.Normal(
                "te",
                mu=te_mu,
                sigma=pt.sqrt(data["e_te"] ** 2.0 + te_sigma**2.0),
                observed=data["te"],
            )
        else:
            te_norm = pm.Normal("te_norm")
            te = pm.Deterministic("te", te_sigma * te_norm + te_mu)

        # Distance (kpc)
        x = pt.sqrt(Rgal**2.0 - Rtan**2.0)
        dtan = R0 * np.cos(np.deg2rad(data["glong"]))
        dist_near = dtan - x
        dist_far = dtan + x
        distance = pt.stack([dist_near, dist_far])
        distance = pm.Deterministic("distance", distance, dims="kdar")
        abs_distance = pm.Deterministic("abs_distance", pt.abs(distance), dims="kdar")

        # KDAR weight
        kdar_w = pm.Categorical("kdar_w", prior_kdar)

        # Ionizing photon rate
        log10_q_norm = pm.Normal("log10_q_norm", dims="kdar")
        log10_q = pm.Deterministic(
            "log10_q", prior_log10_q[0] + prior_log10_q[1] * log10_q_norm, dims="kdar"
        )

        # Electron density
        log10_ne_norm = pm.Normal("log10_ne_norm", dims="kdar")
        log10_ne = pm.Deterministic(
            "log10_ne",
            prior_log10_ne[0] + prior_log10_ne[1] * log10_ne_norm,
            dims="kdar",
        )
        # Stromgren radius (pc)
        log10_Rs = pm.Deterministic(
            "log10_Rs",
            np.log10(1.4) - 2.0 / 3.0 * (log10_ne - 2.0) + 1.0 / 3.0 * (log10_q - 46.0),
            dims="kdar",
        )

        # Emission measure (pc cm-6)
        log10_em = pm.Deterministic(
            "log10_em", np.log10(2.0) + log10_Rs + 2.0 * log10_ne, dims="kdar"
        )

        # Apparent radius (arcsec)
        log10_radius_mu = pm.Deterministic(
            "log10_radius_mu",
            np.log10(206265.0)
            + log10_Rs
            - np.log10(1000.0 * abs_distance + 10.0),  # prevent div by zero
            dims="kdar",
        )
        sel_log10_radius = pt.switch(
            pt.eq(kdar_w, 0), log10_radius_mu[0], log10_radius_mu[1]
        )
        _ = pm.Normal(
            "log10_radius_obs",
            mu=sel_log10_radius,
            sigma=log10_radius_sigma,
            observed=np.log10(0.5 * data["radius"]),
        )

        # RRL optical depth
        log10_tau_line = pm.Deterministic(
            "log10_tau_line",
            np.log10(1.92e3)
            - 2.5 * pt.log10(te)
            + log10_em
            - np.log10(data["fwhm_kHz"]),
            dims="kdar",
        )

        # Intinsic RRL brightness (mJy/arcsec2)
        area = np.pi * (10.0**log10_radius_mu) ** 2.0 / (4.0 * np.log(2.0))
        log10_exp_tau_line = pt.log10(1.0 - pt.exp(-(10.0**log10_tau_line)))
        log10_line_mu = pm.Deterministic(
            "log10_line_mu",
            pt.log10(
                2.0
                * area
                / 206265.0**2.0
                * (c.k_B / c.c**2.0).to("mJy MHz-2 K-1").value
                * data["line_freq"] ** 2.0
                * te
            )
            + log10_exp_tau_line,
            dims="kdar",
        )

        # Observed RRL brightness (mJy/beam)
        log10_line_mu_obs = pt.switch(
            pt.gt(area, data["beam_area"]),
            log10_line_mu,
            log10_line_mu + pt.log10(area / data["beam_area"]),
        )
        sel_log10_line_obs = pt.switch(
            pt.eq(kdar_w, 0), log10_line_mu_obs[0], log10_line_mu_obs[1]
        )
        _ = pm.Normal(
            "log10_line_obs",
            mu=sel_log10_line_obs,
            sigma=log10_line_sigma,
            observed=np.log10(data["line"]),
        )

    return model
