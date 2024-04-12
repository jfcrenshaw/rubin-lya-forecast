"""Utility function that converts a truth catalog to an observed catalog."""

import pandas as pd
from .calculate_extinction import ExtinctionCalculator
from .misc import split_seed
from photerr import EuclidErrorModel, LsstErrorModel, RomanErrorModel
import numpy as np
from scipy.interpolate import griddata


def observe_catalog(
    cat_truth: pd.DataFrame,
    n_years: float,
    min_snr: float,
    euclid: bool,
    roman: bool,
    seed: int,
) -> pd.DataFrame:
    """Convert truth catalog to an observed catalog.

    Parameters
    ----------
    cat_truth : pd.DataFrame
        The truth catalog
    n_years : float
        The number of years observed in LSST
    min_snr : float
        The minimum SNR in the i band
    euclid : bool
        Whether to include Euclid bands
    roman : bool
        Whether to include Roman bands
    seed : int
        The random seed

    Returns
    -------
    pd.DataFrame
        The observed catalog
    """
    if euclid and roman:
        raise ValueError("euclid and roman cannot both be True.")

    # Split the seed
    seed_lsst, seed_euclid, seed_roman, seed_replace = split_seed(seed, 4)

    # Create the observed catalog
    cat_obs = cat_truth.copy()

    # Cut the low and high redshift artifacts
    cat_obs = cat_obs[(cat_obs.redshift > 0.2) & (cat_obs.redshift < 3.5)]

    # Add Lya extinction
    ec = ExtinctionCalculator()
    cat_obs["u0"] = cat_obs.u.copy()
    cat_obs.u += ec.lya_increment(cat_obs.redshift, "u")
    """
    # Add LSST errors
    lsst_error_model = LsstErrorModel(
        nYrObs=n_years,
        airmass=1.2,
        sigLim=1e-3,
    )
    cat_obs = lsst_error_model(cat_obs, random_state=seed_lsst)
    i_limit = lsst_error_model.getLimitingMags(nSigma=min_snr)["i"]
    cat_obs = cat_obs[cat_obs.i <= i_limit]

    # Hacky way to handle non-detections
    # Drop the few galaxies that aren't detected in all bands > u
    cat_obs = cat_obs[np.isfinite(cat_obs.drop(["u", "u_err"], axis=1)).all(axis=1)]

    # Replace u band non-detects with mags from similar galaxies
    idx_det = np.where(np.isfinite(cat_obs.u))
    idx_ndet = np.where(~np.isfinite(cat_obs.u))
    drop_cols = ["u"] + [col for col in cat_obs if "_err" in col]
    u_replace = griddata(
        cat_obs.iloc[idx_det].drop(drop_cols, axis=1).values,
        cat_obs.iloc[idx_det].u.values,
        cat_obs.iloc[idx_ndet].drop(drop_cols, axis=1).values,
        method="nearest",
    )
    rng = np.random.default_rng(seed_replace)
    u_replace = rng.normal(u_replace, 0.05)  # Add some random scatter
    cat_obs.iloc[idx_ndet[0], cat_obs.columns.get_loc("u")] = u_replace

    lsst_error_model.params.decorrelate = False
    lsst_error_model.params.ndMode = "sigLim"
    cat_obs.loc[:, "u_err"] = lsst_error_model(cat_obs[["u"]])["u_err"]
    """
    # Add LSST errors
    lsst_error_model = LsstErrorModel(
        nYrObs=n_years,
        airmass=1.2,
        sigLim=0,
        absFlux=True,
    )
    cat_obs = lsst_error_model(cat_obs, random_state=seed_lsst)
    i_limit = lsst_error_model.getLimitingMags(nSigma=min_snr)["i"]
    cat_obs = cat_obs[cat_obs.i <= i_limit]

    # Handle Euclid and Roman bands
    if euclid:
        euclid_error_model = EuclidErrorModel()
        cat_obs = euclid_error_model(cat_obs, random_state=seed_euclid)
        cat_obs = cat_obs.drop("F", axis=1)
    elif roman:
        roman_error_model = RomanErrorModel()
        cat_obs = roman_error_model(cat_obs, random_state=seed_roman)
    else:
        cat_obs = cat_obs.drop(list("YJHF"), axis=1)

    return cat_obs
