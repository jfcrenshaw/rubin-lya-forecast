"""Function to sample from the ensemble while sampling from the photometric errors."""
from typing import Tuple

import numpy as np
import pandas as pd
from pzflow import FlowEnsemble


def sample_with_errors(
    catalog: pd.DataFrame,
    ensemble: FlowEnsemble,
    M: int,
    N: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from the ensemble while sampling from the photometric errors.

    Parameters
    ----------
    catalog: pd.DataFrame
        The catalog on which to condition the samples.
    ensemble: FlowEnsemble
        The PZFlow FlowEnsemble to sample from.
    M: int
        The number of samples to draw from the photometric error distributions.
    N: int
        The number of z, u samples to draw.
    seed: int
        The random seed for samples.

    Returns
    -------
    np.ndarray
        2D array of redshift samples, shape: (len(catalog), N)
    np.ndarray
        2D array of u band samples, shape: (len(catalog), N)
    """
    # get the list of conditional columns
    cols = ensemble.conditional_columns
    err_cols = [f"{col}_err" for col in cols]

    # get fluxes
    fluxes = 10 ** (catalog[cols].to_numpy() / -2.5)
    flux_errs = (10 ** (catalog[err_cols].to_numpy() / 2.5) - 1) * fluxes

    # get M samples from the photometric error distributions
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(fluxes.shape[0], M, fluxes.shape[-1]))
    fluxes = fluxes[:, None, :] + flux_errs[:, None, :] * eps
    fluxes = fluxes.reshape(-1, fluxes.shape[-1])

    # add a flux floor to avoid infinite magnitudes
    # this flux corresponds to a max magnitude of 30
    fluxes = np.clip(fluxes, 1e-12, None)

    # convert back to magnitudes
    mags = -2.5 * np.log10(fluxes)

    # save samples in pandas dataframe
    conditions = pd.DataFrame(mags, columns=cols)

    # sample from the ensemble
    samples = ensemble.sample(
        N,
        conditions=conditions,
        save_conditions=False,
        seed=rng.integers(int(1e12)),
    )
    z_samples = samples["redshift"].to_numpy().reshape(catalog.shape[0], -1)
    u_samples = samples["u"].to_numpy().reshape(catalog.shape[0], -1)

    return z_samples, u_samples
