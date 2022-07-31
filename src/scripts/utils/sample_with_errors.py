"""Function to sample from the ensemble while sampling from the photometric errors."""
from typing import Tuple

import numpy as np
import pandas as pd
from pzflow import FlowEnsemble

# set the number of samples for inference
m_samples = 100
zu_samples = 1


def sample_with_errors(
    catalog: pd.DataFrame,
    ensemble: FlowEnsemble,
    m_samples: int = m_samples,
    zu_samples: int = zu_samples,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from the ensemble while sampling from the photometric errors.

    Parameters
    ----------
    catalog: pd.DataFrame
        The catalog on which to condition the samples.
    ensemble: FlowEnsemble
        The PZFlow FlowEnsemble to sample from.
    m_samples: int
        The number of samples from the photometric error distribution.
    zu_samples: int
        The number of samples from p(z, u | m).
    seed: int
        The random seed for samples.

    Returns
    -------
    np.ndarray
        2D array of redshift samples
    np.ndarray
        2D array of u band samples
    """
    # get the list of conditional columns
    cols = ensemble.conditional_columns
    err_cols = [f"{col}_err" for col in cols]

    # get fluxes
    fluxes = 10 ** (catalog[cols].to_numpy() / -2.5)
    flux_errs = (10 ** (catalog[err_cols].to_numpy() / 2.5) - 1) * fluxes

    # sample from the photometric error distribution
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(fluxes.shape[0], m_samples, fluxes.shape[-1]))
    fluxes = fluxes[:, None, :] + flux_errs[:, None, :] * eps
    fluxes = fluxes.reshape(-1, fluxes.shape[-1])

    # add a flux floor to avoid infinite magnitudes
    # this flux corresponds to a max magnitude of 30
    fluxes = np.clip(fluxes, 1e-20, None)

    # convert back to magnitudes
    mags = -2.5 * np.log10(fluxes)

    # save samples in pandas dataframe
    conditions = pd.DataFrame(mags, columns=cols)

    # sample from the ensemble
    samples = ensemble.sample(
        zu_samples,
        conditions=conditions,
        save_conditions=False,
        seed=rng.integers(int(1e12)),
    )
    z_samples = samples["redshift"].to_numpy().reshape(catalog.shape[0], -1)
    u_samples = samples["u"].to_numpy().reshape(catalog.shape[0], -1)

    return z_samples, u_samples
