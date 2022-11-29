"""Calculate du likelihoods."""
import pickle

import numpy as np
import pandas as pd
from utils import load_ensemble, paths, sample_with_errors

# create a dictionary to store the sigma_du values
sigma_du_dict = {}

# loop over the background catalogs
for file in paths.bg.glob("*.pkl"):
    name = str(file.stem).removesuffix("_bg")
    print("calculating sigma_du for", name)

    # load the catalog
    catalog = pd.read_pickle(file)

    # and the flow ensemble
    ensemble = load_ensemble(name)

    # draw samples from the ensemble
    z_samples, u_samples = sample_with_errors(catalog, ensemble, seed=0)

    # flag the low-redshift samples
    u_samples[z_samples < 2.36] = np.nan

    # convert the observed u_alpha to flux
    fluxes = 10 ** (catalog.u.to_numpy() / -2.5)
    flux_errs = (10 ** (catalog.u_err.to_numpy() / 2.5) - 1) * fluxes

    # draw samples from the photometric error distribution
    rng = np.random.default_rng(0)
    eps = rng.normal(size=u_samples.shape)
    fluxes = fluxes[:, None] + flux_errs[:, None] * eps

    # add a flux floor to avoid infinite magnitudes
    # this flux corresponds to a max magnitude of 40
    fluxes = np.clip(fluxes, 1e-16, None)

    # convert back to magnitudes
    u_alpha = -2.5 * np.log10(fluxes)

    # calculate the du_samples
    du_samples = u_alpha - u_samples

    # get the mean for each galaxy
    du_samples = np.nanmedian(du_samples, axis=1)

    # calculate sigma_du from the standard deviation of the du_samples
    q75, q25 = np.nanpercentile(du_samples, [75, 25])
    iqr = q75 - q25
    sigma_du = iqr / 1.35

    print(np.nanmedian(du_samples), "+/-", sigma_du / np.sqrt(du_samples.size))
    print(
        (np.nanmedian(du_samples) - 0.20548664) / (sigma_du / np.sqrt(du_samples.size))
    )

    sigma_du_dict[name] = sigma_du

# save the likelihood dictionary
with open(paths.data / "sigma_du.pkl", "wb") as file:
    pickle.dump(sigma_du_dict, file)
