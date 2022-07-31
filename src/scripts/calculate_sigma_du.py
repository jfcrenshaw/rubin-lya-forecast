"""Calculate du likelihoods."""
import pickle

import numpy as np
import pandas as pd
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths
from utils import sample_with_errors

# instantiate the paths
paths = Paths()

# load the flow ensembles
model_dir = paths.data / "models"
flow_ensembles = {
    "lsst": FlowEnsemble(file=model_dir / "lsst_ensemble.pzflow.pkl"),
    "lsst+euclid": FlowEnsemble(file=model_dir / "lsst+euclid_ensemble.pzflow.pkl"),
    "lsst+roman": FlowEnsemble(file=model_dir / "lsst+roman_ensemble.pzflow.pkl"),
    "perfect": FlowEnsemble(file=model_dir / "lsst+roman_ensemble.pzflow.pkl"),
}

# directory where the background catalogs are saved
bg_dir = paths.data / "background_catalogs"

# and a dictionary to store the sigma_du values
sigma_du_dict = {}

# loop over the background catalogs
for file in bg_dir.glob("*.pkl"):
    stem = str(file.stem).removesuffix("_bg")
    print("calculating sigma_du for", stem)

    # load the catalog
    catalog = pd.read_pickle(file)

    # select the correct flow ensemble
    if "euclid" in stem:
        name = "euclid"
        ensemble = flow_ensembles["lsst+euclid"]
    elif "roman" in stem:
        name = "roman"
        ensemble = flow_ensembles["lsst+roman"]
    elif "perfect" in stem:
        name = "perfect"
        ensemble = flow_ensembles["perfect"]
    else:
        name = int(stem[5:])  # type: ignore
        ensemble = flow_ensembles["lsst"]

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
    # this flux corresponds to a max magnitude of 30
    fluxes = np.clip(fluxes, 1e-20, None)

    # convert back to magnitudes
    u_alpha = -2.5 * np.log10(fluxes)

    # calculate the du_samples
    du_samples = u_alpha - u_samples

    # get the mean for each galaxy
    du_samples = np.nanmean(du_samples, axis=1)

    # calculate sigma_du from the standard deviation of the du_samples
    sigma_du = np.nanstd(du_samples)

    sigma_du_dict[name] = sigma_du


# sort the dictionary
euclid_sigma_du = sigma_du_dict.pop("euclid")
roman_sigma_du = sigma_du_dict.pop("roman")
perfect_sigma_du = sigma_du_dict.pop("perfect")
sigma_du_dict = dict(sorted(sigma_du_dict.items()))
sigma_du_dict["euclid"] = euclid_sigma_du
sigma_du_dict["roman"] = roman_sigma_du
sigma_du_dict["perfect"] = perfect_sigma_du

# save the likelihood dictionary
with open(paths.data / "sigma_du.pkl", "wb") as file:
    pickle.dump(sigma_du_dict, file)
