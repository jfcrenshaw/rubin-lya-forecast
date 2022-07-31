"""Calculate du likelihoods."""
# %%
import pickle
from random import sample

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

# setup the grid of Delta u's on which to plot the likelihoods
deltas = np.linspace(-2, 3, 1000)
# %%
# and a dictionary to store the likelihoods
likelihood_dict = {}

# loop over the background catalogs
for file in bg_dir.glob("*.pkl"):
    stem = str(file.stem).removesuffix("_bg")
    print("calculating likelihoods for", stem)

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

    if name != "perfect":
        continue

    # loop over batches
    likelihoods = []
    batch_size = 100
    for idx in range(0, len(catalog), batch_size):
        # get the batch
        batch = catalog[idx : idx + batch_size]

        if name == "perfect":
            # for the perfect catalog, we will draw a ton more samples and
            # build a histogram
            z_samples, u_samples = sample_with_errors(batch, ensemble, 10_000, seed=idx)

            # flag the bad samples
            u_samples[z_samples < 2.36] = np.nan  # type: ignore

            # calculate Delta u from the samples of u and the true u
            delta_samples = u_samples - batch.u.to_numpy()[:, None]

            # create the histogram likelihoods
            diff = deltas[1] - deltas[0]
            bins = np.linspace(
                deltas.min() - diff, deltas.max() + diff, len(deltas) + 1
            )
            Ldu = [
                np.histogram(samples, bins, density=True)[0]
                for samples in delta_samples
            ]
            Ldu = np.array(Ldu)

        else:
            # draw redshift samples
            z_samples, u_samples = sample_with_errors(batch, ensemble, seed=idx)  # type: ignore

            # flag the bad samples
            u_samples[z_samples < 2.36] = np.nan

            # add the grid of deltas
            u_alpha = u_samples[:, None, :] + deltas[None, :, None]

            # convert to flux
            u_alpha = 10 ** (u_alpha / -2.5)
            u_alpha_hat = 10 ** (batch.u.to_numpy() / -2.5)
            u_alpha_err = u_alpha_hat * (10 ** (batch.u_err.to_numpy() / -2.5) - 1)

            # calculate the likelihoods
            Ldu = np.exp(
                -((u_alpha - u_alpha_hat[:, None, None]) ** 2)
                / (2 * u_alpha_err[:, None, None] ** 2)
            ) / np.sqrt(2 * np.pi * u_alpha_err[:, None, None] ** 2)
            Ldu = np.nanmean(Ldu, axis=2)

            # normalize
            norms = np.trapz(Ldu, deltas)
            norms[norms > 0] = 1 / norms[norms > 0]
            norms[norms <= 0] = 0
            Ldu = norms[:, None] * Ldu

        # add to the list!
        likelihoods.append(Ldu)
        break

    likelihoods = np.vstack(likelihoods)  # type: ignore
    likelihood_dict[name] = likelihoods

# %%

# %%
# sort the dictionary
euclid_likelihoods = likelihood_dict.pop("euclid")
roman_likelihoods = likelihood_dict.pop("roman")
perfect_likelihoods = likelihood_dict.pop("perfect")
likelihood_dict = dict(sorted(likelihood_dict.items()))
likelihood_dict["euclid"] = euclid_likelihoods
likelihood_dict["roman"] = roman_likelihoods
likelihood_dict["perfect"] = perfect_likelihoods

# save the deltas too!
likelihood_dict["deltas"] = deltas  # type: ignore

# save the likelihood dictionary
with open(paths.data / "likelihoods.pkl", "wb") as file:
    pickle.dump(likelihood_dict, file)
