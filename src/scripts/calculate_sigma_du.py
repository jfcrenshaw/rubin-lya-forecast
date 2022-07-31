"""Calculate effective per-galaxy error on du."""
import pickle

import numpy as np
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the dictionary of likelihoods
with open(paths.data / "likelihoods.pkl", "rb") as file:
    likelihood_dict = pickle.load(file)

# calculate sigma du for each set of likelihoods
sigma_du_dict = {}
deltas = likelihood_dict.pop("deltas")
for name, likelihoods in likelihood_dict.items():
    # a small fraction of the likelihoods have artifacts - we will remove these
    # remove likelihoods with many modes
    idx = np.argsort(likelihoods, axis=1)[:, -50:]
    idx.sort(axis=1)
    flag1 = np.all(np.diff(idx) == 1, axis=1)
    # remove likelihoods that have all of their mass below Delta = -1
    flag2 = (np.cumsum(likelihoods, axis=1) * (deltas[1] - deltas[0]))[
        :, np.abs(deltas + 1).argmin()
    ] < 0.99
    # combine these two cuts
    flag = flag1 & flag2

    # now combine all of the likelihoods
    L = np.log(likelihoods[flag]).sum(axis=0)
    L = np.exp(L - L.max())
    L /= np.trapz(L, deltas)

    # calculate the variance of the combined likelihood
    mean = np.trapz(deltas * L, deltas)
    variance = np.trapz((mean - deltas) ** 2 * L, deltas)

    # convert this to the effective error per galaxy
    sigma_du = np.sqrt(variance) * np.sqrt(flag.sum())

    sigma_du_dict[name] = sigma_du

# save the dictionary of sigma_du's
with open(paths.data / "sigma_du.pkl", "wb") as file:
    pickle.dump(sigma_du_dict, file)
