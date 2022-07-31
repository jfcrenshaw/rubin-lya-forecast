# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths
from utils import lya_decrement, sample_with_errors

paths = Paths()

# %%
catalog = pd.read_pickle(paths.data / "background_catalogs" / "perfect_bg.pkl")

# %%
ensemble = FlowEnsemble(file=paths.data / "models" / "lsst+roman_ensemble.pzflow.pkl")

# %%
z_samples, u_samples = sample_with_errors(catalog, ensemble, seed=0)

# %%
du_samples = catalog.u.to_numpy()[:, None] - u_samples
du_samples[z_samples < 2.36] = np.nan
np.nanmean(du_samples)

def mean(N):
    return np.nanmean(du_samples[:N])

# %%
x = np.arange(1, len(catalog), 100)
y = np.array([mean(N) for N in x])

# %%
fig, ax = plt.subplots(dpi=200)
ax.plot(x, y)
# %%
Nmeans = []
Ns = np.arange(1, len(catalog), 1_000)
Ns = np.append(Ns, len(catalog))

for N in Ns:
    means = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        dus = rng.choice(du_samples, N, replace=True)
        means.append(np.nanmean(dus))
    
    Nmeans.append(means)

Nmeans = np.array(Nmeans)

 # %%
m = Nmeans.mean(axis=1)
s = Nmeans.std(axis=1)
plt.plot(Ns, Nmeans.mean(axis=1), c="C0")
plt.fill_between(Ns, m-s, m+s, color="C0", alpha=0.25)
plt.axhline(0.20548664, c="C2")
plt.ylim(0.198, 0.21)

# %%
plt.plot(Ns, s)
plt.yscale("log")
plt.plot(Ns, s[-1] * np.sqrt(Ns[-1])/np.sqrt(Ns))
# %%
s[-1] * np.sqrt(Ns[-1])
# %%

Ns = np.arange(1, len(catalog), 1_000)
Ns = np.append(Ns, len(catalog))
Nseeds = 10
rng = np.random.default_rng(0)
Nmeans = []
for N in Ns:
    dus = rng.choice(du_samples, Nseeds * N, replace=True)
    M = np.nanmean(np.nanmean(dus.reshape(Nseeds, N, -1), axis=-1), axis=-1)
    Nmeans.append(M)

Nmeans = np.array(Nmeans)

# %%
dus.reshape(7, , -1).shape

# %%
dus.shape
# %%
np.arange(12).reshape(6, 2).reshape(3, 2, 2)
# %%
