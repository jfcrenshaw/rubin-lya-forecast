"""Plot sigma_du as a function of the catalog."""
import pickle

import matplotlib.pyplot as plt
from utils import paths, plot_metric

# load the saved sigma_du's
with open(paths.data / "sigma_du.pkl", "rb") as file:
    sigma_du = pickle.load(file)

# plot the metrics
fig, ax = plt.subplots(figsize=(3.3, 1.3), constrained_layout=True)
plot_metric(sigma_du, ax)
ax.set(ylabel="$\sigma_{\Delta u}$")
fig.savefig(paths.figures / "sigma_du.pdf")
