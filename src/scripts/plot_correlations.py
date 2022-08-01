"""Plot the correlations."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# laod the correlations
with open(paths.data / "correlations.pkl", "rb") as file:
    correlations = pickle.load(file)

# create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.7), constrained_layout=True)
ax1.set(xscale="log", yscale="log")
ax2.set(xscale="log", yscale="log")
colors = {
    "FF": "C0",
    "Fg": "C5",
    "gg": "C4",
}

# plot the angular power spectra on the left
Cl = correlations["Cl"]
ax1.plot(Cl["ells"], Cl["FF"], label="$\\alpha \, \\alpha$", c=colors["FF"])
ax1.plot(Cl["ells"], np.abs(Cl["Fg"]), label="$\\alpha \, g$", c=colors["Fg"])
ax1.plot(Cl["ells"], Cl["gg"], label="$g \, g$", ls="--", c=colors["gg"])
ax1.set_xlabel("$\\ell$", fontsize=11)
ax1.set_ylabel("$\left|C_\\ell \\right|$", fontsize=11)

# plot the angular correlation functions on the right
w = correlations["w"]
ax2.plot(w["theta_deg"], w["FF"], label="$FF$", c=colors["FF"])
ax2.plot(w["theta_deg"], np.abs(w["Fg"]), label="$Fg$", c=colors["Fg"])
ax2.plot(w["theta_deg"], w["gg"], label="$gg$", ls="--", c=colors["gg"])
ax2.legend()
ax2.set_xlabel("$\\theta$", fontsize=11)
ax2.set_ylabel("$\left| w(\\theta) \\right|$", fontsize=11)

fig.savefig(paths.figures / "correlations.pdf")
