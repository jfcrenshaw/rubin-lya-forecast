"""Plot the training losses for the various PZFlow ensembles."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from utils import paths


# define a function for plotting the losses
def plot_losses(filename: str, ax: plt.Axes, name: str) -> None:
    """Plot the losses saved in the file on the given axis.

    Parameters
    ----------
    filename: str
        The path the the file containing the losses.
    ax: plt.Axes
        The matplotlib axis on which to plot the losses.
    name: str
        The name to print in the top right of the plot
    """
    # open the losses
    with open(filename, "rb") as file:
        losses = pickle.load(file)

    # combine the losses across training iterations
    losses = {
        flow: np.array([np.array(loss[flow]) for loss in losses]).flatten()
        for flow in losses[0]
    }

    # plot the losses
    for loss in losses.values():
        ax.plot(loss, c="C0", alpha=0.5)

    # print the name
    ax.text(0.92, 0.92, name, transform=ax.transAxes, va="top", ha="right")

    # set the x label
    ax.set(xlabel="Epoch")


# create the figure
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(6.75, 2), constrained_layout=True, dpi=200, sharey=True
)

# plot the losses
model_dir = paths.data / "models"
plot_losses(model_dir / "lsstY10_ensemble_losses.pkl", ax1, "LSST")
plot_losses(model_dir / "lsstY10+euclid_ensemble_losses.pkl", ax2, "LSST + Euclid")
plot_losses(model_dir / "lsstY10+roman_ensemble_losses.pkl", ax3, "LSST + Roman")

# set the y label on the leftmost panel
ax1.set(ylabel="Training loss")

# save the figure
fig.savefig(paths.figures / "ensemble_losses.pdf")
