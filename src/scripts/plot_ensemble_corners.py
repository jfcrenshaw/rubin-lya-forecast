"""Make corner plots for the various PZFlow ensembles."""
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from utils import load_truth_catalog, lya_decrement, paths, load_ensemble

# load the truth catalog
truth_catalog = load_truth_catalog()

truth_catalog.g = truth_catalog.g + lya_decrement(truth_catalog.redshift, "g", 0)


# define a function to draw cornerplots
def corner_plot(name: str, figsize: tuple, N: int = 10_000) -> sns.PairGrid:
    """Plot the losses saved in the file on the given axis.

    Parameters
    ----------
    name: str
        The name of the ensemble.
    figsize: tuple
        The size of the figure in inches.
    N: int, default=10_000
        The number of samples to draw for the contours.

    Returns
    -------
    sns.PairGrid
        The seaborn pairgrid.
    """
    # load the ensemble
    ensemble = load_ensemble(name)

    # get samples from the truth catalog
    truth_sample = truth_catalog[:N].copy()
    truth_sample["Set"] = "Truth"

    # get samples from the flow
    flow_sample = ensemble.sample(1, conditions=truth_sample, seed=0)
    flow_sample["Set"] = "Flow"

    # combine the samples into a single DataFrame
    sample = pd.concat([truth_sample, flow_sample], ignore_index=True)

    # build the list of rows and columns
    x_vars = ensemble.data_columns + ensemble.conditional_columns
    y_vars = ensemble.data_columns

    # make the corner plot
    grid = sns.pairplot(
        data=sample,
        hue="Set",
        palette={"Truth": "C4", "Flow": "C0"},
        kind="kde",
        x_vars=x_vars,
        y_vars=y_vars,
        diag_kind="hist",
        diag_kws={"element": "step"},
        plot_kws={"linewidths": 0.75, "alpha": 0.5},
    )
    grid._legend.remove()

    # set the figure size
    grid.fig.set_size_inches(*figsize)

    for ax in grid.axes.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))

    return grid


# path where ensembles are saved
model_dir = paths.data / "models"

# save the lsst corner plot
grid = corner_plot(model_dir / "lsstY10", (5, 1.5))
grid.fig.suptitle("LSST", y=1.1)
grid.savefig(paths.figures / "lsst_corner.pdf")

# save the lsst corner plot
grid = corner_plot(model_dir / "lsstY10+euclid", (7, 1.5))
grid.fig.suptitle("LSST + Euclid", y=1.1)
grid.savefig(paths.figures / "lsst+euclid_corner.pdf")

# save the lsst corner plot
grid = corner_plot(model_dir / "lsstY10+roman", (7, 1.5))
grid.fig.suptitle("LSST + Roman", y=1.1)
grid.savefig(paths.figures / "lsst+roman_corner.pdf")
