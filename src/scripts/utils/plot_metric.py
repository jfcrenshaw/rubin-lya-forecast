"""Define a function to plot metrics as a function of survey duration."""
from typing import Any, List

import matplotlib.pyplot as plt


# define a function to plot metrics
def plot_metric(metric_dict: dict, ax: plt.Axes, xlabel_height: float = -0.3) -> None:
    """Plot the metric as a function of survey duration.

    Parameters
    ----------
    metric_dict: dict
        The dictionary of metric values. Can contain integers corresponding to
        years of LSST, plus "euclid" and "roman", corresponding to metrics for
        LSST year 10 + photometry from these surveys.
    ax: plt.Axes
        The matplotlib axis on which to plot the data.
    xlabel_height: float, default=-0.3
        The height at which to put the xlabel. Value in the axes coordinate system.
    """
    # setup the axis
    xlim = (0, 15)
    lsst_ticks: List[Any] = list(range(2, 12, 2))
    nir_ticks: List[Any] = [12.15, 13.9]
    ax.set(
        xlabel="LSST Duration [years]",
        xlim=xlim,
        xticks=lsst_ticks + nir_ticks,
        xticklabels=lsst_ticks + ["+Euclid", "+Roman"],
    )
    ax.xaxis.set_label_coords(5.5, xlabel_height, transform=ax.get_xaxis_transform())
    for tick in ax.get_xticklabels()[-2:]:
        tick.set_rotation(-45)
        tick.set_ha("left")
        tick.set_rotation_mode("anchor")
    ax.axvline(11, c="k", lw=1, ls="--")

    # pull out the metrics
    md = metric_dict.copy()
    euclid_metric = md.pop("euclid", None)
    roman_metric = md.pop("roman", None)
    lsst_metrics = dict(sorted(md.items()))

    # plot the metrics
    ax.plot(lsst_metrics.keys(), lsst_metrics.values(), c="C0")
    ax.scatter(nir_ticks, [euclid_metric, roman_metric], c="C5", marker="x", lw=1)
