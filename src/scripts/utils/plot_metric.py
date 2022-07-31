"""Define a function to plot metrics as a function of survey duration."""
import matplotlib.pyplot as plt


# define a function to plot metrics
def plot_metric(
    metric_dict: dict,
    ax: plt.Axes,
    xlabel_height: float = -0.3,
    include_perfect: bool = False,
) -> None:
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
    include_perfect: bool, default=False
        Whether to include the metric for the perfect catalog.
    """
    # pull out the metrics
    md = metric_dict.copy()
    euclid_metric = md.pop("euclid", None)
    roman_metric = md.pop("roman", None)
    perfect_metric = md.pop("perfect", None)
    lsst_metrics = dict(sorted(md.items()))

    # setup the axis
    lsst_ticks = [1, 5, 10]
    if include_perfect:
        xlim = (0, 17)
        other_ticks = [12.15, 13.9, 16]
        other_labels = ["+Euclid", "+Roman", "Perfect"]
        other_metrics = [euclid_metric, roman_metric, perfect_metric]
        ax.axvline(15, c="k", lw=1, ls="--")
    else:
        xlim = (0, 15)
        other_ticks = [12.15, 13.9]
        other_labels = ["+Euclid", "+Roman"]
        other_metrics = [euclid_metric, roman_metric]
    ax.set(
        xlabel="LSST Duration [years]",
        xlim=xlim,
        xticks=lsst_ticks + other_ticks,  # type: ignore
        xticklabels=lsst_ticks + other_labels,  # type: ignore
    )
    ax.xaxis.set_label_coords(5.5, xlabel_height, transform=ax.get_xaxis_transform())
    for tick in ax.get_xticklabels()[-len(other_labels) :]:
        tick.set_rotation(-45)
        tick.set_ha("left")
        tick.set_rotation_mode("anchor")
    ax.axvline(11, c="k", lw=1, ls="--")

    # plot the metrics
    ax.plot(lsst_metrics.keys(), lsst_metrics.values(), c="C0")
    ax.scatter(
        other_ticks,
        other_metrics,
        c="C5",
        marker="x",
        lw=1,
    )
