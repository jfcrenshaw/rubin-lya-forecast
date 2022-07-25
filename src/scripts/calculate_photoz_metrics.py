"""Plot metrics for the photo-z cuts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from showyourwork.paths import user as Paths
from utils import load_truth_catalog

# instantiate the paths
paths = Paths()

# directories where the catalogs are saved
catalog_dir = paths.data / "observed_catalogs"
bg_dir = paths.data / "background_catalogs"
fg_dir = paths.data / "foreground_catalogs"


# loop over every observed catalog
bg_purity = {}
bg_completeness = {}
bg_size = {}
fg_purity = {}
fg_completeness = {}
fg_size = {}
for file in catalog_dir.glob("*.pkl"):
    # load the catalogs
    obs_catalog = pd.read_pickle(file)
    bg_catalog = pd.read_pickle(bg_dir / Path(file.stem + "_bg.pkl"))
    fg_catalog = pd.read_pickle(fg_dir / Path(file.stem + "_fg.pkl"))

    # background metrics
    bg_true = obs_catalog.query("redshift > 2.36").index
    bg_pred = bg_catalog.index
    _bg_completeness = np.isin(bg_true, bg_pred, assume_unique=True).mean()
    _bg_purity = np.isin(bg_pred, bg_true, assume_unique=True).mean()
    _bg_size = bg_pred.size

    # foreground metrics
    fg_true = obs_catalog.query("(redshift > 1.63) & (redshift < 2.36)").index
    fg_pred = fg_catalog.index
    _fg_completeness = np.isin(fg_true, fg_pred, assume_unique=True).mean()
    _fg_purity = np.isin(fg_pred, fg_true, assume_unique=True).mean()
    _fg_size = fg_pred.size

    # save the metrics for plotting
    if "euclid" in file.stem:
        bg_euclid_purity = _bg_purity
        bg_euclid_completeness = _bg_completeness
        bg_euclid_size = _bg_size
        fg_euclid_purity = _fg_purity
        fg_euclid_completeness = _fg_completeness
        fg_euclid_size = _fg_size
    elif "roman" in file.stem:
        bg_roman_purity = _bg_purity
        bg_roman_completeness = _bg_completeness
        bg_roman_size = _bg_size
        fg_roman_purity = _fg_purity
        fg_roman_completeness = _fg_completeness
        fg_roman_size = _fg_size
    else:
        year = int(file.stem[5:])
        bg_purity[year] = _bg_purity
        bg_completeness[year] = _bg_completeness
        bg_size[year] = _bg_size
        fg_purity[year] = _fg_purity
        fg_completeness[year] = _fg_completeness
        fg_size[year] = _fg_size

# sort the metrics by year
bg_purity = dict(sorted(bg_purity.items()))
bg_completeness = dict(sorted(bg_completeness.items()))
fg_purity = dict(sorted(fg_purity.items()))
fg_completeness = dict(sorted(fg_completeness.items()))


# save a bunch of numbers for reference in the paper
open(paths.output / "bg_purity_y1.txt", "w").write(f"{bg_purity[1]:.2f}")
open(paths.output / "bg_purity_y10.txt", "w").write(f"{bg_purity[10]:.2f}")
open(paths.output / "bg_purity_y10+euclid.txt", "w").write(f"{bg_euclid_purity:.2f}")
open(paths.output / "bg_purity_y10+roman.txt", "w").write(f"{bg_roman_purity:.2f}")

open(paths.output / "bg_completeness_y1.txt", "w").write(f"{bg_completeness[1]:.2f}")
open(paths.output / "bg_completeness_y10.txt", "w").write(f"{bg_completeness[10]:.2f}")
open(paths.output / "bg_completeness_y10+euclid.txt", "w").write(
    f"{bg_euclid_completeness:.2f}"
)
open(paths.output / "bg_completeness_y10+roman.txt", "w").write(
    f"{bg_roman_completeness:.2f}"
)

open(paths.output / "fg_purity_y1.txt", "w").write(f"{fg_purity[1]:.2f}")
open(paths.output / "fg_purity_y10.txt", "w").write(f"{fg_purity[10]:.2f}")
open(paths.output / "fg_purity_y10+euclid.txt", "w").write(f"{fg_euclid_purity:.2f}")
open(paths.output / "fg_purity_y10+roman.txt", "w").write(f"{fg_roman_purity:.2f}")

open(paths.output / "fg_completeness_y1.txt", "w").write(f"{fg_completeness[1]:.2f}")
open(paths.output / "fg_completeness_y10.txt", "w").write(f"{fg_completeness[10]:.2f}")
open(paths.output / "fg_completeness_y10+euclid.txt", "w").write(
    f"{fg_euclid_completeness:.2f}"
)
open(paths.output / "fg_completeness_y10+roman.txt", "w").write(
    f"{fg_roman_completeness:.2f}"
)


# plot the background metrics
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    CCCD
    """,
    figsize=(3.3, 2.6),
)
plt.subplots_adjust(wspace=0)

# set axis limits, labels, etc.
axes["A"].set(xlim=(0.5, 11), xticks=[], ylabel="Purity")
axes["B"].set(xlim=(0.2, 2.5), xticks=[], yticks=[])
axes["C"].set(
    xlim=(0.5, 11),
    xticks=[2, 4, 6, 8, 10],
    xlabel="Survey Duration",
    ylabel="Completeness",
)
axes["D"].set(
    xlim=(0.2, 2.5),
    xticks=[1, 2],
    yticks=[],
)
axes["D"].set_xticklabels(
    ["$+\!$ Euclid", "$+\!$ Roman"], rotation=-45, ha="left", rotation_mode="anchor"
)
axes["A"].spines.right.set_visible(False)
axes["B"].spines.left.set_linestyle((0, (4, 4)))
axes["C"].spines.right.set_visible(False)
axes["D"].spines.left.set_linestyle((0, (4, 4)))

# plot purity
axes["A"].plot(bg_purity.keys(), bg_purity.values())
axes["B"].scatter([1, 2], [bg_euclid_purity, bg_roman_purity], c="C3", marker="x", lw=1)
axes["A"].set(ylim=(0.96, 1.005))
axes["B"].set(ylim=(0.96, 1.005))

# plot completeness
axes["C"].plot(bg_completeness.keys(), bg_completeness.values())
axes["D"].scatter(
    [1, 2], [bg_euclid_completeness, bg_roman_completeness], c="C3", marker="x", lw=1
)
axes["C"].set(ylim=(0, 0.6))
axes["D"].set(ylim=(0, 0.6))

fig.savefig(paths.figures / "bg_photoz_metrics.pdf", bbox_inches="tight")


# plot the foreground metrics
fig, axes = plt.subplot_mosaic(
    """
    AAAB
    CCCD
    """,
    figsize=(3.3, 2.6),
)
plt.subplots_adjust(wspace=0)

# set axis limits, labels, etc.
axes["A"].set(xlim=(0.5, 11), xticks=[], ylabel="Purity")
axes["B"].set(xlim=(0.2, 2.5), xticks=[], yticks=[])
axes["C"].set(
    xlim=(0.5, 11),
    xticks=[2, 4, 6, 8, 10],
    xlabel="Survey Duration",
    ylabel="Completeness",
)
axes["D"].set(
    xlim=(0.2, 2.5),
    xticks=[1, 2],
    yticks=[],
)
axes["D"].set_xticklabels(
    ["$+\!$ Euclid", "$+\!$ Roman"], rotation=-45, ha="left", rotation_mode="anchor"
)
axes["A"].spines.right.set_visible(False)
axes["B"].spines.left.set_linestyle((0, (4, 4)))
axes["C"].spines.right.set_visible(False)
axes["D"].spines.left.set_linestyle((0, (4, 4)))

# plot purity
axes["A"].plot(fg_purity.keys(), fg_purity.values())
axes["B"].scatter([1, 2], [fg_euclid_purity, fg_roman_purity], c="C3", marker="x", lw=1)
axes["A"].set(ylim=(0.92, 1.005))
axes["A"].set(ylim=(0.92, 1.005))

# plot completeness
axes["C"].plot(fg_completeness.keys(), fg_completeness.values())
axes["D"].scatter(
    [1, 2], [fg_euclid_completeness, fg_roman_completeness], c="C3", marker="x", lw=1
)
axes["C"].set(ylim=(0, 0.65))
axes["D"].set(ylim=(0, 0.65))

fig.savefig(paths.figures / "fg_photoz_metrics.pdf", bbox_inches="tight")


# we also want to calculate the expected sizes of the catalogs
# we can determine the overall scale factor by comparing the predicted
# size of the gold sample, and the size of the gold sample in the simulation
lsst_scale = 4e9 / load_truth_catalog().query("i < 25.3").shape[0]

open(paths.output / "bg_size_y1.txt", "w").write(f"{bg_size[1] * lsst_scale / 1e6:.0f}")
open(paths.output / "bg_size_y10.txt", "w").write(
    f"{bg_size[10] * lsst_scale / 1e6:.0f}"
)

open(paths.output / "fg_size_y1.txt", "w").write(f"{fg_size[1] * lsst_scale / 1e6:.0f}")
open(paths.output / "fg_size_y10.txt", "w").write(
    f"{fg_size[10] * lsst_scale / 1e6:.0f}"
)

# the sizes of Euclid and Roman are more complex...
# we need to rescale by the footprint of those surveys
# we can get these numbers from Graham et al. 2020
f_euclid = 8_000 / 18_000  # arXiv:2108.01201
f_roman = 2_200 / 18_000  # arXiv:1503.03757

bg_size_all = lsst_scale * (
    f_euclid * bg_euclid_size
    + f_roman * bg_roman_size
    + (1 - f_euclid - f_roman) * bg_size[10]
)
open(paths.output / "bg_size_y10+euclid+roman.txt", "w").write(
    f"{bg_size_all / 1e6:.0f}"
)

fg_size_all = lsst_scale * (
    f_euclid * fg_euclid_size
    + f_roman * fg_roman_size
    + (1 - f_euclid - f_roman) * fg_size[10]
)
open(paths.output / "fg_size_y10+euclid+roman.txt", "w").write(
    f"{fg_size_all / 1e6:.0f}"
)
