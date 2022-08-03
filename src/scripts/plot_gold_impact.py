"""Make a plot that shows how Lya scattering impacts the gold sample."""
import matplotlib.pyplot as plt
import numpy as np
from showyourwork.paths import user as Paths
from utils import load_truth_catalog, lya_decrement
from utils.error_models import LSSTErrorModel

# instantiate the paths
paths = Paths()

# setup the redshift grid
zs = np.linspace(1.63, 3.5, 100)

# get the LSST gold sample
gold_sample = load_truth_catalog().query("i < 25.3")
# and calculate what fraction is above redshift 1.63
frac_gold = gold_sample.eval("redshift > 1.63").mean()

# calculate the mean errors as a function of redshift
mean_errs = {}  # type: ignore
zbins = np.arange(zs.min(), zs.max() + 0.05, 0.05)
for year in [1, 10]:
    # get errors for the gold sample
    gold_sample_w_errs = LSSTErrorModel(nYrObs=year)(gold_sample, seed=0)

    # create lists to hold the mean errors
    mean_errs[year] = {"u": [], "g": []}

    # loop over the redshift bins
    for i in range(len(zbins) - 1):
        # get the galaxies in the bin
        galaxies_in_bin = gold_sample_w_errs.query(
            f"(redshift >= {zbins[i]}) & (redshift < {zbins[i+1]})"
        )

        # get the mean u error
        errs = galaxies_in_bin["u_err"]
        errs = errs[np.isfinite(errs)]
        mean_errs[year]["u"].append(errs.mean())

        # get the mean g error
        errs = galaxies_in_bin["g_err"]
        errs = errs[np.isfinite(errs)]
        mean_errs[year]["g"].append(errs.mean())

# create the figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.3, 4), dpi=300)

# plot the redshift distribution at the top
ax1.hist(gold_sample.redshift, bins=20, density=True, color="C4")
ax1.set(xticks=[], xlim=(zs.min(), zs.max()), ylim=(0, 0.25), ylabel="$n(z)$")
ax1.text(
    0.95,
    0.9,
    f"{100 * frac_gold:.0f}% of gold sample",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=9,
)

# plot the u band data
ax2.plot(zs, lya_decrement(zs, "u"), c="C0", label="Decrement")
ax2.plot(zbins[:-1], mean_errs[1]["u"], c="C0", ls="--", label="Year 1 error")
ax2.plot(zbins[:-1], mean_errs[10]["u"], c="C0", ls=":", label="Year 10 error")
ax2.set(xticks=[], xlim=(zs.min(), zs.max()), ylabel="$\Delta u ~ / ~ \\bar{\sigma}_u$")
ax2.legend(fontsize=7)

# plot the g band data
ax3.plot(zs, lya_decrement(zs, "g"), c="C5", label="Decrement")
ax3.plot(zbins[:-1], mean_errs[1]["g"], c="C5", ls="--", label="Year 1 error")
ax3.plot(zbins[:-1], mean_errs[10]["g"], c="C5", ls=":", label="Year 10 error")
ax3.set(
    xlabel="Redshift",
    xlim=(zs.min(), zs.max()),
    ylabel="$\Delta g ~ / ~ \\bar{\sigma}_g$",
)
ax3.legend(fontsize=7)

fig.savefig(paths.figures / "gold_impact.pdf")
