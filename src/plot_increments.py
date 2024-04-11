"""Plot the Lyman-alpha band increments."""

import matplotlib.pyplot as plt
import numpy as np

from .utils import ExtinctionCalculator, Stage


class PlotIncrements(Stage):
    """Stage to plot Lyman-alpha band increments."""

    def _run(self) -> None:
        """Run the stage."""
        # Create the extinction calculator
        ec = ExtinctionCalculator()

        # Set the redshift grid
        zs = np.linspace(1.6, 2.4, 1000)

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.7), constrained_layout=True)

        # Plot the u band decrements as a function of spectral index
        ax1.plot(
            zs,
            ec.lya_increment(zs, "u", 0),
            label="$\\alpha = 0$",
            c="C0",
            ls=":",
        )
        ax1.plot(
            zs,
            ec.lya_increment(zs, "u", -2),
            label="$\\alpha =  -2$",
            c="C0",
            ls="-",
        )
        ax1.plot(
            zs,
            ec.lya_increment(zs, "u", -4),
            label="$\\alpha = -4$",
            c="C0",
            ls="--",
        )
        ax1.legend()
        ax1.set(
            xlabel="redshift",
            ylabel="$\Delta u$ [mags]",
            xlim=(zs.min(), zs.max()),
        )

        # Plot the u band decrements for the 3 relevant bands
        zs = np.linspace(1.5, 5, 1000)
        ax2.plot(zs, ec.lya_increment(zs, "u"), label="$\Delta u$", c="C0")
        ax2.plot(zs, ec.lya_increment(zs, "g"), label="$\Delta g$", c="C5")
        ax2.plot(zs, ec.lya_increment(zs, "r"), label="$\Delta r$", c="C4")
        ax2.legend()
        ax2.set(
            xlabel="redshift",
            ylabel="Lyman-$\\alpha$ increment [mags]",
            xlim=(zs.min(), zs.max()),
        )

        fig.savefig(self.output)
