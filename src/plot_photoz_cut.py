"""Plot the purity and completeness curves for the photo-z cut."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import Stage


class PlotPhotoZCut(Stage):
    """Stage to plot the purity and completeness curves for photo-z cuts."""

    def _run(self) -> None:
        """Run the stage."""
        # Unpack the stage variables
        zmin = self.stage_vars.zmin
        zmax = self.stage_vars.zmax

        # Create the figure
        fig, (ax1, ax2, ax3) = plt.subplots(
            *(3, 1),
            figsize=(3.5, 7),
            constrained_layout=True,
        )
        ax1.set(ylabel="purity", xticks=[], xlim=(0, 1), ylim=(0, 1))
        ax2.set(ylabel="completeness", xticks=[], xlim=(0, 1), ylim=(0, 1))
        ax3.set(
            ylabel="purity $\sqrt{\mathrm{completeness}}$",
            xlabel="$p_\mathrm{cut}$",
            xlim=(0, 1),
            ylim=(0, 1),
        )

        # Loop over the inference catalogs
        for file in list(self.paths.catalogs.glob("*_inference_catalog.parquet")):
            # Load the catalog
            cat = pd.read_parquet(file)

            # Pull out info about redshift inference
            z_true = cat["redshift"].values
            z_samples = np.array(cat["redshift_samples"].tolist())

            # Loop over cuts to calculate purity and completeness
            purity = []
            completeness = []
            pcut_range = np.arange(0, 1, 0.02)
            for pcut in pcut_range:
                # Determine which galaxies are in range, and which are selected
                tru = (z_true >= zmin) & (z_true <= zmax)
                sel = np.mean((z_samples >= zmin) & (z_samples <= zmax), axis=1) >= pcut

                # Append purity and completeness
                purity.append(tru[sel].mean())
                completeness.append(sel[tru].mean())

            # Plot the values
            name = file.name.removesuffix("_inference_catalog.parquet")
            name = name.replace("_", "+")
            ax1.plot(pcut_range, purity)
            ax2.plot(pcut_range, completeness, label=name)
            ax3.plot(pcut_range, purity * np.sqrt(completeness))

        ax2.legend()

        # Save the figure
        fig.savefig(self.output)
