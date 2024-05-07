"""Create catalogs for photo-z and delta-u inference."""

import jax
import numpy as np
import pandas as pd
from pzflow import Flow, FlowEnsemble

from src.utils import Stage, observe_catalog, split_seed

jax.config.update("jax_platform_name", "cpu")


class CreateInferenceCatalogs(Stage):
    """Stage to create catalogs with which to perform photo-z and du inference."""

    def _run(self) -> None:
        """Run the stage."""
        # Unpack the stage variables
        n_galaxies = self.stage_vars.n_galaxies
        n_samples_per_galaxy = self.stage_vars.n_samples_per_galaxy
        batch_size = self.stage_vars.batch_size

        # Determine the number of batches
        n_batches = np.ceil(n_galaxies / batch_size).astype(int)

        # Split seeds
        seed_sample, seed_obs, seed_inf = split_seed(self.stage_vars.seed, 3)
        seed_sample = split_seed(seed_sample, n_batches)

        # Generate a sample from the emulator
        emulator = Flow(file=self.paths.models / "incat_emulator.pkl")
        truth = []
        for batch in range(n_batches):
            truth.append(
                emulator.sample(
                    batch_size,
                    seed=seed_sample[batch],
                )
            )
        truth = pd.concat(truth, ignore_index=True).iloc[:n_galaxies]

        # Loop over catalogs to create
        for file in self.output:
            # Get the model name
            name = file.name.removesuffix("_inference_catalog.parquet")

            # Load the model
            ens = FlowEnsemble(file=self.paths.models / f"{name}_ensemble.pkl")

            # Add observational errors to the sample
            if "truth" in file.name:
                obs = truth
                obs = obs.rename(columns={"u": "u0"})
            else:
                year = int(name.split("_")[0][1:])
                euclid = "euclid" in name
                roman = "roman" in name
                obs = observe_catalog(
                    cat_truth=truth,
                    n_years=year,
                    min_snr=5,
                    euclid=euclid,
                    roman=roman,
                    seed=seed_obs,
                )

            # Perform inference on redshift and u band
            samples = pd.concat(
                [
                    ens.sample(
                        n_samples_per_galaxy,
                        conditions=obs.iloc[batch_idx : batch_idx + batch_size],
                        save_conditions=False,
                        seed=seed_inf,
                    )
                    for batch_idx in range(0, len(obs), batch_size)
                ]
            )

            # Save inference samples
            obs["redshift_samples"] = list(
                samples["redshift"].values.reshape(-1, n_samples_per_galaxy)
            )
            obs["u0_samples"] = list(
                samples["u0"].values.reshape(-1, n_samples_per_galaxy)
            )

            # Save the catalog of samples
            obs.to_parquet(file)
