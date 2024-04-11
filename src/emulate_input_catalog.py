"""Train a flow to emulate the input catalog."""

import pickle

import jax.numpy as jnp
import pandas as pd
from photerr import LsstErrorModel
from pzflow import Flow
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds
from src.utils import Stage, split_seed


class EumlateInputCatalog(Stage):
    """Stage to emulate the input catalog."""

    def _run(self) -> None:
        """Run the stage."""
        # Unpack the output files
        file_flow, file_losses = self.output

        # Get the random seeds
        cat_seed, train_seed = split_seed(self.stage_vars.seed)

        # Load the input catalog
        cat_input = pd.read_parquet(self.paths.inputs / "input_catalog.parquet")

        # Drop the absolute magnitude columns
        cat_input = cat_input.drop([col for col in cat_input if "_abs" in col], axis=1)

        # Cut to the 10 year 5-sigma limit in i band
        cat_input = cat_input[
            cat_input.i <= LsstErrorModel().getLimitingMags(nSigma=5)["i"]
        ]

        # Drop the few galaxies that have magnitudes <19.5 & >34 in any band
        band_columns = cat_input.columns[1:]
        cat_input = cat_input[~(cat_input[band_columns] < 19.5).any(axis=1)]
        cat_input = cat_input[~(cat_input[band_columns] > 34).any(axis=1)]

        # Downsample to 100,000 galaxies for training
        train_set = cat_input.sample(100_000, random_state=cat_seed)
        train_set.to_parquet(self.paths.catalogs / "emulator_training_catalog.parquet")

        # Set the min and max for each column
        mins = jnp.array(train_set.min(axis=0)) - 0.5
        maxs = jnp.array(train_set.max(axis=0)) + 0.5

        # Manually set range for redshifts
        mins = mins.at[0].set(0)
        maxs = maxs.at[0].set(4)

        # Create the bijector
        bijector = Chain(
            ShiftBounds(mins, maxs),
            RollingSplineCoupling(train_set.shape[1]),
        )

        # Create the flow
        flow = Flow(
            data_columns=train_set.columns,
            bijector=bijector,
        )

        # Train the flow
        losses = flow.train(train_set, epochs=200, seed=train_seed, verbose=True)

        # Save the flow and the training losses
        flow.save(file_flow)
        with open(file_losses, "wb") as file:
            pickle.dump(losses, file)
