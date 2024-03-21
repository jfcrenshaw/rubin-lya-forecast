"""Train a flow to emulate the input catalog."""

import pickle

import jax.numpy as jnp
import pandas as pd
from photerr import LsstErrorModel
from pzflow import Flow
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds

from .utils import paths


def emulate_input_catalog(output: list, seed: int):
    """Emulate the input catalog.

    Parameters
    ----------
    output : list
        List of paths at which to save (1) the trained flow and (2)
        the training losses (in that order)
    seed : int
        The random seed for training sample selection
    """
    # Unpack the output files
    file_flow, file_losses = output

    # Load the truth catalog
    cat_truth = pd.read_parquet(paths.inputs / "input_catalog.parquet")

    # Drop the absolute magnitude columns
    cat_truth = cat_truth.iloc[:, :-2]

    # Cut to the 10 year 5-sigma limit in i band
    cat_truth = cat_truth[
        cat_truth.i <= LsstErrorModel().getLimitingMags(nSigma=5)["i"]
    ]

    # Drop the few galaxies that have magnitudes <19.5 & >34 in any band
    band_columns = cat_truth.columns[1:]
    cat_truth = cat_truth[~(cat_truth[band_columns] < 19.5).any(axis=1)]
    cat_truth = cat_truth[~(cat_truth[band_columns] > 34).any(axis=1)]

    # Downsample to 100,000 galaxies for training
    train_set = cat_truth.sample(100_000, random_state=seed)
    train_set.to_parquet(paths.catalogs / "training_cat_for_input.parquet")

    # Set the min and max for each column
    mins = jnp.array(train_set.min(axis=0)).at[0].set(0)
    maxs = jnp.array(train_set.max(axis=0))

    # Create the bijector
    bijector = Chain(
        ShiftBounds(mins, maxs, B=4),
        RollingSplineCoupling(train_set.shape[1], B=5),
    )

    # Create the flow
    flow = Flow(
        data_columns=train_set.columns,
        bijector=bijector,
    )

    # Train the flow
    losses = flow.train(train_set, epochs=200, verbose=True)

    # Save the flow and the training losses
    flow.save(file_flow)
    with open(file_losses, "wb") as file:
        pickle.dump(losses, file)
