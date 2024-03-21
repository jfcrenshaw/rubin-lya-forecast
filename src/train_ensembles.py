"""Train the ensembles for photo-z and u0 estimation."""

import pickle

import jax.numpy as jnp
import optax
import pandas as pd
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds

from .utils import paths, split_catalog, split_seed


def train_ensembles(
    output: list,
    n_flows: int,
    learning_rates: list,
    epochs: list,
    seed: int,
) -> None:
    """Train the ensembles for photo-z and u0 estimation.

    Parameters
    ----------
    output : list
        The list of FlowEnsemble and loss files. Corresponding ensemble and
        losses should be consecutive in list, with ensemble first.
    test_frac : float
        The fraction of galaxies to hold out for testing.
    n_flows : int
        The number of flows in each ensemble.
    learning_rates : list
        A list of learning rates for the optax.adam optimizer.
    epochs : int
        The number of epochs to train at each learning rate. Must be same
        length as learning_rates.
    seed : int
        A seed to use for train/test split and then for training.
    """
    # Split the output files
    ensemble_files = output[::2]
    loss_files = output[1::2]

    # Create a list of seeds
    train_seeds = split_seed(seed, len(learning_rates))

    # Loop over each catalog and train an ensemble
    for ens_file, loss_file in zip(ensemble_files, loss_files):
        # Get the corresponding catalog name
        cat_name = ens_file.name.split("_ensemble")[0] + "_catalog.parquet"
        catalog = pd.read_parquet(paths.catalogs / cat_name)

        # Get column names for pzflow
        data_columns = ["redshift", "u0"]
        conditional_columns = catalog.columns.drop(data_columns + ["u", "u_err"])
        conditional_columns = conditional_columns.drop(
            [col for col in conditional_columns if "_err" in col]
        )

        # Determine range for the data columns
        mins = jnp.array([0, catalog.u0.min()])
        maxs = jnp.array([4, catalog.u0.max()])

        # Create the bijector
        bijector = Chain(
            ShiftBounds(mins, maxs, B=4),
            RollingSplineCoupling(nlayers=2, n_conditions=len(conditional_columns)),
        )

        # Create the ensemble
        ensemble = FlowEnsemble(
            data_columns=data_columns,
            conditional_columns=conditional_columns,
            bijector=bijector,
            N=n_flows,
        )

        # Create training and test sets
        train, _ = split_catalog(catalog)

        # Train the ensembles
        losses = [
            ensemble.train(
                train,
                optimizer=optax.adam(lr),
                epochs=ep,
                seed=seed,
                verbose=True,
            )
            for lr, ep, seed in zip(learning_rates, epochs, train_seeds)
        ]

        # Repackage losses from each stage of training so each
        # is a dict of flow_name: all_losses
        combined_losses = {
            fname: [  # For each flow trained in the ensemble...
                float(loss)  # Save the list of training losses
                for lossDict in losses
                for loss in lossDict[fname]
            ]
            for fname in losses[0]
        }

        # Save the ensemble and losses
        ensemble.save(ens_file)
        with open(loss_file, "wb") as file:
            pickle.dump(combined_losses, file)
