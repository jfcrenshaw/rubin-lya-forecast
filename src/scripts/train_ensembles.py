"""Train the PZFlow ensemble that models the simulated data."""
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds
from utils import paths

# create the directory where the models will be saved
Path.mkdir(paths.models, exist_ok=True)

# loop over the training catalogs
for file in paths.train.glob("*.pkl"):

    print("Training", file.stem)

    # get the training and test sets
    catalog = pd.read_pickle(file)
    train_set = catalog.iloc[:100_000].copy()
    test_set = catalog.iloc[100_000:].copy()

    # set the mins and maxs of the target distribution
    # will be used for the ShiftBounds bijector below
    buffer = 0.5
    mins = jnp.array([0, train_set.u.min() - buffer])
    maxs = jnp.array([4, train_set.u.max() + buffer])

    # get the conditional columns
    conditional_columns = list(train_set.columns)
    conditional_columns.remove("redshift")
    conditional_columns.remove("u")

    # setup the bijector chain
    bijector = Chain(
        ShiftBounds(mins, maxs),
        RollingSplineCoupling(nlayers=2, n_conditions=len(conditional_columns)),
    )

    # create the ensemble
    ensemble = FlowEnsemble(
        data_columns=["redshift", "u"],
        bijector=bijector,
        conditional_columns=conditional_columns,
        N=10,
    )

    # train the ensemble
    learning_rates = [1e-3, 2e-4, 1e-4, 2e-5]
    N_epochs = [40, 40, 40, 40]
    seeds = [1234, 4123, 3412, 2341]
    losses = [
        ensemble.train(
            train_set,
            optimizer=optax.adam(lr),
            epochs=epochs,
            seed=seed,
            verbose=True,
        )
        for lr, epochs, seed in zip(learning_rates, N_epochs, seeds)
    ]

    # repackage the losses from each stage of training so that each losses
    # is a dict of flow_name: all_losses
    combined_losses = {
        fname: [  # for each flow trained in the ensemble...
            float(loss)  # save the list of training losses
            for lossDict in losses
            for loss in lossDict[fname]
        ]
        for fname in losses[0]
    }

    # print the train and test loss
    train_loss = -np.mean(ensemble.log_prob(train_set))
    test_loss = -np.mean(ensemble.log_prob(test_set))
    print(f"train = {train_loss:.3f}    test = {test_loss:.3f}")

    # save the ensemble
    ensemble.save(paths.models / f"{file.stem}_ensemble.pzflow.pkl")

    # and the losses
    with open(paths.models / f"{file.stem}_ensemble_losses.pkl", "wb") as file:
        pickle.dump(losses, file)
