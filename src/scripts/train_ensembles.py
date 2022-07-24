"""Train the PZFlow ensemble that models the simulated data."""
import pickle
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import optax
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds
from showyourwork.paths import user as Paths
from utils import load_truth_catalog, lya_decrement

# instantiate the paths
paths = Paths()

# load the truth catalog
truth_catalog = load_truth_catalog()

# get a random subset for training and testing
train_size = 100_000
test_size = 20_000
sample = truth_catalog.sample(train_size + test_size, random_state=0)

# add Lya decrements to the g band
dg = lya_decrement(sample.redshift, "g", 0)
sample.loc["g"] = sample.g + dg

# split off the train and test sets
train_set = sample.iloc[:train_size].copy()
test_set = sample.iloc[train_size:].copy()

# set the mins and maxs of the target distribution
# will be used for the ShiftBounds bijector below
buffer = 0.5
mins = jnp.array([0, train_set.u.min() - buffer])
maxs = jnp.array([3.5, train_set.u.max() + buffer])


# define a function to train a FlowEnsemble
def train_ensemble(conditional_columns: list) -> Tuple[FlowEnsemble, dict]:
    """Train a FlowEnsemble to model z, u conditioned on the conditional columns.

    Parameters
    ----------
    conditional_columns: list
        List of columns on which to condition the distribution.

    Returns
    -------
    FlowEnsemble
        The PZFlow FlowEnsemble.
    dict
        The dictionary of training losses
    """

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
    learning_rates = [1e-3, 2e-4, 1e-4]
    N_epochs = [40, 40, 20]
    seeds = [123, 312, 231]
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

    return ensemble, combined_losses


# create the directory where the models will be saved
model_dir = paths.data / "models"
Path.mkdir(model_dir, exist_ok=True)

# train the LSST model
print("Training the LSST ensemble...")
ensemble, losses = train_ensemble(list("grizy"))
ensemble.save(model_dir / "lsst_ensemble.pzflow.pkl")
with open(model_dir / "lsst_ensemble_losses.pkl", "wb") as file:
    pickle.dump(losses, file)

# train the LSST+Euclid model
print("Training the LSST+Euclid ensemble...")
ensemble, losses = train_ensemble(list("grizyYJH"))
ensemble.save(model_dir / "lsst+euclid_ensemble.pzflow.pkl")
with open(model_dir / "lsst+euclid_ensemble_losses.pkl", "wb") as file:
    pickle.dump(losses, file)

# train the LSST+Roman model
print("Training the LSST+Roman ensemble...")
ensemble, losses = train_ensemble(list("grizyYJHF"))
ensemble.save(model_dir / "lsst+roman_ensemble.pzflow.pkl")
with open(model_dir / "lsst+roman_ensemble_losses.pkl", "wb") as file:
    pickle.dump(losses, file)
