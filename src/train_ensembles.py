"""Train the ensembles for photo-z and u0 estimation. """

import pickle

import jax.numpy as jnp
import optax
from pzflow import Flow, FlowEnsemble
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import CentBeta13

from src.utils import Stage, observe_catalog, split_seed


class TrainEnsembles(Stage):
    """Stage to train FlowEnsembles for photo-z and u0 estimation."""

    def _run(self) -> None:
        """Run the stage."""
        # Split the output files
        ensemble_files = self.output[::2]
        loss_files = self.output[1::2]

        # Unpack the stage variables
        n_flows = self.stage_vars.n_flows
        learning_rates = self.stage_vars.learning_rates
        epochs = self.stage_vars.epochs
        seed = self.stage_vars.seed

        # Split seed
        seed1, seed2 = split_seed(seed, 2)
        cat_seeds = split_seed(seed1, len(ensemble_files))
        train_seeds = split_seed(seed2, len(learning_rates))

        # Emulate a truth training catalog
        emulator = Flow(file=self.paths.models / "incat_emulator.pkl")
        truth = emulator.sample(100_000, seed=seed)

        for i, (ens_file, loss_file) in enumerate(zip(ensemble_files, loss_files)):
            # Get the training catalog and the column names
            if "truth" in ens_file.name:
                train = truth.rename(columns={"u": "u0"})
                conditional_columns = train.columns
            else:
                year = int(ens_file.name.split("_")[0][1:])
                euclid = "euclid" in ens_file.name
                roman = "roman" in ens_file.name
                train = observe_catalog(
                    cat_truth=truth,
                    n_years=year,
                    min_snr=5,
                    euclid=euclid,
                    roman=roman,
                    seed=cat_seeds[i],
                )
                conditional_columns = train.columns.drop(
                    [col for col in train.columns if "_err" in col or col == "u"]
                )

            data_columns = ["redshift", "u0"]
            conditional_columns = conditional_columns.drop(data_columns)

            # Determine range for the data columns
            mins = jnp.array([0, train.u0.min() - 0.5])
            maxs = jnp.array([4, train.u0.max() + 0.5])

            # Create the bijector
            bijector = Chain(
                ShiftBounds(mins, maxs),
                RollingSplineCoupling(nlayers=2, n_conditions=len(conditional_columns)),
            )

            # Create the ensemble
            ensemble = FlowEnsemble(
                data_columns=data_columns,
                conditional_columns=conditional_columns,
                bijector=bijector,
                latent=CentBeta13(len(data_columns)),
                N=n_flows,
            )

            # Train the ensembles
            print(f"Training {ens_file.name}")
            losses = [
                ensemble.train(
                    train,
                    optimizer=optax.adam(lr),
                    epochs=ep,
                    seed=ts,
                    verbose=True,
                )
                for lr, ep, ts in zip(learning_rates, epochs, train_seeds)
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
