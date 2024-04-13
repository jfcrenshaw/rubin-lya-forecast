"""Run all functions for analysis."""

# import jax

from src import *
from src.utils import Workflow

# jax.config.update("jax_platform_name", "cpu")

# Create the workflow
workflow = Workflow()

# Set the cache tag
workflow.cache_tag = "v2"

# Define the paths where I will save things
paths = workflow.paths
paths.data = paths.root / "data"
paths.inputs = paths.data / "inputs"
paths.catalogs = paths.data / "catalogs"
paths.models = paths.root / "models"
paths.figures = paths.root / "figures"

# Add stages to workflow
workflow.add_stage(
    "download data",
    None,
    [
        paths.inputs / "u_bandpass.dat",
        paths.inputs / "g_bandpass.dat",
        paths.inputs / "r_bandpass.dat",
        paths.inputs / "input_catalog.parquet",
    ],
    cache=True,
)

workflow.add_stage(
    "plot increments",
    PlotIncrements,
    paths.figures / "increments.pdf",
)

workflow.add_stage(
    "emulate input catalog",
    EumlateInputCatalog,
    [
        paths.models / "incat_emulator.pkl",
        paths.models / "incat_emulator_losses.pkl",
    ],
    cache=True,
    seed=1,
)

workflow.add_stage(
    "train ensembles",
    TrainEnsembles,
    [
        paths.models / f"{cat}_{end}"
        for cat in ["y1", "y5", "y10", "y10_roman", "y10_euclid"]
        for end in ["ensemble.pkl", "ensemble_losses.pkl"]
    ],
    dependencies="emulate input catalog",
    cache=True,
    n_flows=4,
    learning_rates=[5e-6, 1e-6, 5e-7],
    epochs=[400, 50, 50],
    seed=2,
)


if __name__ == "__main__":
    # Run the command-line interface
    workflow.cli()
