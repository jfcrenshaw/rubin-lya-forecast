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

"""
workflow.add_stage(
    "create observed catalogs",
    create_observed_catalogs,
    [
        paths.catalogs / "truth_catalog.parquet",
        paths.catalogs / "y1_catalog.parquet",
        paths.catalogs / "y5_catalog.parquet",
        paths.catalogs / "y10_catalog.parquet",
        paths.catalogs / "y10_euclid_catalog.parquet",
        paths.catalogs / "y10_roman_catalog.parquet",
    ],
    dependencies="emulate input catalog",
    cache=True,
    min_snr=10,
    seed=2,
)

workflow.add_stage(
    "train_ensembles",
    train_ensembles,
    [
        paths.models / "y1_ensemble.pkl",
        paths.models / "y1_ensemble_losses.pkl",
        paths.models / "y5_ensemble.pkl",
        paths.models / "y5_ensemble_losses.pkl",
        paths.models / "y10_ensemble.pkl",
        paths.models / "y10_ensemble_losses.pkl",
        paths.models / "y10_euclid_ensemble.pkl",
        paths.models / "y10_euclid_ensemble_losses.pkl",
        paths.models / "y10_roman_ensemble.pkl",
        paths.models / "y10_roman_ensemble_losses.pkl",
    ],
    dependencies="create observed catalogs",
    cache=True,
    n_flows=4,
    learning_rates=[1e-5, 2e-6, 1e-6],
    epochs=[400, 50, 50],
    seed=3,
)
"""

if __name__ == "__main__":
    # Run the command-line interface
    workflow.cli()
