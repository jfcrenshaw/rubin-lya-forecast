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

# Define the models I will train
models = [
    "y1",  # LSST year 1
    "y5",
    "y10",
    # "y10_euclid",  # LSST year 10 + Euclid photometry
    # "y10_roman",  # LSST year 10 + Roman photometry
    "truth",  # True photometry for LSST + euclid + roman
]

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
        paths.models / f"{model}_{end}"
        for model in models
        for end in ["ensemble.pkl", "ensemble_losses.pkl"]
    ],
    dependencies="emulate input catalog",
    cache=True,
    n_flows=4,
    learning_rates=[5e-6, 2e-6, 1e-6],
    epochs=[400, 50, 50],
    seed=2,
)

workflow.add_stage(
    "create inference catalogs",
    CreateInferenceCatalogs,
    [paths.catalogs / f"{model}_inference_catalog.parquet" for model in models],
    dependencies=["emulate input catalog", "train ensembles"],
    cache=True,
    n_galaxies=1_000_000,
    n_samples_per_galaxy=100,
    batch_size=100_000,
    seed=3,
)

workflow.add_stage(
    "plot photo-z cut",
    PlotPhotoZCut,
    paths.figures / "photoz_cut.pdf",
    dependencies=["create inference catalogs"],
    zmin=2.36,
    zmax=4,
)

# Command-line interface
if __name__ == "__main__":
    workflow.cli()
