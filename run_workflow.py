"""Run all functions for analysis."""

from src import *
from src.utils import Workflow, paths

# Create the workflow
workflow = Workflow()

# Set the cache tag
workflow.cache_tag = "v1"

# Add stages to workflow
workflow.add_stage(
    "download data",
    None,
    [
        paths.inputs / "u_bandpass.dat",
        paths.inputs / "g_bandpass.dat",
        paths.inputs / "r_bandpass.dat",
        paths.inputs / "truth_catalog.parquet",
    ],
    cache=True,
)

workflow.add_stage(
    "plot increments",
    plot_increments,
    paths.figures / "increments.pdf",
)

# Run the command-line interface
workflow.cli()
