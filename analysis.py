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
        paths.inputs / "input_catalog.parquet",
    ],
    cache=True,
)

workflow.add_stage(
    "plot increments",
    plot_increments,
    paths.figures / "increments.pdf",
)

workflow.add_stage(
    "create catalogs",
    create_catalogs,
    [
        paths.catalogs / "truth_catalog.parquet",
        paths.catalogs / "y1_catalog.parquet",
        paths.catalogs / "y5_catalog.parquet",
        paths.catalogs / "y10+euclid_catalog.parquet",
        paths.catalogs / "y10+roman_catalog.parquet",
    ],
    cache=True,
    seeds=[1, 5, 10, 11, 12],
)


# Run the command-line interface
workflow.cli()
