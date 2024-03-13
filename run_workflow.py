"""Run all functions for analysis."""

from src import *
from src.utils import Workflow, paths

# Create the workflow
workflow = Workflow()

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


if __name__ == "__main__":
    workflow.run_stages()
