"""Utility function to load the ensembles."""
from pzflow import FlowEnsemble

from .paths import paths


def load_ensemble(name: str) -> FlowEnsemble:
    """Load the FlowEnsemble to use with the given file path.

    Parameters
    ----------
    path: Path
        The path to a file
    """
    return FlowEnsemble(file=paths.models / f"{name}_ensemble.pzflow.pkl")
