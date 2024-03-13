"""Define the paths where everything is saved."""

from dataclasses import dataclass
from pathlib import Path

import git


@dataclass
class _Paths:
    root = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)


paths = _Paths()

paths.data = paths.root / "data"
paths.inputs = paths.data / "inputs"
paths.catalogs = paths.data / "catalogs"
paths.models = paths.root / "models"

paths.figures = paths.root / "figures"
