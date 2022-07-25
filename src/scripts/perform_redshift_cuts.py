"""Perform the redshift cuts."""
from pathlib import Path

import pandas as pd
from pzflow import FlowEnsemble
from showyourwork.paths import user as Paths
from utils import sample_with_errors

# instantiate the paths
paths = Paths()

# load the flow ensembles
model_dir = paths.data / "models"
flow_ensembles = {
    "lsst": FlowEnsemble(file=model_dir / "lsst_ensemble.pzflow.pkl"),
    "lsst+euclid": FlowEnsemble(file=model_dir / "lsst+euclid_ensemble.pzflow.pkl"),
    "lsst+roman": FlowEnsemble(file=model_dir / "lsst+roman_ensemble.pzflow.pkl"),
}

# directory where the observed catalogs are saved
catalog_dir = paths.data / "observed_catalogs"

# create the directories where we will save the bg and fg catalogs
bg_dir = paths.data / "background_catalogs"
Path.mkdir(bg_dir, exist_ok=True)

fg_dir = paths.data / "foreground_catalogs"
Path.mkdir(fg_dir, exist_ok=True)

# loop over every observed catalog
for file in catalog_dir.glob("*.pkl"):
    print("performing redshift cuts for", file.stem)

    # load the catalog
    catalog = pd.read_pickle(file)

    # select the correct flow ensemble
    if "euclid" in file.stem:
        ensemble = flow_ensembles["lsst+euclid"]
    elif "roman" in file.stem:
        ensemble = flow_ensembles["lsst+roman"]
    else:
        ensemble = flow_ensembles["lsst"]

    # loop over batches
    bg_flags = []
    fg_flags = []
    batch_size = 10_000
    for idx in range(0, len(catalog), batch_size):
        # get the batch
        batch = catalog[idx : idx + batch_size]

        # draw redshift samples
        z_samples, u_samples = sample_with_errors(batch, ensemble, seed=idx)

        # photo-z cuts
        bg_flags += list((z_samples > 2.36).mean(axis=1) >= 0.95)
        fg_flags += list(((z_samples > 1.63) & (z_samples < 2.36)).mean(axis=1) >= 0.95)

    # background catalog
    bg_file = bg_dir / Path(file.stem + "_bg.pkl")
    catalog[bg_flags].to_pickle(bg_file)

    # foreground catalog
    fg_file = fg_dir / Path(file.stem + "_fg.pkl")
    catalog[fg_flags].to_pickle(fg_file)
