"""Perform the redshift cuts."""
from pathlib import Path

import pandas as pd
from utils import load_ensemble, paths, sample_with_errors

# create the directories where we will save the bg and fg catalogs
Path.mkdir(paths.bg, exist_ok=True)
Path.mkdir(paths.fg, exist_ok=True)

# loop over every observed catalog
for file in paths.obs.glob("*.pkl"):
    print("performing redshift cuts for", file.stem)

    # load the catalog
    catalog = pd.read_pickle(file)

    # load the flow ensemble
    ensemble = load_ensemble(file.stem)

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
    bg_file = paths.bg / Path(file.stem + "_bg.pkl")
    catalog[bg_flags].to_pickle(bg_file)

    # foreground catalog
    fg_file = paths.fg / Path(file.stem + "_fg.pkl")
    catalog[fg_flags].to_pickle(fg_file)
