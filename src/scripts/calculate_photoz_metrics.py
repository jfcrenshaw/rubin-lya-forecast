"""Calculate metrics for the photo-z cuts."""
import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from utils import paths
from utils.survey_areas import A_RATIO_EUCLID, A_RATIO_ROMAN

# define scaling variable for calculating projected sample size
# the lsst_scale scales sim's gold sample -> projected lsst gold sample
N_projected_gold = 3.11e9
N_simulated_gold = pd.read_pickle(paths.obs / "lsstY10.pkl").query("i < 25.3").shape[0]
lsst_scale = N_projected_gold / N_simulated_gold


def calculate_metrics(dir: Path, query: str) -> dict:
    """Calculate the metrics for this photo-z sample.

    Parameters
    ----------
    dir : Path
        The path to the photo-z selected samples.
    query : str
        The query corresponding to the cut on true redshifts.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    # create the dictionaries that will collect the metrics
    purity_dict: Dict[Union[int, str], float] = {}
    completeness_dict: Dict[Union[int, str], float] = {}
    size_dict: Dict[Union[int, str], float] = {}

    for file in paths.obs.glob("*.pkl"):
        # load the catalogs
        full_catalog = pd.read_pickle(file)
        cut_catalog = pd.read_pickle(list(dir.glob(file.stem + "_*.pkl"))[0])

        # calculate the metrics
        true = full_catalog.query(query).index
        pred = cut_catalog.index
        purity = np.isin(pred, true, assume_unique=True).mean()
        completeness = np.isin(true, pred, assume_unique=True).mean()
        size = lsst_scale * pred.size

        # rescale the size for NIR catalogs
        if "euclid" in file.stem:
            size *= A_RATIO_EUCLID
        elif "roman" in file.stem:
            size *= A_RATIO_ROMAN

        purity_dict[file.stem] = purity
        completeness_dict[file.stem] = completeness
        size_dict[file.stem] = size

    # save a size for the combined catalog
    size_dict["lsstY10+both"] = (
        (1 - A_RATIO_EUCLID - A_RATIO_ROMAN) * size_dict["lsstY10"]
        + size_dict["lsstY10+euclid"]
        + size_dict["lsstY10+roman"]
    )

    # package all the metrics in a single dictionary
    metrics = {
        "purity": purity_dict,
        "completeness": completeness_dict,
        "size": size_dict,
    }

    return metrics


# calculate the metrics for the background sample
bg_metrics = calculate_metrics(paths.bg, "redshift > 2.36")
with open(paths.data / "photoz_metrics_bg.pkl", "wb") as file:
    pickle.dump(bg_metrics, file)

# calculate the metrics for the foreground sample
fg_metrics = calculate_metrics(paths.fg, "(redshift > 1.63) & (redshift < 2.36)")
with open(paths.data / "photoz_metrics_fg.pkl", "wb") as file:
    pickle.dump(fg_metrics, file)
