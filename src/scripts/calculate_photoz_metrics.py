"""Calculate metrics for the photo-z cuts."""
import pickle
from pathlib import PosixPath
from typing import Dict, Union

import numpy as np
import pandas as pd
from showyourwork.paths import user as Paths
from utils import load_truth_catalog

# instantiate the paths
paths = Paths()

# directories where the catalogs are saved
catalog_dir = paths.data / "observed_catalogs"
bg_dir = paths.data / "background_catalogs"
fg_dir = paths.data / "foreground_catalogs"

# define scaling variables for calculating projected sample size
# the lsst_scale scales sim's gold sample -> projected lsst gold sample
lsst_scale = 3e9 / load_truth_catalog().query("i < 25.3").shape[0]
# the euclid ratio is A_euclid_overlap / A_LSST
euclid_ratio = 8_000 / 18_000  # overlap area from arXiv:2108.01201
# same for roman
roman_ratio = 2_200 / 18_000  # overlap area from arXiv:1503.03757


def calculate_metrics(dir: PosixPath, query: str) -> dict:
    """Calculate the metrics for this photo-z sample.

    Parameters
    ----------
    dir : PosixPath
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

    for file in catalog_dir.glob("*.pkl"):
        # load the catalogs
        full_catalog = pd.read_pickle(file)
        cut_catalog = pd.read_pickle(list(dir.glob(file.stem + "_*.pkl"))[0])

        # calculate the metrics
        true = full_catalog.query(query).index
        pred = cut_catalog.index
        purity = np.isin(pred, true, assume_unique=True).mean()
        completeness = np.isin(true, pred, assume_unique=True).mean()
        size = lsst_scale * pred.size

        # save the metrics from this file
        if "euclid" in file.stem:
            euclid_purity = purity
            euclid_completeness = completeness
            euclid_size = euclid_ratio * size
        elif "roman" in file.stem:
            roman_purity = purity
            roman_completeness = completeness
            roman_size = roman_ratio * size
        else:
            year = int(file.stem[5:])
            purity_dict[year] = purity
            completeness_dict[year] = completeness
            size_dict[year] = size

    # sort each dictionary and add the euclid and roman numbers at the end
    purity_dict = dict(sorted(purity_dict.items()))
    purity_dict["euclid"] = euclid_purity
    purity_dict["roman"] = roman_purity

    completeness_dict = dict(sorted(completeness_dict.items()))
    completeness_dict["euclid"] = euclid_completeness
    completeness_dict["roman"] = roman_completeness

    size_dict = dict(sorted(size_dict.items()))
    size_dict["euclid"] = euclid_size
    size_dict["roman"] = roman_size
    size_dict["Y10+euclid+roman"] = (
        (1 - euclid_ratio - roman_ratio) * size_dict[10] + euclid_size + roman_size
    )

    # package all the metrics in a single dictionary
    metrics = {
        "purity": purity_dict,
        "completeness": completeness_dict,
        "size": size_dict,
    }

    return metrics


# calculate the metrics for the background sample
bg_metrics = calculate_metrics(bg_dir, "redshift > 2.36")
with open(paths.data / "photoz_metrics_bg.pkl", "wb") as file:
    pickle.dump(bg_metrics, file)

# calculate the metrics for the foreground sample
fg_metrics = calculate_metrics(fg_dir, "(redshift > 1.63) & (redshift < 2.36)")
with open(paths.data / "photoz_metrics_fg.pkl", "wb") as file:
    pickle.dump(fg_metrics, file)
