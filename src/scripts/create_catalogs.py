"""Simulate photometric errors to create observed catalogs."""
from pathlib import Path

import pandas as pd
from utils import load_truth_catalog, lya_decrement, paths
from utils.error_models import EuclidErrorModel, LSSTErrorModel, RomanErrorModel

# create the directories where the catalogs will be saved
Path.mkdir(paths.obs, exist_ok=True)
Path.mkdir(paths.train, exist_ok=True)

# load the truth catalog
truth_catalog = load_truth_catalog()

# add lya extinction
lya_catalog = truth_catalog.copy()
lya_catalog.u = lya_catalog.u + lya_decrement(lya_catalog.redshift, "u", 0)


# define a function to select a training set
def select_training(catalog: pd.DataFrame) -> pd.DataFrame:
    """Randomly select a subset of the catalog for training.

    Selects indices and columns from the observed catalog, so that the selection
    function is included, but selects magnitudes from the truth catalog.

    Parameters
    ----------
    catalog: pd.DataFrame
        The observed catalog to serve as the source of indices and columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame of truth galaxies for training the flows.
    """
    train_catalog = truth_catalog.iloc[catalog.sample(120_000, random_state=0).index]
    train_catalog = train_catalog[[col for col in catalog.columns if "_err" not in col]]
    return train_catalog


# now, create the LSST Y10 catalog
lsst_error_model = LSSTErrorModel(nYrObs=10)
lsstY10_catalog = lsst_error_model(lya_catalog, seed=10)

# and the +Euclid catalog
euclid_error_model = EuclidErrorModel()
euclid_catalog = euclid_error_model(lsstY10_catalog, seed=0)
euclid_catalog = euclid_catalog.drop("F", axis=1)

# and the +Roman catalog
roman_error_model = RomanErrorModel()
roman_catalog = roman_error_model(lsstY10_catalog, seed=0)

# perform a SNR > 5 cut on LSST i
i_cut = lsst_error_model.get_limiting_mags(Nsigma=5, coadded=True)["i"]
mask = lsstY10_catalog["i"] < i_cut

# apply cuts to our catalogs
lsstY10_catalog = lsstY10_catalog[mask].drop(list("YJHF"), axis=1)
euclid_catalog = euclid_catalog[mask]
roman_catalog = roman_catalog[mask]

# save these catalogs
lsstY10_catalog.to_pickle(paths.obs / "lsstY10.pkl")
select_training(lsstY10_catalog).to_pickle(paths.train / "lsstY10.pkl")

euclid_catalog.to_pickle(paths.obs / "lsstY10+euclid.pkl")
select_training(euclid_catalog).to_pickle(paths.train / "lsstY10+euclid.pkl")

roman_catalog.to_pickle(paths.obs / "lsstY10+roman.pkl")
select_training(roman_catalog).to_pickle(paths.train / "lsstY10+roman.pkl")

# now loop through the earlier years and generate LSST catalogs
restricted_catalog = lya_catalog.iloc[lsstY10_catalog.index].drop(list("YJHF"), axis=1)
for year in [1, 5]:
    # build the error model
    lsst_error_model = LSSTErrorModel(nYrObs=year)

    # add LSST errors
    lsst_catalog = lsst_error_model(restricted_catalog, seed=year)

    # apply the SNR > 5 cut on the i band
    i_cut = lsst_error_model.get_limiting_mags(Nsigma=5, coadded=True)["i"]
    lsst_catalog = lsst_catalog.query(f"i < {i_cut}")

    # save the catalogs
    lsst_catalog.to_pickle(paths.obs / f"lsstY{year}.pkl")
    select_training(lsst_catalog).to_pickle(paths.train / f"lsstY{year}.pkl")
