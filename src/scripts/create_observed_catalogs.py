"""Simulate photometric errors to create observed catalogs."""
from pathlib import Path

import numpy as np
from showyourwork.paths import user as Paths
from utils import load_truth_catalog, lya_decrement
from utils.error_models import EuclidErrorModel, LSSTErrorModel, RomanErrorModel

# instantiate the paths
paths = Paths()

# load the truth catalog
truth_catalog = load_truth_catalog()

# add lya extinction
spec_idx = 0
truth_catalog.u = truth_catalog.u + lya_decrement(truth_catalog.redshift, "u", spec_idx)
truth_catalog.g = truth_catalog.g + lya_decrement(truth_catalog.redshift, "g", spec_idx)
truth_catalog.r = truth_catalog.r + lya_decrement(truth_catalog.redshift, "r", spec_idx)

# first I will determine the set of galaxies that have positive fluxes in LSST Y10,
# Euclid, and Roman, AND passes the LSST Y10 SNR cut
# This determines the set of galaxies that can be included in any of my catalogs

# first, create the LSST Y10 catalog
lsst_error_model = LSSTErrorModel(nYrObs=10)
lsstY10_catalog = lsst_error_model(truth_catalog, seed=10)

# now create the +Euclid catalog
euclid_error_model = EuclidErrorModel()
euclid_catalog = euclid_error_model(lsstY10_catalog, seed=0)
euclid_catalog = euclid_catalog.drop("F", axis=1)

# and the +Roman catalog
roman_error_model = RomanErrorModel()
roman_catalog = roman_error_model(lsstY10_catalog, seed=0)

# now make an SNR cut on the LSST reference band
ref_band = "i"
snr_cut = 5
mag_cut = lsst_error_model.get_limiting_mags(Nsigma=snr_cut, coadded=True)[ref_band]
mask1 = lsstY10_catalog[ref_band] < mag_cut

# also make a cut on galaxies with any negative fluxes
mask2 = np.isfinite(euclid_catalog).all(axis=1) & np.isfinite(roman_catalog).all(axis=1)

# combine these masks into a single cut
mask = mask1 & mask2

# apply cuts to our catalogs
lsstY10_catalog = lsstY10_catalog[mask].drop(list("YJHF"), axis=1)
euclid_catalog = euclid_catalog[mask]
roman_catalog = roman_catalog[mask]

# create the directory where the catalogs will be saved
catalog_dir = paths.data / "observed_catalogs"
Path.mkdir(catalog_dir, exist_ok=True)

# save these maximal catalogs
lsstY10_catalog.to_pickle(catalog_dir / "lsstY10.pkl")
euclid_catalog.to_pickle(catalog_dir / "lsstY10+euclid.pkl")
roman_catalog.to_pickle(catalog_dir / "lsstY10+roman.pkl")

# now loop through the earlier years and generate LSST catalogs
restricted_catalog = truth_catalog[mask].drop(list("YJHF"), axis=1)
for year in range(1, 10):
    # build the error model
    lsst_error_model = LSSTErrorModel(nYrObs=year)

    # add LSST errors
    lsst_catalog = lsst_error_model(restricted_catalog, seed=year)

    # apply the SNR cut on the LSST reference band
    mag_cut = lsst_error_model.get_limiting_mags(Nsigma=snr_cut, coadded=True)[ref_band]
    lsst_catalog = lsst_catalog.query(f"{ref_band} < {mag_cut}")

    # cut negative fluxes
    lsst_catalog = lsst_catalog[np.isfinite(lsst_catalog).all(axis=1)]

    # save the catalog
    lsst_catalog.to_pickle(catalog_dir / f"lsstY{year}.pkl")
