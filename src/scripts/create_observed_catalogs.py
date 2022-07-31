"""Simulate photometric errors to create observed catalogs."""
from pathlib import Path

import numpy as np
from showyourwork.paths import user as Paths
from utils import load_truth_catalog, lya_decrement
from utils.error_models import EuclidErrorModel, LSSTErrorModel, RomanErrorModel

# instantiate the paths
paths = Paths()

# create the directory where the catalogs will be saved
catalog_dir = paths.data / "observed_catalogs"
Path.mkdir(catalog_dir, exist_ok=True)

# load the truth catalog
truth_catalog = load_truth_catalog()

# add lya extinction
truth_catalog.u = truth_catalog.u + lya_decrement(truth_catalog.redshift, "u", 0)

# save the perfect catalog
perfect_catalog = truth_catalog.copy()
for col in truth_catalog.columns[1:]:
    perfect_catalog[f"{col}_err"] = np.zeros(len(perfect_catalog))
perfect_catalog.to_pickle(catalog_dir / "perfect.pkl")

# now I will determine the set of galaxies that have positive fluxes in LSST Y10,
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

# make SNR > 3 cuts on all bands
mask1 = lsstY10_catalog.eval(
    " & ".join(
        f"({band} < {cut})"
        for band, cut in lsst_error_model.get_limiting_mags(
            Nsigma=1, coadded=True
        ).items()
    )
)
mask2 = euclid_catalog.eval(
    " & ".join(
        f"({band} < {cut})"
        for band, cut in euclid_error_model.get_limiting_mags(
            Nsigma=1, coadded=True
        ).items()
    )
)
mask3 = roman_catalog.eval(
    " & ".join(
        f"({band} < {cut})"
        for band, cut in roman_error_model.get_limiting_mags(
            Nsigma=1, coadded=True
        ).items()
    )
)

# and a SNR > 5 cut on LSST i
i_cut = lsst_error_model.get_limiting_mags(Nsigma=5, coadded=True)["i"]
mask4 = lsstY10_catalog["i"] < i_cut

# also make a cut on galaxies with any negative fluxes
mask5 = (
    np.isfinite(lsstY10_catalog).all(axis=1)
    & np.isfinite(euclid_catalog).all(axis=1)
    & np.isfinite(roman_catalog).all(axis=1)
)

# combine these masks into a single cut
mask = mask1 & mask2 & mask3 & mask4 & mask5

# apply cuts to our catalogs
lsstY10_catalog = lsstY10_catalog[mask].drop(list("YJHF"), axis=1)
euclid_catalog = euclid_catalog[mask]
roman_catalog = roman_catalog[mask]

# save these maximal catalogs
lsstY10_catalog.to_pickle(catalog_dir / "lsstY10.pkl")
euclid_catalog.to_pickle(catalog_dir / "lsstY10+euclid.pkl")
roman_catalog.to_pickle(catalog_dir / "lsstY10+roman.pkl")

# now loop through the earlier years and generate LSST catalogs
restricted_catalog = truth_catalog[mask].drop(list("YJHF"), axis=1)
for year in [1, 5, 10]:
    # build the error model
    lsst_error_model = LSSTErrorModel(nYrObs=year)

    # add LSST errors
    lsst_catalog = lsst_error_model(restricted_catalog, seed=year)

    # apply the SNR > 3 cut
    lsst_catalog = lsst_catalog.query(
        " & ".join(
            f"({band} < {cut})"
            for band, cut in lsst_error_model.get_limiting_mags(
                Nsigma=1, coadded=True
            ).items()
        )
    )

    # apply the SNR > 5 cut on the i band
    i_cut = lsst_error_model.get_limiting_mags(Nsigma=5, coadded=True)["i"]
    lsst_catalog = lsst_catalog.query(f"i < {i_cut}")

    # cut negative fluxes
    lsst_catalog = lsst_catalog[np.isfinite(lsst_catalog).all(axis=1)]

    # save the catalog
    lsst_catalog.to_pickle(catalog_dir / f"lsstY{year}.pkl")
