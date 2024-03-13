"""Create observed catalogs."""

import pandas as pd
from photerr import EuclidErrorModel, LsstErrorModel, RomanErrorModel

from .utils import lya_increment, paths


def create_catalogs(output: list, seeds: list) -> None:
    """Create observed catalogs."""
    # Unpack the output files
    file_truth, file_y1, file_y5, file_y10E, file_y10R = output

    # And the seeds
    seed_1, seed_5, seed_10, seed_E, seed_R = seeds

    # Load the input catalog
    cat_truth = pd.read_parquet(paths.inputs / "input_catalog.parquet")
    cat_truth = cat_truth.iloc[:, :-2]

    # Add Lya extinction
    cat_truth["u0"] = cat_truth.u.copy()
    cat_truth.u += lya_increment(cat_truth.redshift, "u")

    # Save the truth catalog
    cat_truth.to_parquet(file_truth)

    # Create and save LSST Y1 catalog
    error_model_y1 = LsstErrorModel(nYrObs=1, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y1 = error_model_y1(cat_truth, random_state=seed_1)
    cat_y1.drop(list("YJHF"), axis=1, inplace=True)  # Drop the Euclid and Roman bands
    cat_y1.to_parquet(file_y1)

    # Create and save LSST Y5 catalog
    error_model_y5 = LsstErrorModel(nYrObs=5, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y5 = error_model_y5(cat_truth, random_state=seed_5)
    cat_y5.drop(list("YJHF"), axis=1, inplace=True)  # Drop the Euclid and Roman bands
    cat_y5.to_parquet(file_y5)

    # Create and save LSST Y10 +Euclid & +Roman catalogs
    error_model_y10 = LsstErrorModel(nYrObs=10, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y10 = error_model_y10(cat_truth, random_state=seed_10)

    cat_y10E = EuclidErrorModel()(cat_y10, random_state=seed_E)
    cat_y10E.drop("F", axis=1, inplace=True)  # Drop Roman band
    cat_y10E.to_parquet(file_y10E)

    cat_y10R = RomanErrorModel()(cat_y10, random_state=seed_R)
    cat_y10R.to_parquet(file_y10R)
