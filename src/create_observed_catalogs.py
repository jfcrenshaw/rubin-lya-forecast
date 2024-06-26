"""Create observed catalogs."""

from photerr import EuclidErrorModel, LsstErrorModel, RomanErrorModel
from pzflow import Flow

from .utils import lya_increment, paths, split_seed


def create_observed_catalogs(output: list, min_snr: float, seed: int) -> None:
    """Create observed catalogs.

    Parameters
    ----------
    output: list
        List of paths at which to save the catalogs. In order,
        truth, yr 1, yr 5, yr 10, yr 10 + Euclid, yr 10 + Roman
    min_snr: float
        Minimum SNR in the i band
    seed: int
        The random seed
    """
    # Unpack the output files
    file_truth, file_y1, file_y5, file_y10, file_y10E, file_y10R = output

    # And the seeds
    seed_truth, seed_1, seed_5, seed_10, seed_E, seed_R = split_seed(seed, len(output))

    # Load the input catalog emulator
    flow = Flow(file=paths.models / "incat_emulator.pkl")

    # Sample a truth catalog from the emulator
    cat_truth = flow.sample(2_000_100, seed=seed_truth)
    cat_truth = cat_truth[cat_truth.redshift > 0].iloc[:2_000_000]

    # Add Lya extinction
    cat_truth["u0"] = cat_truth.u.copy()
    cat_truth.u += lya_increment(cat_truth.redshift, "u")

    # Save the truth catalog
    cat_truth.to_parquet(file_truth)

    # Create and save LSST Y1 catalog
    error_model_y1 = LsstErrorModel(nYrObs=1, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y1 = error_model_y1(cat_truth, random_state=seed_1)
    cat_y1.drop(list("YJHF"), axis=1, inplace=True)  # Drop the Euclid and Roman bands
    mask = cat_y1.i <= error_model_y1.getLimitingMags(nSigma=min_snr)["i"]
    cat_y1[mask].to_parquet(file_y1)

    # Create and save LSST Y5 catalog
    error_model_y5 = LsstErrorModel(nYrObs=5, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y5 = error_model_y5(cat_truth, random_state=seed_5)
    cat_y5.drop(list("YJHF"), axis=1, inplace=True)  # Drop the Euclid and Roman bands
    mask = cat_y5.i <= error_model_y5.getLimitingMags(nSigma=min_snr)["i"]
    cat_y5[mask].to_parquet(file_y5)

    # Create and save LSST Y10 catalog
    error_model_y10 = LsstErrorModel(nYrObs=10, airmass=1.2, sigLim=1, ndMode="sigLim")
    cat_y10 = error_model_y10(cat_truth, random_state=seed_10)
    mask = cat_y10.i <= error_model_y10.getLimitingMags(nSigma=min_snr)["i"]
    cat_y10 = cat_y10[mask]
    cat_y10.drop(list("YJHF"), axis=1).to_parquet(file_y10)

    # Create and save LSST Y10 +Euclid & +Roman catalogs
    cat_y10E = EuclidErrorModel()(cat_y10, random_state=seed_E)
    cat_y10E.drop("F", axis=1, inplace=True)  # Drop Roman band
    cat_y10E.to_parquet(file_y10E)

    cat_y10R = RomanErrorModel()(cat_y10, random_state=seed_R)
    cat_y10R.to_parquet(file_y10R)
