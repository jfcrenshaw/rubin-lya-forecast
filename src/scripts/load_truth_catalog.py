"""Load Melissa's truth catalog."""

import pandas as pd
import paths


def load_truth_catalog() -> pd.DataFrame:
    """Return the truth catalog as a pandas DataFrame.
    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of the truth catalog.
    """
    return pd.read_csv(
        paths.data / "Euclid_trim_27p10_3p5_IR_4NUV.dat",
        delim_whitespace=True,
        comment="#",
        nrows=None,
        header=0,
        usecols=list(range(1, 12)),
        names=[
            "redshift",  # truth
            "u",
            "g",
            "r",
            "i",
            "z",
            "y",
            "Y",  # Euclid & Roman
            "J",  # Euclid & Roman
            "H",  # Euclid & Roman
            "F",  # Roman
        ],
    )
