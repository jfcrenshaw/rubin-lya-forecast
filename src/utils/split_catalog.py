"""Utility that always splits catalogs in the same way."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_catalog(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the catalog into train and test sets

    Parameters
    ----------
    catalog: pd.DataFrame
        Catalog to split into train and test sets

    Returns
    -------
    pd.DataFrame
        The training set
    pd.DataFrame
        The test set
    """
    seed = 1234321  # Random seed
    train, test = train_test_split(catalog, train_size=100_000, random_state=seed)

    return train, test
