"""Utility function for random seeds."""

import numpy as np


def split_seed(seed: int, N: int, max_seed: int = int(1e12)) -> np.ndarray:
    """Split the random seed.

    Parameters
    ----------
    seed : int
        The initial seed
    N : int
        The number of seeds to produce
    max_seed : int, default=1e12
        The maximum allowed value for seeds
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_seed, size=N)
