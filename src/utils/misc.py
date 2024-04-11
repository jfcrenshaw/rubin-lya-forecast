"""Utility function for random seeds."""

import numpy as np


def split_seed(seed: int, N: int = 2, max_seed: int = 2**32 - 1) -> np.ndarray:
    """Split the random seed.

    Parameters
    ----------
    seed : int
        The initial seed
    N : int, default=2
        The number of seeds to produce
    max_seed : int, default=2**32 - 1
        The maximum allowed value for seeds
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_seed, size=N)
