"""Module to calculate the expected Lyman-alpha magnitude increment."""

import numpy as np

from .constants import LYMAN_WAVELEN
from .paths import paths


class Bandpass:
    """An LSST bandpass."""

    def __init__(self, band: str) -> None:
        """An LSST bandpass.

        Parameters
        ----------
        band: str
            The name of the bandpass to load.
        """
        # load the raw bandpass
        wavelen, R = np.loadtxt(paths.bandpasses / f"{band}_bandpass.dat", unpack=True)

        # nm -> angstroms
        wavelen *= 10

        # get the range where R is non-zero
        wavelen, R = wavelen[R > 0], R[R > 0]

        # normalize R
        R *= wavelen
        R /= np.trapz(R, wavelen)

        # save the values
        self.wavelen = wavelen
        self._R = R

        # initialize the weighted R
        self._R_weighted = self._R

    def R(self, wavelen: np.ndarray | float) -> np.ndarray | float:
        """Return the bandpass response at the requested wavelength.

        Parameters
        ----------
        wavelen: np.ndarray or float
            Wavelength(s) in angstroms.

        Returns
        -------
        np.ndarray or float
            The bandpass response at the requested wavelength(s).
        """
        return np.interp(wavelen, self.wavelen, self._R_weighted, left=0, right=0)

    def reweight_bandpass(self, wavelen: np.ndarray, weights: np.ndarray) -> None:
        """Re-weight the bandpass.

        Only the shape of the function specified by f: wavelen -> weights matters.
        The normalization is irrelevant.

        After called, bandpass.R will use the new, re-weighted bandpass.
        This method has no memory of past calls - i.e. it re-weights the original
        bandpass, not the most recent re-weighted version.

        Parameters
        ----------
        wavelen: np.ndarray
            The wavelengths where the weights are defined
        weights: np.ndarray
            The weights used to re-weight the bandpass. Must be non-negative.
        """
        # check that all weights are positive
        if any(weights < 0):
            raise ValueError("The weights must be non-negative.")

        # re-grid the weights
        rg_weights = np.interp(self.wavelen, wavelen, weights)

        # re-weight R
        R_weighted = self._R * rg_weights

        # calculate the new norm
        norm = np.trapz(R_weighted, self.wavelen)

        if norm > 0:
            # re-normalize
            R_weighted /= norm
        else:
            R_weighted *= 0

        # save the new weights
        self._R_weighted = R_weighted

    def reset(self) -> None:
        """Reset the bandpass weights."""
        self._R_weighted = self._R


def tau_eff(z: np.ndarray | float) -> np.ndarray | float:
    """Return the effective optical depth of the Lyman-alpha Forest at redshift z.

    Parameters
    ----------
    z: np.ndarray or float
        Redshift(s)

    Returns
    -------
    np.ndarray or float
        The effective optical depth
    """
    # parameters of the mean Lyman-alpha optical depth
    # values from https://arxiv.org/abs/1904.01110
    TAU_0 = 5.54e-3
    TAU_GAMMA = 3.182

    return TAU_0 * (1 + z) ** TAU_GAMMA


def F_bar(z: np.ndarray | float) -> np.ndarray | float:
    """Return the mean Lyman-alpha transmission at redshift z.

    Parameters
    ----------
    z: np.ndarray or float
        Redshift(s)

    Returns
    -------
    np.ndarray or float
        The mean Lyman-alpha transmission.
    """
    return np.exp(-tau_eff(z))


def lya_increment(
    redshift: np.ndarray | float,
    band: str,
    spectral_index: float = 1,
    batch_size: int = 1000,
) -> np.ndarray | float:
    """Calculate mean Lya increment for the given band as a function of redshift.

    Supports re-weighting the increments using a different mean galaxy spectrum.
    Galaxy spectra are assumed to be ~ lambda^spectral_index.
    By default, we use a flat f_nu, i.e. spectral_index = 1.
    spectral_index=0 corresponds to a flat f_lambda.

    Parameters
    ----------
    redshift: np.ndarray or float
        Array of redshifts at which to calculate the increment.
    band: str
        Name of the band to calculate increment for. Supports "u", "g", and "r".
    spectral_index: float, default=1
        Controls the mean galaxy spectrum. See the note above.
    batch_size: int, default=1000
        The batch size for calculating increments.

    Returns
    -------
    np.ndarray or float
        Lyman-alpha increments.
    """
    # make sure redshift is an array
    z: np.ndarray = np.atleast_1d(redshift)

    # load the bandpass
    bandpass = Bandpass(band)

    # weight the bandpass according to the mean galaxy spectrum
    mean_sed = bandpass.wavelen**spectral_index
    bandpass.reweight_bandpass(bandpass.wavelen, mean_sed)

    increments = []
    for i in range(0, len(z), batch_size):
        z_batch = z[i : i + batch_size]

        # get the redshift grid
        z_grid = np.tile(bandpass.wavelen / LYMAN_WAVELEN - 1, (len(z_batch), 1))

        # calculate the transmission as a function of redshift
        F_grid = F_bar(z_grid)

        # beyond the source redshift, set transmission = 1
        F_grid[z_grid > z_batch[:, None]] = 1  # type: ignore

        # convert the redshift grid to wavelength
        wavelens = LYMAN_WAVELEN * (1 + z_grid)

        # calculate the increments
        incr = -2.5 * np.log10(np.trapz(bandpass.R(wavelens) * F_grid, wavelens))

        increments += list(incr)

    return np.array(increments).squeeze()
