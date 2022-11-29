"""Photometric Error Models from RAIL - the old functional version."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


class LSSTErrorModel:
    """LSST Model for photometric errors.

    Implements the error model from the LSST Overview Paper:
    https://arxiv.org/abs/0805.2366

    Note however that this paper gives the high SNR approximation.
    By default, this model uses the more accurate version of the error model
    where Eq. 5 = (N/S)^2, in flux, and the error is Gaussian in flux space.
    There is a flag allowing you to use the high SNR approximation instead.
    See the __init__ docstring.

    Create an instance by calling the class, then use the instance as a
    callable on pandas DataFrames.
    Example usage:
    errModel = LSSTErrorModel()
    data_with_errs = errModel(data)
    """

    # the default settings of the error model
    # all values come from the LSST Overview Paper: https://arxiv.org/abs/0805.2366
    # Each setting is defined with an inline comment, and the location
    # of the number in the paper is designated with the following codes:
    #   pN - page number N
    #   T1 - Table 1, on page 11
    #   T2 - Table 2, on page 26
    default_settings = {
        "bandNames": {  # provided so you can alias the names of the bands
            "u": "u",
            "g": "g",
            "r": "r",
            "i": "i",
            "z": "z",
            "y": "y",
        },
        "tvis": 30.0,  # exposure time for a single visit in seconds, p12
        "nYrObs": 10.0,  # number of years of observations
        "nVisYr": {  # mean number of visits per year in each filter (T1)
            "u": 5.6,
            "g": 8.0,
            "r": 18.4,
            "i": 18.4,
            "z": 16.0,
            "y": 16.0,
        },
        "gamma": {  # band dependent parameter (T2)
            "u": 0.038,
            "g": 0.039,
            "r": 0.039,
            "i": 0.039,
            "z": 0.039,
            "y": 0.039,
        },
        "airmass": 1.2,  # fiducial airmass (T2)
        "extendedSource": 0.0,  # constant added to m5 for extended sources
        "sigmaSys": 0.005,  # expected irreducible error, p26
        "magLim": np.inf,  # dimmest allowed magnitude; dimmer mags set to ndFlag
        "ndFlag": np.inf,  # flag for non-detections (all mags > magLim)
        "m5": {},  # explicit list of m5 limiting magnitudes
        "Cm": {  # band dependent parameter (T2)
            "u": 23.09,
            "g": 24.42,
            "r": 24.44,
            "i": 24.32,
            "z": 24.16,
            "y": 23.73,
        },
        "msky": {  # median zenith sky brightness at Cerro Pachon (T2)
            "u": 22.99,
            "g": 22.26,
            "r": 21.20,
            "i": 20.48,
            "z": 19.60,
            "y": 18.61,
        },
        "theta": {  # median zenith seeing FWHM, arcseconds (T2)
            "u": 0.81,
            "g": 0.77,
            "r": 0.73,
            "i": 0.71,
            "z": 0.69,
            "y": 0.68,
        },
        "km": {  # atmospheric extinction (T2)
            "u": 0.491,
            "g": 0.213,
            "r": 0.126,
            "i": 0.096,
            "z": 0.069,
            "y": 0.170,
        },
        "highSNR": False,
    }

    def __init__(
        self,
        bandNames: dict = None,
        tvis: float = None,
        nYrObs: float = None,
        nVisYr: dict = None,
        gamma: dict = None,
        airmass: float = None,
        extendedSource: float = None,
        sigmaSys: float = None,
        magLim: float = None,
        ndFlag: float = None,
        m5: dict = None,
        Cm: dict = None,
        msky: dict = None,
        theta: dict = None,
        km: dict = None,
        highSNR: bool = None,
    ) -> None:
        """Error model from the LSST Overview Paper.

        All parameters are optional. To see the default settings, do
        `LSSTErrorModel().default_settings`

        By default, this model uses the more accurate version of the error
        model. See the explanations in the class docstring and the description
        for highSNR below.

        Note that the dictionary bandNames sets the bands for which this model
        calculates photometric errors. The dictionary keys are the band names
        that the error model uses internally to search for parameters, and the
        corresponding dictionary values are the band names as they appear in
        your data set. By default, the LSST bands are named "u", "g", "r", "i",
        "z", and "y". You can use the bandNames dictionary to alias them differently.

        For example, if in your DataFrame, the bands are named lsst_u, lsst_g, etc.
        you can set bandNames = {"u": "lsst_u", "g": "lsst_g", ...},
        and the error model will work automatically.

        You can also add other bands to bandNames. For example, if you want to
        use the same model to calculate photometric errors for Euclid bands, you
        can include {"euclid_y": "euclid_y", "euclid_j": "euclid_j", ...}.

        In this case, you must include the additional information listed below...
        IMPORTANT: For every band in bandNames, you must provide:
            - nVisYr
            - gamma
            - the single-visit 5-sigma limiting magnitude. You can do this either by
                (1) explicitly providing it in the m5 dictionary, or
                (2) by adding the corresponding parameters to Cm, msky, theta,
                and km, in which case the limiting magnitude will be calculated
                for you, using Eq. 6 from the LSST Overview Paper.

        Note if for any bands, you explicitly pass a limiting magnitude in the
        m5 dictionary, the model will use the explicitly passed value,
        regardless of the values in Cm, msky, theta, and km.

        Parameters
        ----------
        bandNames : dict, optional
            A dictionary of bands for which to calculate errors. The dictionary
            keys are the band names that the Error Model uses internally to
            search for parameters, and the corresponding dictionary values
            are the names of those bands as they appear in your data set.
            Can be used to alias the default names of the LSST bands, or to add
            additional bands. See notes above.
        tvis : float, optional
            Exposure time for a single visit
        nYrObs : float, optional
            Number of years of observations
        nVisYr : dict, optional
            Mean number of visits per year in each band
        gamma : dict, optional
            A band dependent parameter defined in the LSST Overview Paper
        airmass : float, optional
            The fiducial airmass
        extendedSource : float, optional
            Constant to add to magnitudes of extended sources.
            The error model is designed to emulate magnitude errors for point
            sources. This constant provides a zeroth order correction accounting
            for the fact that extended sources have larger uncertainties. Note
            this is only meant to account for small, slightly extended sources.
            For typical LSST galaxies, this may be of order ~0.3.
        sigmaSys : float, optional
            The irreducible error of the system in AB magnitudes.
            Set's the minimum photometric error.
        magLim : float, optional
            The dimmest magnitude allowed. All dimmer magnitudes are set to ndFlag.
        ndFlag : float, optional
            The flag for non-detections. All magnitudes greater than magLim (and
            their corresponding errors) will be set to this value.
        m5 : dict, optional
            A dictionary of single visit 5-sigma limiting magnitudes. For any
            bands for which you pass a value in m5, this will be the 5-sigma
            limiting magnitude used, and any values for that band in Cm, msky,
            theta, and km will be ignored.
        Cm : dict, optional
            A band dependent parameter defined in the LSST Overview Paper
        msky : dict, optional
            Median zenith sky brightness in each band
        theta : dict, optional
            Median zenith seeing FWHM (in arcseconds) for each band
        km : dict, optional
            Atmospheric extinction in each band
        highSNR : bool, default=False
            Sets whether you use the high SNR approximation given in the LSST
            Overview Paper. If False, then Eq. 5 from the LSST Error Model is
            used to calculate (N/S)^2 in flux, and errors are Gaussian in flux
            space. If True, then Eq. 5 is used to calculate the squared error
            in magnitude space, and errors are Gaussian in magnitude space.
        """
        # update the settings
        settings: dict = self.default_settings.copy()
        if bandNames is not None:
            settings["bandNames"] = bandNames
        if tvis is not None:
            settings["tvis"] = tvis
        if nYrObs is not None:
            settings["nYrObs"] = nYrObs
        if nVisYr is not None:
            settings["nVisYr"] = nVisYr
        if gamma is not None:
            settings["gamma"] = gamma
        if airmass is not None:
            settings["airmass"] = airmass
        if extendedSource is not None:
            settings["extendedSource"] = extendedSource
        if sigmaSys is not None:
            settings["sigmaSys"] = sigmaSys
        if magLim is not None:
            settings["magLim"] = magLim
        if ndFlag is not None:
            settings["ndFlag"] = ndFlag
        if Cm is not None:
            settings["Cm"] = Cm
        if msky is not None:
            settings["msky"] = msky
        if theta is not None:
            settings["theta"] = theta
        if km is not None:
            settings["km"] = km
        if m5 is not None:
            # make sure it's a dictionary
            if not isinstance(m5, dict):
                raise TypeError("m5 must be a dictionary, or None.")
            # save m5
            settings["m5"] = m5
            # remove these bands from the dictionaries that hold information
            # about how to calculate m5
            for key1 in ["Cm", "msky", "theta", "km"]:
                for key2 in m5:
                    settings[key1].pop(key2, None)
        if highSNR is not None:
            settings["highSNR"] = highSNR

        self.settings = settings

        # calculate the single-visit 5-sigma limiting magnitudes using the settings
        self._all_m5 = self._calculate_m5()

        # update the limiting magnitudes with any m5s passed
        self._all_m5.update(self.settings["m5"])

    def _calculate_m5(self) -> dict:
        """Calculate the single-visit m5 limiting magnitudes.

        Uses Eq. 6 from https://arxiv.org/abs/0805.2366
        Note this is only done for the bands for which an m5 wasn't
        explicitly passed.
        """

        # get the settings
        settings = self.settings

        # get the list of bands for which an m5 wasn't explicitly passed
        bands = set(self.settings["bandNames"]) - set(self.settings["m5"])
        bands = [
            band for band in self.settings["bandNames"] if band in bands
        ]  # type: ignore

        # calculate the m5 limiting magnitudes using Eq. 6
        m5 = {
            band: settings["Cm"][band]
            + 0.50 * (self.settings["msky"][band] - 21)
            + 2.5 * np.log10(0.7 / settings["theta"][band])
            + 1.25 * np.log10(settings["tvis"] / 30)
            - settings["km"][band] * (settings["airmass"] - 1)
            - settings["extendedSource"]
            for band in bands
        }

        return m5

    def _get_bands_and_names(
        self, columns: Iterable[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Get the bands and bandNames that are present in the given data columns.
        """

        # get the list of bands present in the data
        bandNames = list(set(self.settings["bandNames"].values()).intersection(columns))

        # sort bandNames to be in the same order provided in settings["bandNames"]
        bandNames = [
            band for band in self.settings["bandNames"].values() if band in bandNames
        ]

        # get the internal names of the bands from bandNames
        bands = [
            {bandName: band for band, bandName in self.settings["bandNames"].items()}[
                bandName
            ]
            for bandName in bandNames
        ]

        return bands, bandNames

    def _get_NSR(self, mags: np.ndarray, bands: list) -> np.ndarray:
        """Calculate the noise-to-signal ratio using Eqs 4 and 5 from
        https://arxiv.org/abs/0805.2366
        """

        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])
        # and the values for gamma
        gamma = np.array([self.settings["gamma"][band] for band in bands])

        # calculate x as defined in the paper
        x = 10 ** (0.4 * np.subtract(mags, m5))

        # calculate the squared NSR for a single visit
        # Eq. 5 in https://arxiv.org/abs/0805.2366
        nsrRandSqSingleExp = (0.04 - gamma) * x + gamma * x**2

        # calculate the random NSR for the stacked image
        nVisYr = np.array([self.settings["nVisYr"][band] for band in bands])
        nStackedObs = nVisYr * self.settings["nYrObs"]
        nsrRand = np.sqrt(nsrRandSqSingleExp / nStackedObs)

        # get the irreducible system NSR
        if self.settings["highSNR"]:
            nsrSys = self.settings["sigmaSys"]
        else:
            nsrSys = 10 ** (self.settings["sigmaSys"] / 2.5) - 1

        # calculate the total NSR
        nsr = np.sqrt(nsrRand**2 + nsrSys**2)

        return nsr

    def _get_obs_and_errs(
        self,
        mags: np.ndarray,
        bands: list,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return observed magnitudes and magnitude errors."""

        rng = np.random.default_rng(seed)

        # get the NSR for all the galaxies
        nsr = self._get_NSR(mags, bands)

        if self.settings["highSNR"]:
            # in the high SNR approximation, mag err ~ nsr, and we can
            # model errors as Gaussian in magnitude space

            # calculate observed magnitudes
            obsMags = rng.normal(loc=mags, scale=nsr)

            # decorrelate the magnitude errors from the true magnitudes
            obsMagErrs = self._get_NSR(obsMags, bands)

        else:
            # in the more accurate error model, we acknowledge err != nsr,
            # and we model errors as Gaussian in flux space

            # calculate observed magnitudes
            fluxes = 10 ** (mags / -2.5)
            obsFluxes = fluxes * (1 + rng.normal(scale=nsr))
            # obsFluxes = np.clip(obsFluxes, 0, None)
            obsFluxes = np.abs(obsFluxes)
            with np.errstate(divide="ignore"):
                obsMags = -2.5 * np.log10(obsFluxes)

            # decorrelate the magnitude errors from the true magnitudes
            obsFluxNSR = self._get_NSR(obsMags, bands)
            obsMagErrs = 2.5 * np.log10(1 + obsFluxNSR)

        # flag magnitudes beyond magLim as non-detections
        idx = np.where(obsMags >= self.settings["magLim"])
        obsMags[idx] = self.settings["ndFlag"]
        obsMagErrs[idx] = self.settings["ndFlag"]

        return obsMags, obsMagErrs

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Calculate errors for data, and save the results in a pandas DataFrame.
        """

        # get the bands and bandNames present in the data
        bands, bandNames = self._get_bands_and_names(data.columns)

        # get numpy array of magnitudes
        mags = data[bandNames].to_numpy()

        # get observed magnitudes and magnitude errors
        obsMags, obsMagErrs = self._get_obs_and_errs(mags, bands, seed)

        # save the observations in a DataFrame
        obsData = data.copy()
        obsData[bandNames] = obsMags
        obsData[[band + "_err" for band in bandNames]] = obsMagErrs

        # re-order columns so that the error columns come right after the
        # respective magnitudes
        columns = data.columns.tolist()
        for band in bandNames:
            columns.insert(columns.index(band) + 1, band + "_err")
        obsData = obsData[columns]

        return obsData

    def get_limiting_mags(
        self,
        Nsigma: float = 5,
        coadded: bool = False,
    ) -> dict:
        """Return the limiting magnitudes for all bands.
        Note this method essentially reverse engineers the _get_NSR() method
        so that we calculate what magnitude results in NSR = 1/Nsigma.
        (NSR is noise-to-signal ratio; NSR = 1/SNR)
        Parameters
        ----------
        Nsigma : float, default=5
            Sets which limiting magnitude to return, e.g. Nsigma = 1 returns
            the 1-sigma limiting magnitude. In other words, Nsigma is equal
            to the signal-to-noise ratio (SNR) of the limiting magnitudes.
        coadded : bool, default=False
            Whether to return the limiting magnitudes for a single visit,
            or for a coadded image.
        """

        # get the bands and bandNames
        bands, bandNames = self._get_bands_and_names(
            self.settings["bandNames"].values()
        )
        # get the 5-sigma limiting magnitudes for these bands
        m5 = np.array([self._all_m5[band] for band in bands])

        # and the values for gamma
        gamma = np.array([self.settings["gamma"][band] for band in bands])

        # get the number of exposures
        if coadded:
            nVisYr = np.array([self.settings["nVisYr"][band] for band in bands])
            nStackedObs = nVisYr * self.settings["nYrObs"]
        else:
            nStackedObs = 1

        # get the irreducible system error
        if self.settings["highSNR"]:
            nsrSys = self.settings["sigmaSys"]
        else:
            nsrSys = 10 ** (self.settings["sigmaSys"] / 2.5) - 1

        # calculate the square of the random NSR that a single exposure must have
        nsrRandSqSingleExp = (1 / Nsigma**2 - nsrSys**2) * nStackedObs

        # calculate the value of x that corresponds to this NSR
        # note this is just the quadratic equation,
        # applied to NSR^2 = (0.04 - gamma) * x + gamma * x^2
        x = (
            (gamma - 0.04)
            + np.sqrt((gamma - 0.04) ** 2 + 4 * gamma * nsrRandSqSingleExp)
        ) / (2 * gamma)

        # convert x to a limiting magnitude
        limiting_mags = m5 + 2.5 * np.log10(x)

        # return as a dictionary
        return {bandName: mag for bandName, mag in zip(bandNames, limiting_mags)}


class EuclidErrorModel(LSSTErrorModel):
    """Euclid version of the LSST Error Model.

    Parameters from Melissa Graham's 2020 paper:
    https://arxiv.org/abs/2004.07885
    """

    # default settings for Euclid
    default_settings = {
        "bandNames": {  # Euclid bands
            "Y": "Y",
            "J": "J",
            "H": "H",
        },
        "nYrObs": 1.0,  # set obs numbers to 1, bc Euclid is point-and-stare
        "nVisYr": {
            "Y": 1.0,
            "J": 1.0,
            "H": 1.0,
        },
        "m5": {  # hardcode the 5-sigma limiting mags from Melissa's paper
            "Y": 24.0,
            "J": 24.2,
            "H": 23.9,
        },
        "gamma": {  # use values given by Melissa
            "Y": 0.04,
            "J": 0.04,
            "H": 0.04,
        },
        "sigmaSys": 0.005,  # same irreducible error
        "magLim": np.inf,  # dimmest allowed magnitude; dimmer mags set to ndFlag
        "ndFlag": np.inf,  # flag for non-detections (all mags > magLim)
        "highSNR": False,
    }


class RomanErrorModel(LSSTErrorModel):
    """Roman version of the LSST Error Model.

    Parameters from Melissa Graham's 2020 paper:
    https://arxiv.org/abs/2004.07885
    """

    # default settings for Euclid
    default_settings = {
        "bandNames": {  # Euclid bands
            "Y": "Y",
            "J": "J",
            "H": "H",
            "F": "F",
        },
        "nYrObs": 1.0,  # set obs numbers to 1, bc Euclid is point-and-stare
        "nVisYr": {
            "Y": 1.0,
            "J": 1.0,
            "H": 1.0,
            "F": 1.0,
        },
        "m5": {  # hardcode the 5-sigma limiting mags from Melissa's paper
            "Y": 26.9,
            "J": 26.95,
            "H": 26.9,
            "F": 26.25,
        },
        "gamma": {  # use values given by Melissa
            "Y": 0.04,
            "J": 0.04,
            "H": 0.04,
            "F": 0.04,
        },
        "sigmaSys": 0.005,  # same irreducible error
        "magLim": np.inf,  # dimmest allowed magnitude; dimmer mags set to ndFlag
        "ndFlag": np.inf,  # flag for non-detections (all mags > magLim)
        "highSNR": False,
    }
