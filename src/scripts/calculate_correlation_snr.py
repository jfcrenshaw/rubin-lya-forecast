"""Calculate SNR for the correlation functions"""
import pickle

import numpy as np
from utils import paths
from utils.survey_areas import (
    A_LSST,
    A_RATIO_EUCLID,
    A_RATIO_ROMAN,
    SQDEG_PER_STERADIAN,
)

# load all the relevant data saved by other scripts
with open(paths.data / "correlations.pkl", "rb") as file:
    correlations = pickle.load(file)

with open(paths.data / "photoz_metrics_bg.pkl", "rb") as file:
    bg_size = pickle.load(file)["size"]

with open(paths.data / "photoz_metrics_fg.pkl", "rb") as file:
    fg_size = pickle.load(file)["size"]

with open(paths.data / "sigma_du.pkl", "rb") as file:
    sigma_du = pickle.load(file)

# create dictionaries to save the SNRs
snr_wFF = {}
snr_wFg = {}

# loop over the different survey durations
for name in bg_size.keys():
    # get galaxy numbers and error on Delta_F
    if name == "euclid":
        # get the number of galaxies
        N_bg_lsst = (1 - A_RATIO_EUCLID) * bg_size["lsstY10"]
        N_bg_euclid = bg_size["lsstY10+euclid"]
        N_bg = N_bg_lsst + N_bg_euclid  # background galaxies

        N_fg_lsst = (1 - A_RATIO_EUCLID) * fg_size["lsstY10"]
        N_fg_euclid = fg_size["lsstY10+euclid"]
        N_fg = N_fg_lsst + N_fg_euclid  # foreground galaxies

        # calculate the per-galaxy error on Delta_F
        sigma_dF_lsst = np.log(10) / 2.5 * sigma_du["lsstY10"]
        sigma_dF_euclid = np.log(10) / 2.5 * sigma_du["lsstY10+euclid"]
        sigma_dF = np.sqrt(
            N_bg
            / (
                N_bg_lsst * sigma_dF_lsst ** (-2)
                + N_bg_euclid * sigma_dF_euclid ** (-2)
            )
        )

    elif name == "roman":
        # get the number of galaxies
        N_bg_lsst = (1 - A_RATIO_ROMAN) * bg_size["lsstY10"]
        N_bg_roman = bg_size["lsstY10+roman"]
        N_bg = N_bg_lsst + N_bg_roman  # background galaxies

        N_fg_lsst = (1 - A_RATIO_ROMAN) * fg_size["lsstY10"]
        N_fg_roman = fg_size["lsstY10+roman"]
        N_fg = N_fg_lsst + N_fg_roman  # foreground galaxies

        # calculate the per-galaxy error on Delta_F
        sigma_dF_lsst = np.log(10) / 2.5 * sigma_du["lsstY10"]
        sigma_dF_roman = np.log(10) / 2.5 * sigma_du["lsstY10+roman"]
        sigma_dF = np.sqrt(
            N_bg
            / (N_bg_lsst * sigma_dF_lsst ** (-2) + N_bg_roman * sigma_dF_roman ** (-2))
        )

    elif name == "lsstY10+both":
        # get the number of galaxies
        N_bg_lsst = (1 - A_RATIO_EUCLID - A_RATIO_ROMAN) * bg_size["lsstY10"]
        N_bg_euclid = bg_size["lsstY10+euclid"]
        N_bg_roman = bg_size["lsstY10+roman"]
        N_bg = N_bg_lsst + N_bg_euclid + N_bg_roman  # background galaxies

        N_fg_lsst = (1 - A_RATIO_EUCLID - A_RATIO_ROMAN) * fg_size["lsstY10"]
        N_fg_euclid = fg_size["lsstY10+euclid"]
        N_fg_roman = fg_size["lsstY10+roman"]
        N_fg = N_fg_lsst + N_fg_euclid + N_fg_roman  # foreground galaxies

        # calculate the per-galaxy error on Delta_F
        sigma_dF_lsst = np.log(10) / 2.5 * sigma_du["lsstY10"]
        sigma_dF_euclid = np.log(10) / 2.5 * sigma_du["lsstY10+euclid"]
        sigma_dF_roman = np.log(10) / 2.5 * sigma_du["lsstY10+roman"]
        sigma_dF = np.sqrt(
            N_bg
            / (
                N_bg_lsst * sigma_dF_lsst ** (-2)
                + N_bg_euclid * sigma_dF_euclid ** (-2)
                + N_bg_roman * sigma_dF_roman ** (-2)
            )
        )

    else:
        # get the number of galaxies
        N_bg = bg_size[name]  # background galaxies
        N_fg = fg_size[name]  # foreground galaxies

        # calculate the per-galaxy error on Delta_F
        sigma_dF = np.log(10) / 2.5 * sigma_du[name]

    # calculate the shell areas
    theta_rad = np.deg2rad(correlations["w"]["theta_deg"])
    dcosTheta = np.abs(np.diff(np.cos(theta_rad)))
    A_shell = 2 * np.pi * dcosTheta * SQDEG_PER_STERADIAN

    # calculate the number of background galaxies in the shell
    N_shell = A_shell * N_bg / A_LSST

    # calculate errors on wFF
    bgbg_pairs = 1 / 2 * N_bg * N_shell
    sigma_wFF = sigma_dF / np.sqrt(bgbg_pairs)

    # now lets get the values of w at the midpoints of these bins
    # because it is a power law, I will use the geometric mean
    wFF = correlations["w"]["FF"]
    wFF_midpoints = np.sqrt(np.abs(wFF[1:] * wFF[:-1]))

    # calculate the SNR on the amplitude
    snr_wFF[name] = np.sqrt(np.sum(np.square(wFF_midpoints / sigma_wFF)))

    # calculate errors on wFg
    bgfg_pairs = N_fg * N_shell
    sigma_wFg = sigma_dF / np.sqrt(bgfg_pairs)

    # now lets get the values of w at the midpoints of these bins
    # because it is a power law, I will use the geometric mean
    wFg = correlations["w"]["Fg"]
    wFg_midpoints = np.sqrt(np.abs(wFg[1:] * wFg[:-1]))

    # calculate the SNR on the amplitude
    snr_wFg[name] = np.sqrt(np.sum(np.square(wFg_midpoints / sigma_wFg)))

# save the SNRs
with open(paths.data / "correlation_snr.pkl", "wb") as file:
    pickle.dump(
        {
            "FF": snr_wFF,
            "Fg": snr_wFg,
        },
        file,
    )
