"""Calculate the correlation functions."""
import pickle

import numpy as np
import pyccl as ccl
from utils import (
    LYMAN_WAVELEN,
    Bandpass,
    F_bar,
    build_tracer,
    load_truth_catalog,
    paths,
)

# setup the cosmology
# using parameters from Table 2 of Planck 2018
h = 0.6766
cosmo = ccl.Cosmology(
    Omega_c=0.11933 / h**2,
    Omega_b=0.02242 / h**2,
    h=h,
    A_s=np.exp(3.047) / 1e10,
    n_s=0.9665,
)

# need to call this to fix a bug in ccl 2.4.0 and 2.5.0
cosmo.compute_distances()

# setup a redshift grid
z = np.linspace(0.0, 4.0, 200)


# create the kernels for the different tracers
# --------------------------------------------

# kernel for the Lya tracer
bandpass = Bandpass("u")
nz_F = bandpass.R(LYMAN_WAVELEN * (1 + z)) * F_bar(z)
nz_F /= np.trapz(nz_F, z)

# kernel for the foreground galaxies
true_fg_catalog = load_truth_catalog().query("(redshift > 1.63) & (redshift < 2.36)")
weights, bins = np.histogram(true_fg_catalog.redshift, z, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2
nz_g = np.interp(z, bin_centers, weights)


# build the CCL tracers
# ---------------------

# Lya tracer
b_F = -0.13  # Lya bias
rsd_F = 1.5 * b_F
F_tracer = build_tracer(b_F, rsd_F, (z, nz_F), cosmo)

# foreground galaxy tracer
b_g = 2
rsd_g = 1
g_tracer = build_tracer(b_g, rsd_g, (z, nz_g), cosmo)


# calculate the angular power spectra
# -----------------------------------
ells = np.geomspace(2, 1000, 1000)
Cl_FF = ccl.angular_cl(cosmo, F_tracer, F_tracer, ells)
Cl_Fg = ccl.angular_cl(cosmo, F_tracer, g_tracer, ells)
Cl_gg = ccl.angular_cl(cosmo, g_tracer, g_tracer, ells)


# calculate angular correlation functions
# ---------------------------------------
theta_deg = np.geomspace(1e-3, 1e1, 1000)  # degrees
w_FF = ccl.correlation(cosmo, ells, Cl_FF, theta_deg, type="NN")
w_Fg = ccl.correlation(cosmo, ells, Cl_Fg, theta_deg, type="NN")
w_gg = ccl.correlation(cosmo, ells, Cl_gg, theta_deg, type="NN")


# save the correlations
correlations = {
    "Cl": {
        "ells": ells,
        "FF": Cl_FF,
        "Fg": Cl_Fg,
        "gg": Cl_gg,
    },
    "w": {
        "theta_deg": theta_deg,
        "FF": w_FF,
        "Fg": w_Fg,
        "gg": w_gg,
    },
}
with open(paths.data / "correlations.pkl", "wb") as file:
    pickle.dump(correlations, file)
