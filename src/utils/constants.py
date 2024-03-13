"""Save constants used in other scripts."""

# Wavelength of the Lyman-alpha transition
LYMAN_WAVELEN = 1215.67  # Angstroms

# Survey areas in square degrees
AREA_LSST = 18_000
AREA_EUCLID = 8_000
AREA_ROMAN = 2_200

# Fraction of LSST area overlapped by Euclid and Roman
FRAC_OVERLAP_EUCLID = AREA_EUCLID / AREA_LSST
FRAC_OVERLAP_ROMAN = AREA_ROMAN / AREA_LSST

# Angular conversion factor
SQDEG_PER_STERADIAN = 3282.80635
