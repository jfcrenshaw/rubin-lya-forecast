"""Download the LSST bandpasses."""
from pathlib import Path

import requests  # type: ignore
from utils import paths

# create the directory where the bandpasses will be saved
Path.mkdir(paths.bandpasses, exist_ok=True)

# the url to the repo where the bandpasses live (with the raw prefix for raw data)
repo_url = "https://raw.githubusercontent.com/lsst/throughputs/main/"

# loop over the requested bands
for band in ["u", "g", "r"]:
    # build the url to the file for this band
    file_url = repo_url + f"baseline/total_{band}.dat"

    # download the file
    file = requests.get(file_url, allow_redirects=True)

    # save the bandpass
    open(paths.bandpasses / f"{band}_bandpass.dat", "wb").write(file.content)
