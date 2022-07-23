"""Download the LSST bandpasses."""
from pathlib import Path

import requests
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# create the directory where the bandpasses will be saved
bandpass_dir = paths.data / "bandpasses"
Path.mkdir(bandpass_dir, exist_ok=True)

# the url to the repo where the bandpasses live (with the raw prefix for raw data)
repo_url = "https://raw.githubusercontent.com/lsst/throughputs/main/"

# loop over the requested bands
for band in ["u", "g", "r"]:
    # build the url to the file for this band
    file_url = repo_url + f"baseline/total_{band}.dat"

    # download the file
    file = requests.get(file_url, allow_redirects=True)

    # save the bandpass
    open(bandpass_dir / f"{band}_bandpass.dat", "wb").write(file.content)
