"""Save variables for use in the tex file."""
import pickle

from showyourwork.paths import user as Paths
from utils import lya_decrement
from utils.sample_with_errors import m_samples, zu_samples

# instantiate the paths
paths = Paths()

# save the sample numbers
open(paths.output / "m_samples.txt", "w").write(f"{m_samples}")
open(paths.output / "zu_samples.txt", "w").write(f"{zu_samples}")

# save photo-z metric numbers
with open(paths.data / "photoz_metrics_bg.pkl", "rb") as file:
    metrics = pickle.load(file)
    open(paths.output / "bg_purity_y1.txt", "w").write(f"{metrics['purity'][1]:.2f}")
    open(paths.output / "bg_purity_y10.txt", "w").write(f"{metrics['purity'][10]:.2f}")
    open(paths.output / "bg_completeness_y1.txt", "w").write(
        f"{metrics['completeness'][1]:.2f}"
    )
    open(paths.output / "bg_completeness_y10.txt", "w").write(
        f"{metrics['completeness'][10]:.2f}"
    )
    open(paths.output / "bg_completeness_y10+euclid.txt", "w").write(
        f"{metrics['completeness']['euclid']:.2f}"
    )
    open(paths.output / "bg_completeness_y10+roman.txt", "w").write(
        f"{metrics['completeness']['roman']:.2f}"
    )
    open(paths.output / "bg_size_y1.txt", "w").write(f"{metrics['size'][1] / 1e6:.0f}")
    open(paths.output / "bg_size_y10.txt", "w").write(
        f"{metrics['size'][10] / 1e6:.0f}"
    )
    open(paths.output / "bg_size_y10+euclid+roman.txt", "w").write(
        f"{metrics['size']['Y10+euclid+roman'] / 1e6:.0f}"
    )

# save sigma_du SNR
with open(paths.data / "sigma_du.pkl", "rb") as file:
    noise = pickle.load(file)[10]  # sigma_du for LSST Y10
    signal = lya_decrement(3, "u", 0)
    snr = signal / noise
    open(paths.output / "snr_y10.txt", "w").write(f"{snr:.1f}")
