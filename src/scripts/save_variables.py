"""Save variables for use in the tex file."""
import pickle

from utils import lya_decrement, paths, survey_areas
from utils.sample_with_errors import m_samples, zu_samples

# save the sample numbers
open(paths.output / "m_samples.txt", "w").write(f"{m_samples}")
open(paths.output / "zu_samples.txt", "w").write(f"{zu_samples}")

# save photo-z metric numbers
with open(paths.data / "photoz_metrics_bg.pkl", "rb") as file:
    metrics = pickle.load(file)
    open(paths.output / "bg_purity_y1.txt", "w").write(f"{metrics['purity']['lsstY1']:.2f}")
    open(paths.output / "bg_purity_y10.txt", "w").write(f"{metrics['purity']['lsstY10']:.2f}")
    open(paths.output / "bg_completeness_y1.txt", "w").write(
        f"{metrics['completeness']['lsstY1']:.2f}"
    )
    open(paths.output / "bg_completeness_y10.txt", "w").write(
        f"{metrics['completeness']['lsstY10']:.2f}"
    )
    open(paths.output / "bg_completeness_y10+euclid.txt", "w").write(
        f"{metrics['completeness']['lsstY10+euclid']:.2f}"
    )
    open(paths.output / "bg_completeness_y10+roman.txt", "w").write(
        f"{metrics['completeness']['lsstY10+roman']:.2f}"
    )
    open(paths.output / "bg_size_y1.txt", "w").write(f"{metrics['size']['lsstY1'] / 1e6:.0f}")
    open(paths.output / "bg_size_y10.txt", "w").write(
        f"{metrics['size']['lsstY10'] / 1e6:.0f}"
    )
    open(paths.output / "bg_size_y10+euclid+roman.txt", "w").write(
        f"{metrics['size']['lsstY10+both'] / 1e6:.0f}"
    )

with open(paths.data / "photoz_metrics_fg.pkl", "rb") as file:
    metrics = pickle.load(file)
    open(paths.output / "fg_size_y1.txt", "w").write(f"{metrics['size']['lsstY1'] / 1e6:.0f}")
    open(paths.output / "fg_size_y10.txt", "w").write(
        f"{metrics['size']['lsstY10'] / 1e6:.0f}"
    )
    open(paths.output / "fg_size_y10+euclid+roman.txt", "w").write(
        f"{metrics['size']['lsstY10+both'] / 1e6:.0f}"
    )

# save sigma_du SNR
with open(paths.data / "sigma_du.pkl", "rb") as file:
    signal = lya_decrement(3, "u", 0)
    noise = pickle.load(file)
    open(paths.output / "snr_y10.txt", "w").write(f"{signal / noise['lsstY10']:.1f}")

# save the correlation SNRs
with open(paths.data / "correlation_snr.pkl", "rb") as file:
    snr = pickle.load(file)
    open(paths.output / "snr_wff_y1.txt", "w").write(f"{snr['FF']['lsstY1']:.1f}")
    open(paths.output / "snr_wff_y10.txt", "w").write(f"{snr['FF']['lsstY10']:.1f}")
    open(paths.output / "snr_wff_y10+euclid+roman.txt", "w").write(
        f"{snr['FF']['lsstY10+both']:.0f}"
    )
    open(paths.output / "snr_wfg_y1.txt", "w").write(f"{snr['Fg']['lsstY1']:.1f}")
    open(paths.output / "snr_wfg_y10.txt", "w").write(f"{snr['Fg']['lsstY10']:.0f}")
    open(paths.output / "snr_wfg_y10+euclid+roman.txt", "w").write(
        f"{snr['Fg']['lsstY10+both']:.0f}"
    )

# save the survey overlaps
open(paths.output / "euclid_overlap.txt", "w").write(
    f"{100 * survey_areas.A_RATIO_EUCLID:.0f}"
)
open(paths.output / "roman_overlap.txt", "w").write(
    f"{100 * survey_areas.A_RATIO_ROMAN:.0f}"
)
