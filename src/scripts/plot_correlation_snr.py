"""Plot the SNR of the correlation functions."""
import pickle

import matplotlib.pyplot as plt
from utils import paths, plot_metric

# instantiate paths

with open(paths.data / "correlation_snr.pkl", "rb") as file:
    correlation_snr = pickle.load(file)

# create the figure
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.3, 2.6), constrained_layout=True, dpi=300
)
plot_metric(correlation_snr["FF"], ax1, include_both=True)
plot_metric(correlation_snr["Fg"], ax2, include_both=True)
ax1.set(xticks=[], xlabel=None)
ax1.set_ylabel("SNR($w_{FF}$)", labelpad=13)
ax2.set(ylabel="SNR($w_{Fg}$)")

fig.savefig(paths.figures / "correlation_snr.pdf")
