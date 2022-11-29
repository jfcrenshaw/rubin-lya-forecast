"""Plot the photo-z metrics."""
import pickle

import matplotlib.pyplot as plt
from utils import paths, plot_metric

# plot the metrics for the background sample
with open(paths.data / "photoz_metrics_bg.pkl", "rb") as file:
    metrics = pickle.load(file)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.6), constrained_layout=True)
plot_metric(metrics["purity"], ax1)
plot_metric(metrics["completeness"], ax2)
ax1.set(xticks=[], xlabel="", ylabel="Purity", yticks=[0.94, 0.96, 0.98, 1])
ax2.set(ylabel="Completeness", yticks=[0, 0.1, 0.2, 0.3, 0.4])
fig.savefig(paths.figures / "photoz_metrics_bg.pdf")

# plot the metrics for the foreground sample
with open(paths.data / "photoz_metrics_fg.pkl", "rb") as file:
    metrics = pickle.load(file)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.6), constrained_layout=True)
plot_metric(metrics["purity"], ax1)
plot_metric(metrics["completeness"], ax2)
ax1.set(xticks=[], xlabel="", ylabel="Purity", yticks=[0.94, 0.96, 0.98, 1])
ax2.set(ylabel="Completeness", yticks=[0, 0.1, 0.2, 0.3, 0.4])
fig.savefig(paths.figures / "photoz_metrics_fg.pdf")
