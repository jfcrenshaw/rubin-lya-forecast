"""Plot the photo-z metrics."""
import pickle

import matplotlib.pyplot as plt
from showyourwork.paths import user as Paths
from utils import plot_metric

# instantiate the paths
paths = Paths()

# plot the metrics for the background sample
with open(paths.data / "photoz_metrics_bg.pkl", "rb") as file:
    metrics = pickle.load(file)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.6), constrained_layout=True)
plot_metric(metrics["purity"], ax1)
plot_metric(metrics["completeness"], ax2)
ax1.set(xticks=[], xlabel="", ylabel="Purity")
ax2.set(ylabel="Completeness", yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.savefig(paths.figures / "photoz_metrics_bg.pdf")

# plot the metrics for the foreground sample
with open(paths.data / "photoz_metrics_fg.pkl", "rb") as file:
    metrics = pickle.load(file)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.6), constrained_layout=True)
plot_metric(metrics["purity"], ax1)
plot_metric(metrics["completeness"], ax2)
ax1.set(xticks=[], xlabel="", ylabel="Purity")
ax2.set(ylabel="Completeness", yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.savefig(paths.figures / "photoz_metrics_fg.pdf")
