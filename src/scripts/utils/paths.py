"""Define the paths where everything is saved."""
from showyourwork.paths import user

paths = user()

# catalog paths
paths.train = paths.data / "training_catalogs"
paths.obs = paths.data / "observed_catalogs"
paths.bg = paths.data / "background_catalogs"
paths.fg = paths.data / "foreground_catalogs"

# bandpasses
paths.bandpasses = paths.data / "bandpasses"

# flow ensembles and training losses
paths.models = paths.data / "models"
