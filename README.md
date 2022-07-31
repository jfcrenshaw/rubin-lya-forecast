<p align="center">
<a href="https://github.com/jfcrenshaw/rubin-lya-forecast/actions/workflows/build.yml">
<img src="https://github.com/jfcrenshaw/rubin-lya-forecast/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<a href="https://github.com/jfcrenshaw/rubin-lya-forecast/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/jfcrenshaw/rubin-lya-forecast/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
</p>

Paper forecasting the SNR for a photometric Lyman-alpha signal detected with the Vera Rubin Observatory.

To Do:

- I will change the way I am calculating the variance of my estimators. I have confirmed using the perfect catalog that I can generate samples of Delta u, and take the mean of those samples for each galaxy as an MLE estimator. I can then take the mean of those estimates as an unbiased estimator of the true Delta u, and the variance of those estimates as the per-galaxy variance on Delta u. However, this leads to a biased estimate for lsstY10+roman. This is probably because I have not accounted for the selection function in my inference pipeline. I think I can fix this by applying the selection function to my training catalogs as well. When I generate observed catalogs, I should get the indices of surviving galaxies, and reserve a subset of those to generate a training catalog. I.e. I have determined that the galaxies with idx pass the quality cuts. Take a random 100k of those from idx, and select the original galaxies from the truth catalog, and store those in a training catalog. I will need to have a separate ensemble trained for each of the LSST years. That is okay! Will just add more corner plots, and training curves lol.
- Write a util that returns the cosmology and other CCL stuff we need in every correlation function
- write a script that calculates the autocorrelation
- write a script that calculates the cross correlation with galaxy clustering. Put the figure for the foreground redshift sample in this section!
- End: troubleshoot the github action
- End: do a real training set split. Train the FlowEnsembles on a holdout set that I do not estimates redshifts/du for!
- End: Re-train the ensembles for longer. It looks like they could train longer! ALSO extend upper range for redshift models to z=4. Hopefully this removes edge artifact. And make sure I am training on the right data set. Increase the buffers on u as well.
- End: check that the euclid purity for the bg sample worked out in the end. and the lsst purity for the fg sample.
- End: Use truncated normals for photometric errors so that we don't get negative fluxes.
