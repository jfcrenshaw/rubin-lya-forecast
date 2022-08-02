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

- Write a script that calculates correlation functions. Save their values and x grids in a dictionary. (Maybe also put the SNR in this script too?)
- Write a script that plots the correlation functions.
- Write a script that plots SNR as a function of survey duration.
- Fill out the intro, discussion, and conclusion.
- End: troubleshoot the github action
- End: do a real training set split. Train the FlowEnsembles on a holdout set that I do not estimates redshifts/du for.
- End: Re-train the ensembles for longer. It looks like they could train longer! ALSO extend upper range for redshift models to z=4. Hopefully this removes edge artifacts. And make sure I am training on the right data set. Increase the buffers on u as well.
- End: check that the euclid purity for the bg sample worked out in the end. and the lsst purity for the fg sample.
- End: try generating training sets for each of the flows I train that have the corresponding selection functions already applied to them. See if this results in un-biased inference on $\Delta u$ like the perfect catalog does.
- End: relax the quality cuts so that LSSTY10 doesn't have to pass Euclid cuts, etc. Just use these catalogs to get photo-z metrics and sigma_du, etc. and then use the weighting scheme to predict SNR.
