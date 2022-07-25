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

- perform du inference. Check for bias. calculate the effective error as a function of survey duration.
- Write a util that returns the cosmology and other CCL stuff we need in every correlation function
- write a script that calculates the autocorrelation
- write a script that calculates the cross correlation with galaxy clustering. Put the figure for the foreground redshift sample in this section!
- End: troubleshoot the github action
- End: do a real training set split. Train the FlowEnsembles on a holdout set that I do not estimates redshifts/du for!
- End: Re-train the ensembles for longer. It looks like they could train longer! ALSO extend upper range for redshift models to z=4. Hopefully this removes edge artifact. And make sure I am training on the right data set. Increase the buffers on u as well.
- End: check that the euclid purity for the bg sample worked out in the end. and the lsst purity for the fg sample.
