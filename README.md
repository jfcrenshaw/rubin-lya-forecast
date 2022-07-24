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

- Troubleshoot the github action.
- Add lya extinction to the observed catalogs
- change processed catalogs back to observed catalogs
- perform the redshift cuts. Plot metrics as a function of survey duration.
- perform du inference. Check for bias. calculate the effective error as a function of survey duration.
- remove the separate NIR section? Integrate that stuff in the catalog section, so that it makes sense to plot everything with LSST+Euclid and LSST+Rubin throughout?
- Write a util that returns the cosmology and other CCL stuff we need in every correlation function
- write a script that calculates the autocorrelation
- write a script that calculates the cross correlation with galaxy clustering. Put the figure for the foreground redshift sample in this section!
- End: Re-train the ensembles for longer. It looks like they could train longer!
