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

- Figure out how to perform the quality cuts in a way that results in max SNR but without bias.
- Break down the error budget (see notes below).
- Try relaxing the redshift cut to increase the completeness and thereby boost the SNR. We just need to test how bad this biases the analysis. We want the optimal trade-off between SNR and bias.
- Test the analysis without the g band. Since it will be impacted, let's just test if we can get by without it. If we can, let's just do that!
- Test the analysis on Delta g instead of Delta u
- reframe the paper as measuring the transmission of the IGM? We can extract the Lya signal given assumptions about modeling each component.
- End: change decrement -> increment, because this is actually increasing the magnitude (I hate how astronomical magnitudes are inverse to luminosity - it makes the language so awkward!)
- End: check that the euclid purity for the bg sample worked out in the end. and the lsst purity for the fg sample.
- End: try generating training sets for each of the flows I train that have the corresponding selection functions already applied to them. See if this results in un-biased inference on $\Delta u$ like the perfect catalog does.

Notes on the error budget:

sample ua ~ p(ua|uah), u ~ p(u|m).

1. Assume true u is known. Then du = ua - u error is from the photometric error on ua, via p(ua|uah). This piece should just be the mean effective u band error.
2. Assume observed ua is perfect. Then du = ua - u error is from the prediction error for u, via p(u|m). If we first assume that observed m is perfect, we can get an intrinsic prediction uncertainty.
3. Then we turn on the errors in m, and see how that increases the error.

Should I do this directly on the u predictions, or on the du predictions?
Also want to make sure these errors all pass the smell test.
Do the correct quantities add in quadrature like you would expect?
