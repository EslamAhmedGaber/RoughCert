# RoughCert v5.0.0

The release accompanies the revised Quantitative Finance manuscript by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady.

* All eight article tables and both supplementary tables are generated from the archived numerical inputs. Bold identifies the RoughCert method or its reported output. A star marks a price separation strictly greater than 0.05 volatility points. Statistical quantile coverage and sampling floors are labelled separately.
* The display audit checks 1,175 printed numbers, directed rounding and threshold markers. The figure audit checks 164 properties across nine vector figures.
* The article, both LaTeX wrappers, bibliography, supplement, proof-verification inputs and package builder accompany the numerical engines. The proof treats the zero-variance kink in the Scarf--Lo majorant explicitly. The abstract and highlights express the largest certified separation as a lower bound.
* The numerical archive includes the 169-bit Arb pricing results, independent adaptive-quadrature/Sobol checks and analytic controls. Reproducing the displays reads the recorded results. The numerical drivers can be run separately.

## Reproduction

With Python 3.12, install the packages in requirements.txt and run python reproduce.py in the repository root. The code-and-data attachment has the same files as this tagged source tree, excluding Git metadata.
