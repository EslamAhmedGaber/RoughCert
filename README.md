# RoughCert

Research code and synthetic benchmark data for *RoughCert: Machine-Verified VIX Derivative Bounds and Conditional Risk Certificates under Rough Volatility*, by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady. Version `v3.0.0` accompanies the audited Quantitative Finance manuscript.

## Reproduce the published displays

Use Python 3.12 in a virtual environment. From the repository root, run

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script rebuilds the eight main tables, two supplementary tables and nine vector figures from the recorded JSON inputs. It verifies 1,151 printed numeric entries and 164 properties of the figures. Tables and figures appear in `manuscript/tables/` and `manuscript/figures/`. Machine-readable checks appear in `audit/`. The display rebuild works offline after installing the dependencies.

## Numerical experiments

| Location | Contents |
|---|---|
| `params_manifest.json` | Model parameters, contract conventions, volatility units and analytic controls |
| `scripts/arb_vix_core.py`, `scripts/arb_vix_options.py` | Arb pricing engines for futures, options and Black-76 implied-volatility intervals |
| `scripts/run_week1_gates.py`, `scripts/run_vix_futures_full_suite.py`, `scripts/run_vix_options_smile_suite.py` | Analytic controls, futures and option experiments |
| `scripts/refinement.py`, `scripts/curved_expansions.py`, `scripts/v3_comparisons.py` | Stress-grid refinement, curved-contract intervals and method comparisons |
| `scripts/factor_thresholds.py`, `scripts/band_refine.py`, `scripts/v3_factor_extension.py` | One-tick and fixed-band factor decisions and finite-model inputs through 19 factors |
| `results/` | Recorded synthetic benchmarks, factor coefficients, quadrature comparisons, statistical controls and the legacy inputs used by the display builders |
| `legacy/04_CODE/`, `legacy/05_EXPERIMENTS/` | Historical Gaussian-functional and risk code, signed-atom controls, experiment drivers and their companion inputs |

For a short arithmetic check, run `python scripts/adversarial_checks.py`. For the external comparison drivers (including LAB, BDG and the Bayer–Breneis optimiser), run `python fetch_references.py` first. It obtains the specific upstream files listed in `upstream_sources.json` and checks each SHA-256 digest before use. Upstream licenses and attribution remain with their respective authors.

To repeat a numerical experiment, run its driver from the repository root in a separate checkout. Drivers can overwrite their recorded JSON outputs. Historical drivers write beside their own source under `legacy/05_EXPERIMENTS/`, while the v3 display scripts read the frozen inputs under `results/legacy_inputs/`. The two directories let readers compare a rerun with the deposited data. Elapsed times depend on the machine. Arb containment, floating-point quadrature and statistical coverage are identified separately in the paper.
