# Finite-Sample Qualification of Approximate Scientific Kernels under QoI Chance Constraints

Research code and recorded data for the manuscript of the same title.

The implementation selects from a fixed menu of approximate scientific kernels
using a Bonferroni-adjusted, one-sided Clopper--Pearson qualification gate and
reports reference execution separately when no candidate is qualified.
Experiments cover FFT, CSR SpMV, periodic stencils, Monte Carlo integration,
a manufactured Poisson problem, and heterogeneous heat conduction.

## Setup

Python 3.10+ is supported by the implementation. The recorded measurements used
Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1, and Matplotlib 3.10.9.
`requirements-lock.txt` records those package versions; use a compatible Python
version (preferably 3.13) to install the recorded environment. Runtime measurements
are specific to the original Windows CPU and thread environment.

```powershell
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python prepare_data.py
```

For a compatible, unpinned environment, use `python -m pip install -e ".[plotting]"`
instead of installing `requirements-lock.txt`. PyTorch is optional.

Large raw traces are losslessly compressed in `data/`. `prepare_data.py` uses only
the Python standard library, checks compressed and restored SHA-256 hashes, and
refuses to overwrite an existing file with different bytes. Allow approximately
1 GB of free disk space for restored observations. No Git LFS setup is needed.

## Validate the existing evidence

Run these commands from the repository root after restoring data:

```powershell
python prepare_data.py --verify-only
python -m unittest discover -s . -p "test_*.py" -q
python validate_publication.py --pair publication-light.csv publication-light-traces.jsonl --pair publication-heavy.csv publication-heavy-traces.jsonl --pair publication-correlated.csv publication-correlated-traces.jsonl
python validate_supplemental_audits.py
python audit_manuscript_evidence.py
```

These commands validate recorded observations or run small implementation checks;
they do not rerun the full timing campaigns. `audit_manuscript_evidence.py` retains
its original filename but the public version checks numerical evidence only.
Derived Student-t confidence endpoints allow a relative tolerance of `1e-10`
and absolute tolerance of `1e-12` across SciPy versions; the remaining confirmation
summary values must match exactly. Original data files are always checked by hash.

## Canonical evidence

| Study | Configuration | Data and summaries |
|---|---|---|
| Broad five-case study | `publication-config.json` | `publication-{light,heavy,correlated}.csv`, matching `-traces.jsonl` and `-summary.json` |
| Fair mean-error policy | `publication-policy-replay.json` | Canonical broad CSVs reflect this replay, including the reference in O* |
| Fresh tight-tolerance Monte Carlo | `confirmatory-mc-config.json` | `confirmatory-mc.csv`, `confirmatory-mc-traces.jsonl`, `confirmatory-mc-summary.json` |
| Legacy 24x24 heat | `heat-experiment-config-full.json` | `heat-audit-full-fair.csv`, legacy `heat-audit-full.jsonl`, fair summary |
| 64x64 heat | `heat-production-config.json` | `heat-production-audit.csv/jsonl`, `heat-production-audit-summary.json` |
| Numerical stability | `stability-audit-config.json` | `stability-audit.csv/jsonl`, `stability-audit-summary.json` |
| Finite-range timing | `scaling-experiment-config.json` | `scaling-audit.csv/json` |
| Post hoc policy analyses | Recorded observations | `sensitivity-audit-delivery.csv/json`, `call-count-audit-fair.csv/json`, `distribution-shift-audit.csv/json` |

The broad configuration retains its original bytes and historical baseline
description. The sidecar replay specification defines the final O* baseline.
The primary schedule has disjoint trial seeds across groups and splits.
The small-grid heat test trace has 1,280 records but only 1,036 unique trial
seeds per candidate and regime; it supports descriptive group statistics,
not pooled independent test-confidence or timing inference. Its validation
seeds are distinct. The larger-grid heat and stability traces are all finite.

## Reproduce figures

```powershell
python regenerate_delivery_figures.py
```

This uses existing observations to recreate the 12 figures listed in `FIGURES.json`
and records input/output hashes in `figures-publication/delivery-figure-provenance.json`.
The numerical audit writes `evidence/numerical-audit.json` for the heat qualification
figure. Outputs are generated locally and are excluded from version control.
Rendering may vary with library and font versions.

## Run new measurements

Write new results to new paths so the archived observations remain identifiable.
The following commands remeasure workloads and will produce new CPU timings.

```powershell
python benchmark.py --kernel all --n-cal 8 --n-val 8 --n-test 16 --seed 7
python run_publication_parallel.py --config publication-config.json --output-dir new-publication-run --workers 1 --keep-parts --resume
python confirmatory_mc.py --config confirmatory-mc-config.json --output-csv new-confirmatory.csv --output-traces new-confirmatory.jsonl --output-summary new-confirmatory-summary.json
python run_heat_sweep.py --config heat-experiment-config-full.json --output new-heat.csv --raw-output new-heat.jsonl
python production_heat_audit.py --config heat-production-config.json --output new-production-heat.csv --raw-output new-production-heat.jsonl
python stability_audit.py --config stability-audit-config.json --output new-stability.csv --raw-output new-stability.jsonl
python scaling_experiment.py --config scaling-experiment-config.json --output new-scaling.json --csv new-scaling.csv
```

The eight-trial smoke example cannot generally qualify a candidate at the
paper's confidence level. The original heat budgets of 32 and 16 validation
trials are also below their 53-success qualification floor. Reproducing those
protocols reproduces this design restriction.

## Statistical and numerical semantics

U selects the cheapest accepted candidate using validation errors and timing.
Acceptance uses a one-sided Clopper--Pearson bound at alpha/M for the fixed
candidate menu. The bound controls erroneous acceptance under iid validation
sampling. If no candidate passes, a reference is executed without certification.
A qualified reference selection and an empty-set fallback are different events;
some legacy trace fields used "fallback" for absence of an accepted approximation.

O* selects by calibration mean error and calibration cost, including the
reference. Held-out tests evaluate fixed choices. Matched costs charge both
policies calibration and validation; policy-specific costs charge O* only
calibration. Reported production cost extrapolates per-call timings to the
declared horizon. It is not an observed complete-workflow time.

The Monte Carlo integral uses uniform draws that are unchanged across external
regime labels. Vector kernels use relative Euclidean error, Monte Carlo uses
absolute integral error, and heat uses relative mean-temperature error.
The heat conductivity is an exponentiated clipped random field; the heavy
and correlated cases are not described as unqualified lognormal laws.
Its discrete direct reference does not certify a continuous physical model.
FFT float16 denotes input quantization followed by a complex64 transform.

SpMV value thresholding preserves the original CSR structure and stored-entry
count. Heat precision labels specify diagonal/right-hand-side quantization;
the sparse operator and iteration updates use float64 arithmetic. The heat
iteration sweep does not compare native low-precision solver arithmetic.

## Release contents and provenance

`DATA_MANIFEST.json` preserves original sizes and SHA-256 hashes for all 38
configuration, observation, and summary files. `RELEASE_MANIFEST.json` identifies
the public snapshot. The numerical kernels and measured observations are unchanged.
The public numerical-audit and figure helpers use `evidence/` and `FIGURES.json`
so reproduction does not depend on manuscript source files. Heat and SpMV wording
and figure labels reflect the actual implemented arithmetic and storage semantics.

See `CITATION.cff` for attribution and `DATA_AVAILABILITY.md` for archive details.
The code retains the project's existing [MIT license](LICENSE).
