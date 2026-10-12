# Changelog

## e6692fc - Refactor modeling stack into reusable packages

This update restructures the repository into a reusable three-layer modeling stack for industrial asset behavioral monitoring.

`Model_A` has been refactored into the installable `iabm` package, with a cleaner object-oriented design for preprocessing, training, persistence, and inference. Random Forest and XGBoost now share a unified API, saved artifacts include all required preprocessing metadata, and the end-to-end workflow has been validated for both algorithms. Documentation, i18n, and package metadata were also aligned, and automated tests were added for persistence, CLI workflows, and cross-validation edge cases.

`Model_B` is introduced as the `iabm_behavior` package for behavioral sequence analysis. It supports run extraction, active sequence construction, sequence-word summaries, nominal reference building, anomaly-oriented comparison, DTW-style alignment metrics, and configurable anomaly thresholds. A CLI, tests, and bilingual user-facing messaging are included.

`Model_C` is introduced as the `iabm_semantics` package for semantic interpretation of behavioral sequences. It maps sequence-level patterns into components, operating modes, and working modes, supports optional anomaly enrichment from `Model_B`, and allows explicit semantic rules through JSON configuration. It also includes a CLI, tests, and localization support.

Documentation has been substantially strengthened across the three packages, with English docstrings, improved helper documentation, and targeted inline comments for complex logic. A minimal Sphinx documentation site has been added under `docs/`, and a GitHub Actions workflow now runs package tests and builds the documentation automatically.

Generated XGBoost artifacts and metrics were intentionally left out of this commit so that code, tests, documentation, and CI remain separate from experiment outputs.

### Model D electrical evidence extension

- Added command-response deviation, signed phase residuals and imbalance evidence.
- Added the atlas QIP causal eight-coordinate fourth-order operator and exact
  transient-concentration indicator, with an independent three-qubit simulated
  VQE direction solver and approximation-deficit audit.
- Integrated frozen nominal electrical models into indicator pipelines and
  experiments, weighted scoring and episode descriptors while retaining legacy
  inputs. Added explicit command/reference coverage and scientific tests.

### Event-conditioned frequency-response interpretation

- Added post hoc exact transient-direction interpretation with channel/delay
  taps, directional FIR gain/phase, descriptive band gains and masked projected
  traces with retrospective impulse localization.
- Added optional pipeline/experiment outputs and standalone CSV/JSON/HTML-SVG
  export without changing reliability dimensions, scores or incident decisions.
- Documented standardized-residual units, cadence abstention, replay provenance
  and the distinction from pump transfer functions in English and Sphinx.

### Matched four-level Model D ablation

- Added M0–M3 evidence ablations with common support, nominal-only scaling,
  calibration-only thresholds and exact/VQE matched scoring; VQE replaces the
  exact coordinate rather than adding another reliability dimension.
- Added a frozen seven-coordinate raw-command encoder, vectorized independent
  VQE batches, source/coverage audits, and retrospective projection localization.
- Executed the focused 2022 real-data study and exported English detection,
  fragmentation, weak-reference onset/family, recovery and fixed-target risk
  comparisons, scientific figures and reproducible artifacts.
- Audited the retained 2024–2026 reported-maintenance comparison and absent Word
  sources. Unavailable independent metrics remain unavailable, not zero.
- Documented negative risk results, identical exact/VQE decisions, unstable
  projection peaks and the inherited duration-splitting limitation in Sphinx.

### One multiscale Model D sensitivity

- Added one prespecified 15-min/1-h/6-h/24-h M0–M3 sensitivity on a constant
  15-minute decision grid, with frozen primary electrical models/circuit budget,
  common endpoint support and a fixed risk horizon-purge boundary.
- Kept retrospective reference eligibility separate from the full forecasting
  target registry, preserving comparable cohorts without dropping future labels.
- Executed the real-data sensitivity and exported English reports, scientific
  figures, source/support audits and complete per-arm evidence. Verified all
  41 primary artifact files remain unchanged.
- Documented marked scale dependence, sparse short-window support, identical
  exact/VQE decisions and persistent localization/risk limitations in Sphinx.
