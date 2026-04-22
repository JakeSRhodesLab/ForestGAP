## Benchmark Run Summary

Date: April 21, 2026

Environment:
- All experiment runs used `PYTHONNOUSERSITE=1` in the `generalforest` conda environment.
- Benchmarks used `OpenML-CC18` for classification and `OpenML-CTR23` for regression.
- Requested model families were run wherever the current environment supports them:
  `RandomForest`, `RotationForest`, `HonestForest`, `ObliqueRandomForest`, `PatchObliqueRandomForest`, `ExtraTrees`.

Code changes used for this run:
- `forestgap/datasets.py`: OpenML suite alias/prefix resolution now accepts `OpenML-CC18` and `OpenML-CTR23`.
- `forestgap/core.py`: retained the treeple/sklearn compatibility shim for sample weights.
- `experiments/common.py`: model registry expanded to the requested families, runtime columns added across applications, and `FORESTGAP_N_JOBS` can now override estimator parallelism.
- `experiments/run_local_feature_importance.py`: `HonestForest` is recorded as unsupported for this application because the current treeple build segfaults during fit on the benchmark sweep.

Stored outputs:
- Raw results: `results/raw/*`
- Compiled tables and figures: `results/prediction_identity`, `results/imputation`, `results/outlier_detection`, `results/prototypes`, `results/local_feature_importance`, `results/mds_visualization`
- Runtime rollups: `results/process_runtimes.csv`, `results/process_runtime_overview.csv`

Coverage:
- `prediction_identity`
  Classification: `864/864` rows succeeded.
  Regression: `350/420` rows succeeded; `70` `HonestForest` regression rows were logged as unsupported.
- `imputation`
  Classification: `3396/3411` rows succeeded; `15` rows errored.
  Regression: `1376/1452` rows succeeded; `70` `HonestForest` regression rows were unsupported and `6` rows errored.
- `outlier_detection`
  Classification: `10344/10346` rows succeeded; `2` `HonestForest` rows errored.
  Regression: `2100/2170` rows succeeded; `70` `HonestForest` regression rows were unsupported.
- `prototypes`
  Classification: `862/864` rows succeeded; `2` `HonestForest` rows errored.
  Regression: `420/420` rows were logged as unsupported because prototypes are classifier-only in `forestgap.core`.
- `local_feature_importance`
  Classification: `720/864` rows succeeded; `144` `HonestForest` rows were logged as unsupported because the current build segfaults during fit.
  Regression: `420/420` rows were logged as unsupported because local feature importance is classifier-only in `forestgap.core`.
- `mds_visualization`
  Classification: `860/864` rows succeeded; `4` `HonestForest` rows errored.
  Regression: `350/420` rows succeeded; `70` `HonestForest` regression rows were unsupported.

Key comparative findings:
- Prediction identity:
  Classification identity was effectively exact for `RandomForest`, `ExtraTrees`, `ObliqueRandomForest`, and `PatchObliqueRandomForest` with mean test max absolute differences around `1e-7`.
  `RotationForest` stayed small but materially worse at `3.35e-07` and was much slower.
  `HonestForest` was the outlier: classification mean test max absolute difference was `2.43e-01`, far worse than every other family.
  Regression identity remained small for the supported models; `ObliqueRandomForest` was best at `0.0196`, followed closely by `RandomForest` at `0.0203`.
- Imputation:
  Classification numeric RMSE favored `HonestForest` (`3819.47`) and `PatchObliqueRandomForest` (`4363.71`), while `RandomForest` and `ObliqueRandomForest` were competitive and much faster than `RotationForest`.
  Classification categorical accuracy was highest for `RandomForest` (`0.5938`) and `ExtraTrees` (`0.5926`).
  Regression numeric RMSE favored `ExtraTrees` (`204862.53`), followed by `RandomForest` (`241225.11`).
- Outlier detection:
  Classification average precision was best for `ObliqueRandomForest` (`0.2726`), followed by `RandomForest` (`0.2574`) and `ExtraTrees` (`0.2485`).
  Regression average precision was again best for `ObliqueRandomForest` (`0.2031`).
- Prototypes:
  `RotationForest` produced the strongest prototype-induced classification accuracy (`0.7572`), ahead of `RandomForest` (`0.7379`) and `ObliqueRandomForest` (`0.7375`).
  `PatchObliqueRandomForest` was the weakest prototype model on this benchmark (`0.6466`).
- Local feature importance:
  `PatchObliqueRandomForest` had the highest top-feature win rate (`0.4931`) with `ObliqueRandomForest` close behind (`0.4922`).
  The gap between the best feature and a random feature remained modest overall; even the best models stayed below `0.50` win rate under this lightweight setup.
  `RotationForest` had the largest top-feature probability drop (`0.1213`) but was much slower than the tree-based baselines.
- MDS:
  All supported models produced plots across the capped benchmark sweep.
  `RotationForest` was consistently the slowest MDS model by a wide margin; `ExtraTrees`, `RandomForest`, and the oblique forests stayed under `1` second mean total runtime per plot on both suites.

Runtime notes:
- Mean per-row total runtimes are summarized in `results/process_runtime_overview.csv`.
- The largest average runtime burden came from `prediction_identity` classification (`6.89s` mean per successful row) and `local_feature_importance` classification (`3.96s` mean per successful row), both dominated by `RotationForest`.
- On classification tasks, `RotationForest` was typically the slowest family by one to two orders of magnitude:
  `prediction_identity`: `36.16s` mean row runtime vs about `0.43-0.62s` for most forest baselines.
  `outlier_detection`: `7.12s` vs about `0.13-0.17s`.
  `prototypes`: `6.04s` vs about `0.68-1.04s`.
  `local_feature_importance`: `8.76s` vs about `2.69-2.78s`.
- `HonestForest` was also noticeably slower than `RandomForest`/`ExtraTrees`, but its bigger issue in this sweep was stability and prediction-identity mismatch rather than runtime alone.

Important limitations:
- `HonestForestRegressor` is not available in the current environment, so all regression `HonestForest` rows are unsupported.
- `get_prototypes()` and `local_feature_importance()` remain classifier-only in `forestgap.core`.
- Several small or highly imbalanced CC18 datasets still fail for `HonestForest` because the least-populated class can drop below `2` after the capped split.
- Some imputation rows still fail on datasets where NaNs remain unacceptable to specific estimators after the masking workflow.
