# Forest-Method and Proximity-Application Roadmap

## Summary

Create `ObliqueProximitiesCode/notes/application_roadmap.md` as the decision ledger connecting the existing literature review, repository coverage, implementation availability, and phased TODOs. Add a pointer to it from `TODO.md`.

The documentation pass will not change runtime code or results. It will use a 2026-07-14 evidence cutoff and distinguish:

- estimator families from proximity/relation definitions;
- reusable application primitives from domain case studies;
- established applications from conceived research directions;
- `reuse`, `integrate`, `modernize/port`, `build`, `specialty`, and `reference-only` methods.

## Roadmap Content

### Method decision matrix

Give every foundational method an individual row with task support, geometry, Python/sklearn status, alternative-language implementation, leaf/OOB APIs, license, maintenance, scientifically valid relations, and final decision.

Use these decisions:

| Decision | Methods |
|---|---|
| Reuse and stabilize | sklearn RandomForest and ExtraTrees; aeon RotationForest; treeple ObliqueRF/SPORF; HonestForest with `honest_gap`; Isolation/Extended Isolation; native unsupervised axis-aligned and oblique forests |
| Restricted roles | PatchOblique for spatial/image structure; ExtraOblique as an ablation; Isolation methods only with native path scores as required baselines |
| Integrate next | `quantile-forest`; scikit-survival RandomSurvivalForest; Shi–Horvath synthetic-null unsupervised RF meta-estimator |
| Modernize/build next | Canonical Correlation Forest: rehabilitate the Apache-licensed Python port and validate classification/regression behavior against the author’s MATLAB implementation ([Python port](https://github.com/plai-group/ccfs-python), [MATLAB reference](https://github.com/twgr/ccfs)) |
| Add relation constructors | KeRF/leaf-mass normalization, path/branch similarity, random-height partition kernels, alongside existing original/OOB/RF-GAP/Honest-GAP |
| Specialty tier | XGBoost as the initial boosted-tree representative; LightGBM and CatBoost only for scale/categorical hypotheses; PF-GAP for structured objects; R `grf` with Python causal comparators; PyMC-BART after defining posterior co-clustering semantics |
| Reference/defer | CART, ID3, C4.5, bagging, random-subspace, AdaBoost/classic GBM ancestors, Bayesian CART, OC1, and archived Menze `obliqueRF`, unless later hypotheses require their unique behavior |

Record that treeple is an optional research dependency under a noncommercial license and must not be vendored; preserve explicit unsupported rows when unavailable. Treeple currently covers oblique, honest, unsupervised, and extended-isolation families through a sklearn-like API ([treeple documentation](https://docs.neurodata.io/treeple/v0.9/index.html)). Quantile and survival integrations should use their existing sklearn-style leaf interfaces ([quantile-forest](https://zillow.github.io/quantile-forest/), [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html)).

### Application decision matrix

Give each distinct use case—not each domain paper—a row containing evidence, required relation type, supervision/leakage risk, data requirements, baselines, metrics, current API/runner/test status, priority, and acceptance criterion.

- Tier 0, correct and complete current work: prediction identity, imputation, anomaly/novelty/mislabel auditing, local feature importance, prototypes/retrieval, and manifold visualization.
- Tier 1, broadly reusable additions: kernel SVM/KRR/kNN prediction, clustering, conditional distributions and quantiles, conformal/selective prediction, batch active learning, training-instance attribution, retrieval, and proximity-neighborhood feature selection.
- Tier 2, specialized data/outcomes: personalized local models, causal matching and balancing, survival stratification, process/change monitoring, multiview fusion/alignment, time series, and arbitrary structured objects.
- Tier 3, higher-overhead capstones: proximity graphs/GNNs, autoencoding/denoising/compression, and representative open domain studies in biology, medicine, finance, astronomy, or remote sensing.
- Conceived/deferred: label propagation, proximity-based resampling, and fairness-neighborhood auditing until direct evidence and core benchmarks are stronger.

Correct the current evidence boundary by adding the earlier proximity-based conformal work from 2009, active learning, process monitoring, survival clustering, causal balancing, personalized modeling, and feature selection. Representative primary evidence includes [Yang et al. conformal RF](https://link.springer.com/article/10.1186/1471-2105-10-S1-S22), [Kee et al. active learning](https://www.sciencedirect.com/science/article/abs/pii/S0020025518303700), and [random-survival-forest clustering](https://www.nature.com/articles/s41598-021-92072-8). Mark the existing 50-work ledger as a strong snapshot, not an exhaustive final set, and add a TODO to reconcile its counts, bibliography, and PDFs.

## Repository and Experimental Roadmap

1. **Evidence freeze**
   - Finish targeted backward/forward searches from the newly found application seeds until two passes add no new reusable theme.
   - Complete both decision matrices and verify URLs, versions, licenses, officiality, and alternative-language oracles.
   - Acceptance: every existing method/application and every newly found theme has one status and rationale.

2. **Correctness and provenance**
   - Replace the legacy MDS conversion with the tested shared conversion where implicit zero proximity becomes maximal distance; quarantine existing MDS conclusions pending rerun.
   - Re-evaluate HonestForest with `honest_gap`; label the older ordinary RF-GAP mismatch as a legacy result.
   - Preserve old outputs as versioned legacy artifacts, then rerun the redesigned imputation, prototype, prediction-identity, and visualization smoke suites.
   - Record code revision, configuration, dataset/split identifiers, runtime, memory, and failure status.

3. **Architecture**
   - Preserve the public `with_sparse_proximity`, `get_proximities`, and `prox_extend` APIs.
   - Extract an adapter-capability contract covering leaf assignment, estimator inventory, task type, in-bag multiplicities, OOB/query masks, and tree contribution weights.
   - Separate estimator adapters, relation constructors, and downstream application modules.
   - Reject scientifically invalid combinations explicitly—for example, ordinary RF-GAP semantics on boosted, Bayesian, or isolation ensembles without the required bootstrap structure.

4. **Core and specialty expansion**
   - Implement CCF and the missing relation definitions first, followed by quantile/survival adapters and Tier 1 applications.
   - Add causal, time-series, multiview, graph, and autoencoding tracks only after their data-specific leakage and evaluation protocols are defined.
   - Use R/MATLAB implementations as conformance oracles; production experiment paths remain Python/sklearn-style where feasible.

5. **Cross-method study**
   - Analyze only valid cells of `estimator family × relation definition × application × data regime`.
   - Reuse identical splits and fitted parent models when comparing relations/applications.
   - Record parent predictive performance, downstream utility, proximity sparsity/effective-neighbor count, asymmetry/PSD diagnostics, runtime, peak memory, and failures.
   - Stratify data by axis-aligned versus rotated/oblique signal, \(p/n\), feature correlation, mixed/categorical data, imbalance, missingness, cluster structure, censoring, temporal dependence, multiview structure, and non-vector objects.
   - Estimate within-dataset utility deltas and method–data interactions rather than comparing pooled raw means.

## Validation and Assumptions

- The note must contain source-linked method and application matrices plus a phased TODO table with dependencies, blockers, status, and measurable acceptance criteria.
- Primary-source links and exact implementation identity must be checked; “not found” remains a bounded search result.
- Documentation assertions must agree with the current registry, proximity modes, tests, and runners.
- Future implementation gates include adapter conformance, sparse/dense parity, relation-specific invariants, leakage checks, deterministic shared splits, absent-dependency behavior, CLI smoke tests, and runtime/memory capture.
- “All methods” defaults to all scientifically distinct families and task regimes; redundant historical ancestors remain documented rather than mandatory implementations.
- The default implementation policy is Python-first with R/MATLAB validation oracles.
- Applications default to reusable primitives plus a small set of open representative case studies, not one reproduction per domain paper.
