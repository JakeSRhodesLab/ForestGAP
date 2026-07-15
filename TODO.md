# TODOs

The implementation decision ledger, evidence boundary, valid
estimator–relation combinations, and phased acceptance criteria are in
[`notes/application_roadmap.md`](notes/application_roadmap.md).  The immediate
documentation follow-up is to reconcile the literature ledger's counts,
bibliography, and local PDF provenance.

1. Outline all applications for which proximities have been used.
2. Determine which are crucial to keep for comparisons across tree models.
   1. Side note: for applications, relate to overall (test) performance of the respective models.
3. Code base updates:
   1. Reformulate package, separate classes:
      1. For general (separable) proximity construction.
      2. Adapters for various tree models.
      3. Separate functions/modules for applications.
4. Determine which tree-based models are fundamental and to be included in comparisons.
   1. Separate models for supervised (some only class, some only reg), unsupervised, or other?
5. Build up package modules for each included model.
   1. If built on sklearn, use this version.
   2. If multiple versions (e.g., via github repos), set criteria for selection, then build.
   3. May need to adapt from other language (e.g., CCF on Matlab).
   4. If no implementation exists, build following the sklearn style.
