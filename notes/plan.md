# Tree-Method and Forest-Proximity Literature Review

## Summary

- Build a reproducible review containing roughly 50–80 unique works: approximately 20–30 foundational tree-method works and 30–50 forest-proximity/kernel application papers, allowing overlap.
- Create `ObliqueProximitiesCode/notes/proximity_applications_literature.md` as the evidence ledger.
- Add every verified final-table work to `ObliqueProximitiesPaper/main.bib`.
- Download lawful open full text for core proximity/application papers and Tier A method works into `ObliqueProximitiesCode/papers/`. Retain existing filenames; name new files `FirstAuthorYYYYShortTitle.pdf`.
- Do not modify `main.tex` or runtime code during this pass.

## Search and Selection

- Audit the existing 23 BibTeX entries and 11 PDFs first. Reconcile known metadata problems involving Whitmore, Bladen, RF-GAP, CCF, Zheng, Sathe, Leaf-Incidence Kernels, and the ambiguously named `RandomProjectionForests.pdf`.
- Search the method landscape across:
  - CART/C4.5 and foundational tree induction.
  - Bagging, Random Forests, and Extra-Trees.
  - AdaBoost, gradient boosting, XGBoost, LightGBM, and CatBoost.
  - Rotation, projection, and oblique forests.
  - Isolation/anomaly, quantile/survival, causal/generalized/honest, and Bayesian tree families where they meet the evidence threshold.
- Include canonical books, monographs, journal and conference papers, preprints, and canonical software papers, labeling the work type.
- Assign transparent evidence tiers:
  - **Tier A:** canonical/seminal evidence plus strong practical adoption, or exceptional age-adjusted citation influence plus adoption evidence.
  - **Tier B:** at least two strong signals among citations, adoption, historical influence, and direct project relevance.
  - **Tier C:** emerging, specialized, or adjacent work retained for proximity relevance but not described as broadly important.
- Record raw citations and citations per year from the [OpenAlex API](https://docs.openalex.org/) as a dated primary snapshot. Cross-check title identity, citation count, and influential-citation information with the [Semantic Scholar Academic Graph API](https://www.semanticscholar.org/product/api). Preserve source-specific counts rather than merging them into a composite score.
- Treat practical usage as triangulated proxies: maintained major-library support, official or author-linked implementations, package/repository activity, and credible survey or benchmark evidence. Do not claim measured usage frequency unless a source directly reports it.
- Search proximity applications using seed papers, backward references, forward citations, and related-paper expansion for terms covering random-forest proximity/dissimilarity, RF kernels, leaf/path/partition kernels, RF-GAP, terminal-node similarity, and proximity-weighted methods.
- Include papers that define, extend, evaluate, or use a forest-induced similarity downstream. Place methods such as Proximity Forest or Similarity Forest—which consume distances rather than induce the studied proximity—in a clearly marked adjacent-work section.
- Exclude generic kernel papers with no forest-induced similarity, ordinary RF applications that never use proximities, ecological uses of “forest proximity,” and duplicate preprint/final versions.
- Stop after the target range is reached and two consecutive snowballing passes add no new application theme or close comparison and fewer than 5% new eligible papers.

## Catalog, PDFs, and Bibliography

- Structure the Markdown document with:
  1. Search protocol, sources, queries, cutoff date, and inclusion rules.
  2. Tree-method importance table.
  3. Proximity/kernel application taxonomy.
  4. Paper evidence table.
  5. Closest-comparison set, gap analysis, and screening exclusions.
- The method table will record family, method, canonical work, work type, seminal/influential role, evidence tier, dated citation metrics, adoption evidence, proximity relevance, implementation, PDF, and BibTeX key.
- The application evidence table will record paper, relation type, proximity/kernel definition, task, domain/modality, reported evidence and limitations, code URL/status/license, canonical identifier, OA/local-PDF status, citation snapshot, BibTeX key, manuscript relevance, and verification status.
- Use application themes including prediction/smoothing, clustering, embeddings/manifold learning, imputation, anomaly detection, prototypes, explanations/feature importance, multi-view/data integration, time series or structured data, causal matching/effect estimation, and kernel classifiers.
- Mark uncertain claims as `Needs verification`; separate paper-reported facts from inferred project positioning. Identify the 2–5 closest comparison papers for each central theme.
- Download only lawful publisher, repository, accepted-manuscript, author, or arXiv copies. If the final publication is paywalled, cite its canonical metadata while documenting the local preprint/accepted-manuscript relationship. Do not download paywalled-only works.
- Record code as `official`, `author-linked`, `third-party`, or `not found`. A repository counts as open source only when an explicit license is present.
- Add every final-table work to `main.bib`, deduplicating first by DOI and then normalized title. Prefer final publication metadata over preprints and use `firstauthorYYYYslug` for new keys. Preserve existing keys—even imperfect ones—to avoid changing current manuscript citations.
- Normalize entry types and required metadata, protect model-name capitalization, and add DOI/URL fields where verified. Keep download provenance in Markdown rather than BibTeX.

## Validation

- Verify every catalog BibTeX key exists exactly once and every included BibTeX entry has the required fields for its work type.
- Verify local links use `../papers/...`, each downloaded file begins with `%PDF`, and checksums reveal no duplicate content.
- Confirm canonical title, authors, year, venue, DOI/arXiv identifier, OA source, and local-version relationship.
- Audit cross-source citation discrepancies and identity mismatches before assigning Tier A; retain different legitimate index counts with their source and retrieval date.
- Check every external code link, repository license, archival/maintenance status, and access date.
- Compile the manuscript with `latexmk` into a temporary output directory and require no BibTeX errors or undefined citations, without overwriting tracked `main.pdf`.
- Run link/completeness checks and `git diff --check` independently in both repositories.

## Assumptions

- The audience is a technical exploratory ML paper, with literature positioning suitable for later NeurIPS/ICML-style related work.
- Citation and adoption evidence will be snapshotted as of the execution date; initial planning date is July 14, 2026.
- The skipped PDF-breadth decision defaults to downloading core proximity/application papers and Tier A method works, not every cataloged item.
- Existing application and manifold notes seed the taxonomy but remain separate; this review will not duplicate their implementation documentation.