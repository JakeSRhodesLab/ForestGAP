# Proximity-Based Applications in `forestgap/core.py`

This note summarizes the proximity-based applications implemented in [forestgap/core.py](forestgap/core.py), with notation aligned as closely as possible to [RF-GAP](papers/RF-GAP.pdf) and [LeafIncidenceKernels.pdf](papers/LeafIncidenceKernels.pdf), together with the leaf-incidence / kernel factorization used by the sparse implementation. The main goal is to make the downstream methods explicit enough that we can later turn them into controlled validation experiments across multiple forest-type models.

The four primary applications are:

1. proximity-weighted prediction (`prox_predict` and `prox_extend`);
2. missing-data imputation (`impute`);
3. outlier detection (`outlier_scores`);
4. local feature importance (`local_feature_importance`).

The same sparse proximity engine also supports visualization (`plot_mds`) and class prototypes (`get_prototypes`), but those are secondary here.

## 1. Core notation

Let the training data be

$$
\mathcal D = \{(x_i, y_i)\}_{i=1}^n, \qquad x_i \in \mathbb R^p.
$$

Let the forest contain $T$ trees, indexed by $t \in \{1,\dots,T\}$.

For tree $t$:

- $B_t$ is the bootstrap multiset of training indices used to fit the tree.
- $O_t = \{1,\dots,n\} \setminus B_t$ is the out-of-bag index set.
- $c_j^{(t)}$ is the multiplicity of training sample $j$ in $B_t$.
- $\ell_t(x)$ is the terminal leaf reached by point $x$ in tree $t$.
- $J_t(x) = \{j : \ell_t(x_j) = \ell_t(x)\}$ is the set of training points sharing the leaf with $x$.
- $J_t^{\mathrm{IB}}(x) = J_t(x) \cap B_t$ is the corresponding in-bag set.
- $\mathcal J_t^{\mathrm{IB}}(x)$ is the multiset version of $J_t^{\mathrm{IB}}(x)$, so repeated in-bag draws are counted with multiplicity.
- $m_t(x) = |\mathcal J_t^{\mathrm{IB}}(x)| = \sum_{k \in J_t(x)} c_k^{(t)}$ is the in-bag leaf mass.

For a training point $x_i$, define the OOB-tree set

$$
\mathcal S_i = \{t : i \in O_t\}, \qquad S_i = |\mathcal S_i|.
$$

For the code path, it is also useful to index leaves explicitly. Let tree $t$ have $L_t$ observed leaves on the fitted data, and let

$$
\phi_t(x) \in \{0,1\}^{L_t}
$$

be the one-hot leaf-incidence vector, with a 1 at the leaf reached by $x$. Stacking trees gives a global leaf-incidence feature map

$$
\phi(x) = [\phi_1(x),\dots,\phi_T(x)] \in \{0,1\}^{L}, \qquad L = \sum_{t=1}^T L_t.
$$

The sparse implementation in `core.py` constructs proximities as

$$
P = QW^\top,
$$

where:

- rows of $Q$ encode how a query point distributes mass across leaves;
- rows of $W$ encode how each training point contributes label mass through leaves.

This is the key leaf-incidence factorization behind `get_proximities()` and `prox_extend()`.

## 2. Proximity definitions implemented in `core.py`

### 2.1 Original Breiman proximity

The original RF proximity is

$$
P_{ij}^{\mathrm{orig}}
= \frac{1}{T}\sum_{t=1}^T \mathbf{1}\{\ell_t(x_i)=\ell_t(x_j)\}.
$$

In the sparse factorization, this is represented with

$$
Q^{\mathrm{orig}} = W^{\mathrm{orig}} = \frac{1}{\sqrt T}\Phi,
$$

where $\Phi$ is the training leaf-incidence matrix.

This proximity is symmetric, dense in concept, and tends to over-separate classes because it does not distinguish between in-bag and OOB roles.

### 2.2 OOB proximity

The code supports `prox_method="oob"`, but the implementation is best viewed as a separable sparse approximation rather than a literal reimplementation of the classical randomForest package formula.

If

$$
M_i = \sum_{t=1}^T \mathbf{1}\{i \in O_t\},
$$

then the code assigns each OOB leaf incidence a weight $\sqrt T / M_i$, so that

$$
W^{\mathrm{oob}}_{i,\ell}
= \frac{\sqrt T}{M_i}\mathbf{1}\{i \in O_t,\ \ell_t(x_i)=\ell\},
\qquad
Q^{\mathrm{oob}} = W^{\mathrm{oob}}.
$$

This yields

$$
P^{\mathrm{oob}} = Q^{\mathrm{oob}}(W^{\mathrm{oob}})^\top,
$$

followed by clipping entries above 1. This mode is useful as a baseline, but it is not the mode that enjoys the exact prediction identity.

### 2.3 RF-GAP proximity

This is the main construction.

For training-to-training proximities, the RF-GAP weight from row $i$ to column $j$ is

$$
P_{ij}^{\mathrm{rfgap}}
=
\frac{1}{S_i}
\sum_{t \in \mathcal S_i}
\frac{c_j^{(t)}}{m_t(x_i)}
\mathbf{1}\{j \in J_t^{\mathrm{IB}}(x_i)\}.
$$

Equivalently, using the leaf-incidence factorization:

$$
Q_{i,\ell}^{\mathrm{rfgap}}
=
\frac{1}{S_i}
\sum_{t \in \mathcal S_i}
\mathbf{1}\{\ell=\ell_t(x_i)\},
$$

and

$$
W_{j,\ell}^{\mathrm{rfgap}}
=
\frac{c_j^{(t)}}{m_\ell^{(t)}}
\mathbf{1}\{j \in B_t,\ \ell_t(x_j)=\ell\},
$$

where

$$
m_\ell^{(t)} = \sum_{k:\ell_t(x_k)=\ell} c_k^{(t)}.
$$

For a new test point $x_\ast$, every tree treats it as externally OOB, so

$$
P_{\ast j}^{\mathrm{rfgap}}
=
\frac{1}{T}
\sum_{t=1}^T
\frac{c_j^{(t)}}{m_t(x_\ast)}
\mathbf{1}\{j \in J_t^{\mathrm{IB}}(x_\ast)\}.
$$

Important structural properties:

- $P^{\mathrm{rfgap}}$ is generally asymmetric.
- Under the default unsymmetrized construction, each row is a nonnegative weight vector and sums to 1 when the relevant leaves are observed.
- `core.py` exposes optional symmetrization (`symm_mode="arithmetic"` or `"geometric"`) and optional diagonal augmentation (`non_zero_diagonal=True`) for similarity-style applications, but these should be off when testing prediction identities.

## 3. Proximity-weighted prediction

### 3.1 Regression identity

Let $y = (y_1,\dots,y_n)^\top$. The RF-GAP regression predictor is

$$
\hat f_{\mathrm{prox}}(x)
=
\sum_{j=1}^n P^{\mathrm{rfgap}}(x,j)y_j.
$$

For training rows, using OOB weights, this matches the forest OOB regression prediction. For test rows, using `prox_extend(X_test)`, it matches the ordinary forest prediction.

This is exactly what `prox_predict` implements in the regressor case:

$$
\hat y_{\mathrm{prox}} =
P y.
$$

### 3.2 Classification identity

Let $C$ be the number of classes, let $\mathcal C = \{c_1,\dots,c_C\}$, and define the one-hot class matrix

$$
Y \in \{0,1\}^{n \times C}, \qquad
Y_{jk} = \mathbf{1}\{y_j = c_k\}.
$$

Then the RF-GAP class-mass predictor is

$$
\hat \Pi_{\mathrm{prox}}(x) = P^{\mathrm{rfgap}}(x,\cdot)\,Y.
$$

For a test point, $\hat \Pi_{\mathrm{prox}}(x)$ matches `predict_proba(x)`. For a training row, the unsymmetrized OOB RF-GAP predictor matches the forest OOB class-probability vector, i.e. `oob_decision_function_`.

In `core.py`, `prox_predict` implements exactly this matrix product. For the training/OOB case it additionally row-normalizes the class masses before returning, which is harmless under the exact RF-GAP assumptions and stabilizes edge cases.

### 3.3 What should be validated experimentally

The most important prediction checks are:

1. `prox_predict(X_test)` vs. `predict_proba(X_test)` for classifiers and `predict(X_test)` for regressors.
2. `prox_predict()` vs. `oob_decision_function_` / `oob_prediction_`.
3. `prox_extend(X_test) @ Y` vs. forest predictions.
4. The effect of turning on `symm_mode`, `non_zero_diagonal`, or `max_normalize`; these options are useful for similarity applications but should break the exact RF-GAP prediction identity.

## 4. Imputation

### 4.1 Code-level computation

Let $X \in \mathbb R^{n \times p}$ have missing entries. For feature $m$, define the observed-reference indicator

$$
o_j^{(m)} = \mathbf{1}\{\mathrm{obs}(X_{jm})\}.
$$

Given a proximity matrix $P$ from query rows to reference rows, `core.py` forms a feature-specific normalized weight matrix

$$
w_{ij}^{(m)}
=
\frac{P_{ij}\,o_j^{(m)}}
{\sum_{k} P_{ik}\,o_k^{(m)}},
$$

with the convention that zero row sums are replaced by 1 to avoid division by zero.

Then the imputation rule is:

- continuous feature:

$$
\tilde X_{im}
=
\sum_j w_{ij}^{(m)} X_{jm};
$$

- categorical feature:

$$
\tilde X_{im}
=
\arg\max_{a \in \mathcal A_m}
\sum_{j: X_{jm}=a} w_{ij}^{(m)}.
$$

Here $\mathcal A_m$ is the set of observed categories in column $m$.

### 4.2 Training vs. test imputation

The implementation has two passes:

1. training imputation:
   $P = P_{\mathrm{train,train}}$, computed by `get_proximities()`;
2. optional test imputation:
   $P = P_{\mathrm{test,train}}$, computed by `prox_extend(X_{\mathrm{test}}^{\mathrm{init}})`.

For the test pass, `prox_extend` cannot route NaNs through the forest, so the code first creates a rough-filled copy $X_{\mathrm{test}}^{\mathrm{init}}$ using one of:

- column mean;
- column median;
- `KNNImputer`.

Crucially, those rough fills are only used to obtain leaf assignments for the test queries. The actual imputed values are then computed against the training reference matrix, but feature-wise donor weights are still masked by the original training missingness pattern. So, for a given feature $m$, only training rows that were originally observed in column $m$ contribute donor mass.

### 4.3 Difference from the RF-GAP paper algorithm

The RF-GAP paper describes an outer iterative scheme:

1. initialize missing values;
2. train a forest;
3. compute proximities;
4. re-impute from proximities;
5. repeat.

`core.py` currently implements only the inner proximity-weighted replacement step on a fixed fitted forest. In other words:

- the forest is assumed to have already been fitted, often on a rough-filled training matrix;
- `impute` itself does not refit the forest and does not iterate to convergence.
- even in the test pass, donor values for a feature come only from originally observed training entries for that feature, not from recursively reusing newly imputed donors.

So the current method is best described as:

single-pass proximity smoothing on a fixed forest geometry.

That distinction matters for experiments. If we want to match the paper’s iterative imputation algorithm exactly, we will need an outer loop that alternates `fit` and `impute`.

### 4.4 What should be validated experimentally

Recommended imputation checks:

1. single-pass RF-GAP vs. original vs. OOB proximity, holding the fitted forest fixed;
2. single-pass vs. iterative refit-and-impute;
3. train-only missingness vs. train+test missingness;
4. continuous vs. categorical columns;
5. sensitivity to the rough-fill strategy used before `prox_extend`.

## 5. Outlier detection

### 5.1 Breiman-style score

The classical proximity-based outlier logic is: points with small within-class proximity mass are outliers.

`core.py` first recomputes proximities with:

- `non_zero_diagonal = False`,
- `symm_mode = None`,

so self-proximity does not artificially inflate the score.

For supervised labels $g_i$, define the within-class index set

$$
G_i = \{j : g_j = g_i\}.
$$

The code’s Breiman-style raw score is

$$
r_i^{\mathrm{Breiman}}
=
\frac{|G_i|}{\sum_{j \in G_i} P_{ij}^2}.
$$

If labels are omitted, the global version is

$$
r_i^{\mathrm{Breiman}}
=
\frac{n}{\sum_{j=1}^n P_{ij}^2}.
$$

The returned score is then MAD-standardized:

$$
s_i
=
\frac{|r_i - \mathrm{median}(r)|}
{\mathrm{MAD}(r)},
$$

either within class (supervised case) or globally (unsupervised case).

### 5.2 ENE variant

The code also exposes `method="ene"` based on the effective number of neighbors

$$
\mathrm{ENE}_i
=
\frac{\left(\sum_j P_{ij}\right)^2}
{\sum_j P_{ij}^2}.
$$

Then

$$
r_i^{\mathrm{ENE}}
=
\frac{N_i}{\mathrm{ENE}_i},
$$

where $N_i = |G_i|$ in the supervised case and $N_i=n$ in the label-free case.

This variant reduces the confounding effect of many weak neighbors and is especially relevant for IsolationForest-style proximities, where dense regions can accumulate many small similarities.

### 5.3 Scaling options

After MAD standardization, the code can optionally apply:

- `scaling="normalize"`:
  min-max scale to $[0,1]$;
- `scaling="log"`:
  $\log(1+s_i)$.

### 5.4 What should be validated experimentally

Recommended outlier checks:

1. supervised within-class score vs. label-free score;
2. Breiman raw score vs. ENE raw score;
3. behavior under asymmetric RF-GAP vs. symmetrized variants;
4. whether known corruptions or mislabeled points rank near the top;
5. whether the score aligns with MDS geometry from the same proximity matrix.

## 6. Local feature importance

### 6.1 Relation to the Whitmore et al. idea

The implementation is closest in spirit to [Explicating feature contribution using Random Forest proximity distances](papers/ExplicatingFeature.pdf): perturb one feature at a time, recompute the sample’s position in decision/proximity space relative to the fixed training set, and measure whether that perturbation moves the sample away from its focal class or toward other classes.

However, `core.py` generalizes the original binary bit-flip setting in several ways:

- it works with arbitrary dense numeric data, not only binary vectors;
- it supports multiple perturbation schemes;
- it operates on query samples against a fixed training reference set through `prox_extend`;
- it can return directional summaries for ordered numeric features.

### 6.2 Query labels and perturbations

Let $x_i^{(q)}$ be a query point, and let $g_i$ be its focal class:

- by default $g_i$ is the model prediction for the query;
- optionally the caller can provide labels explicitly, e.g. true labels for error analysis.

For feature $m$ and perturbation replicate $r$, let

$$
x_i^{(m,r)}
$$

denote the query point with only feature $m$ perturbed. The perturbation modes are:

- `flip`: swap binary values;
- `permute`: replace with sampled training values;
- `quantile`: replace with a deterministic grid of training quantiles;
- `auto`: choose `flip` for binary columns, `quantile` for numeric non-binary columns, and `permute` otherwise.

For a whole query batch, define:

$$
P_i = P(x_i^{(q)}, X_{\mathrm{train}}),
\qquad
P_i^{(m,r)} = P(x_i^{(m,r)}, X_{\mathrm{train}}).
$$

Then define the squared perturbation shift

$$
\Delta_{ij}^{(m,r)}
=
\left(P_{ij}^{(m,r)} - P_{ij}\right)^2.
$$

### 6.3 Contribution score implemented in `core.py`

Let

$$
G_i = \{j : y_j = g_i\},
\qquad
G_i^c = \{j : y_j \neq g_i\}.
$$

The implemented local contribution for feature $m$ and replicate $r$ is

$$
L_{im}^{(r)}
=
\sum_{j \in G_i^c} \Delta_{ij}^{(m,r)}
-
\sum_{j \in G_i} \Delta_{ij}^{(m,r)}.
$$

Equivalently, with a signed group vector

$$
z_{ij}
=
\begin{cases}
-1, & j \in G_i,\\
+1, & j \in G_i^c,
\end{cases}
$$

we can write

$$
L_{im}^{(r)} = \sum_{j=1}^n z_{ij}\Delta_{ij}^{(m,r)}.
$$

The returned score is the average over repeats:

$$
L_{im}
=
\frac{1}{R_m}
\sum_{r=1}^{R_m} L_{im}^{(r)}.
$$

Interpretation used by the code and notebooks:

- $L_{im} > 0$: the observed value of feature $m$ supported the focal class $g_i$;
- $L_{im} < 0$: the observed value pushed the sample away from $g_i$;
- $L_{im} \approx 0$: little local effect under the chosen perturbation scheme.

This is a proximity-space perturbation score, not an additive decomposition of the prediction.

### 6.4 Closeness outputs

When `return_closeness=True`, the code also returns

$$
C_{im}^{\mathrm{in}}
=
\frac{1}{R_m}
\sum_{r=1}^{R_m}
\frac{1}{|G_i|}
\sum_{j \in G_i}
\left(P_{ij}^{(m,r)}\right)^2,
$$

$$
C_{im}^{\mathrm{out}}
=
\frac{1}{R_m}
\sum_{r=1}^{R_m}
\frac{1}{|G_i^c|}
\sum_{j \in G_i^c}
\left(P_{ij}^{(m,r)}\right)^2.
$$

These are the code’s proximity-space analogues of Whitmore-style closeness summaries.

### 6.5 Directionality outputs

When `return_directionality=True`, numeric perturbations are split according to whether the perturbed value is above or below the observed one. Define indicator sets

$$
\mathcal R_{im}^{\uparrow}
=
\{r : x_{im}^{(m,r)} > x_{im}^{(q)}\},
\qquad
\mathcal R_{im}^{\downarrow}
=
\{r : x_{im}^{(m,r)} < x_{im}^{(q)}\}.
$$

Then the code reports

$$
L_{im}^{\uparrow}
=
\frac{1}{|\mathcal R_{im}^{\uparrow}|}
\sum_{r \in \mathcal R_{im}^{\uparrow}} L_{im}^{(r)},
\qquad
L_{im}^{\downarrow}
=
\frac{1}{|\mathcal R_{im}^{\downarrow}|}
\sum_{r \in \mathcal R_{im}^{\downarrow}} L_{im}^{(r)},
$$

together with the corresponding counts. If no perturbation of a given direction occurs, the returned directional value is `NaN`.

### 6.6 What should be validated experimentally

Recommended local-importance checks:

1. signal-vs.-noise recovery on synthetic data;
2. sensitivity to focal-class choice (`labels=` true vs. predicted);
3. `flip` vs. `permute` vs. `quantile` vs. `auto`;
4. directional asymmetry for ordered features;
5. stability across repeats and forest seeds;
6. agreement or disagreement with split-based importances and SHAP-style explanations.

## 7. Leaf-incidence view of all four applications

The most useful unifying perspective for experiments is that every method starts from the same object:

$$
P = QW^\top.
$$

Then:

- prediction uses $P$ as a row-stochastic weight operator on labels;
- imputation uses $P$ as a feature-wise smoothing operator after masking missing references;
- outlier detection uses row functionals of $P$, mainly $\sum_j P_{ij}^2$;
- local feature importance compares $P$ before and after controlled perturbations.

This suggests a clean experimental decomposition:

1. validate the proximity construction itself;
2. validate the algebra performed on top of $P$;
3. isolate which downstream gains come from RF-GAP specifically, versus from any reasonable forest-induced similarity.

## 8. Practical implications for future experiments

For later benchmark design, the most important implementation-aware takeaways are:

1. RF-GAP is the only mode in `core.py` with an exact prediction-reproduction target.
2. The current imputation method is single-pass on a fixed fitted forest, not the full iterative refit loop from the RF-GAP paper.
3. The local feature importance routine is a generalized perturbation-in-proximity-space method; it is closest to Whitmore et al., but broadened beyond binary bit flips.
4. Asymmetry matters. For prediction and outlier scoring, the unsymmetrized RF-GAP matrix is the primary object. For visualization-style applications, the symmetrized variants may be preferable.
5. Because all applications depend on the same $P$, a strong experiment suite should reuse the exact same fitted forests when comparing downstream methods, so that differences are attributable to the proximity definition rather than to model randomness.

## 9. Minimal experiment checklist

If we want a compact but high-value validation suite, it should include:

1. Classification:
   verify `prox_predict(X_test)` = `predict_proba(X_test)` and `prox_predict()` = `oob_decision_function_`.
2. Regression:
   verify `prox_predict(X_test)` = `predict(X_test)` and `prox_predict()` = `oob_prediction_`.
3. Imputation:
   compare original / OOB / RF-GAP under MCAR missingness, with both single-pass and iterative outer-loop variants.
4. Outliers:
   compare Breiman and ENE scores on synthetic corruption and on label-free anomaly settings.
5. Local importance:
   verify recovery of known local signals, focal-class sign reversal, and directionality behavior under quantile perturbations.

That set would give us an end-to-end test of both the RF-GAP theory and the concrete `core.py` implementation.
