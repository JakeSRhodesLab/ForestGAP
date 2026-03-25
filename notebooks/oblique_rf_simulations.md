# Simulated Datasets for Oblique Forest Comparisons

This file documents the simulated classification datasets implemented in
[`forestgap.datasets`](c:/Users/rhodes8/My%20Drive/repos/ObliqueProximities/ObliqueProximitiesCode/forestgap/datasets.py)
for studying when oblique random forests can outperform axis-aligned
random forests.

The general pattern is the same across all of them: standard random
forests split one coordinate at a time, while oblique forests can split
on linear combinations of coordinates. These simulations are designed so
that the class boundary is easier to express in a rotated, correlated, or
projected feature space than as a union of axis-aligned rectangles.

## Implemented Dataset Generators

### 1. `make_rotated_linear_boundary`

- Core idea: a linear separator exists in a rotated 2D subspace.
- Construction:
  `y = 1(x1 + x2 + eps > 0)`, then the observed coordinates are rotated.
- Why it is hard for standard RF:
  the Bayes boundary is diagonal, so axis-aligned trees need many small
  rectangular refinements to approximate one clean split.
- Good knobs:
  `angle_degrees`, `noise_std`, `n_features`

### 2. `make_rotated_xor`

- Core idea: XOR structure survives, but the useful quadrants are rotated.
- Construction:
  `y = 1((z1 > 0) xor (z2 > 0))`, then rotate and add feature noise.
- Why it is hard for standard RF:
  XOR is already interaction-heavy, and rotation removes any simple
  axis-aligned decomposition.
- Good knobs:
  `angle_degrees`, `feature_noise_std`, `n_features`

### 3. `make_oblique_gaussian_clusters`

- Core idea: class separation is governed by correlated elliptical clouds.
- Construction:
  two Gaussian classes with shared correlated covariance and rotated mean
  separation.
- Why it is hard for standard RF:
  the informative direction is a correlated projection, not a single
  coordinate threshold.
- Good knobs:
  `correlation`, `class_sep`, `angle_degrees`, `n_features`

### 4. `make_rotated_subspace_signal`

- Core idea: signal lives in a low-dimensional latent space embedded in a
  higher-dimensional observed space.
- Construction:
  generate latent variables `z`, define the label from a linear rule in
  `z`, then map into the observed space using a random orthonormal mixing
  matrix and add noise.
- Why it is hard for standard RF:
  no single observed feature is especially informative; the signal is
  spread across a projection of many coordinates.
- Good knobs:
  `n_features`, `signal_noise_std`, `feature_noise_std`

### 5. `make_additive_interaction_correlated`

- Core idea: the label depends on an interaction plus a weak linear term
  under correlated predictors.
- Construction:
  `y = 1(x1 * x2 + 0.3 * x3 + eps > 0)` with correlated Gaussian inputs.
- Why it is hard for standard RF:
  the response is not axis-separable, and the correlation structure can
  make projection-based splits more efficient.
- Good knobs:
  `correlation`, `noise_std`, `n_features`

### 6. `make_diagonal_band`

- Core idea: the positive class lies in a narrow band around the line
  `x1 = x2`.
- Construction:
  `y = 1(|x1 - x2 + eps| < c)`.
- Why it is hard for standard RF:
  a thin diagonal strip requires many axis-aligned boxes, while an oblique
  split naturally aligns with `x1 - x2`.
- Good knobs:
  `band_width`, `noise_std`, `n_features`

### 7. `make_sparse_oblique_hyperplane`

- Core idea: in high dimensions, only a few coordinates matter, but they
  matter through a sparse linear combination.
- Construction:
  sample a sparse projection vector and define
  `y = 1(w^T x + eps > 0)`.
- Why it is hard for standard RF:
  even though the signal is sparse, it is still projection-based rather
  than axis-based.
- Good knobs:
  `n_features`, `n_informative`, `margin_noise_std`

### 8. `make_oblique_slab`

- Core idea: the positive class occupies a thin slab centered on an
  oblique hyperplane.
- Construction:
  `y = 1(|w^T x + eps| < c)` for a random sparse projection `w`.
- Why it is hard for standard RF:
  two roughly parallel oblique boundaries must be approximated
  simultaneously, which is awkward for axis-aligned partitions.
- Good knobs:
  `n_features`, `slab_half_width`, `margin_noise_std`

### 9. `make_dual_projection_xor`

- Core idea: XOR is defined on two different linear projections in a
  higher-dimensional space.
- Construction:
  sample two projection vectors `w1`, `w2`, compute
  `y = 1((w1^T x > 0) xor (w2^T x > 0))`.
- Why it is hard for standard RF:
  the class structure is interaction-driven and tied to two separate
  oblique directions.
- Good knobs:
  `n_features`, `projection_noise_std`

## Design Principles Behind These Simulations

The simulations above intentionally emphasize one or more of the
following:

- Rotated decision boundaries
- Correlated predictors
- Signal concentrated in a low-dimensional projection
- Thin bands or slabs that are not axis-aligned
- Interactions that become easier to approximate after projection
- High-dimensional sparse hyperplanes

These are all settings where an oblique split can represent a large
useful region of the decision space with fewer nodes than an
axis-aligned split.

## Recommended Starting Points

If you want a small subset that usually gives the clearest contrast
between standard RF and oblique RF, start with:

- `make_rotated_linear_boundary`
- `make_rotated_xor`
- `make_diagonal_band`
- `make_sparse_oblique_hyperplane`
- `make_oblique_slab`

## Experimental Suggestions

Useful parameters to vary across these generators:

- rotation angle
- predictor correlation
- label noise / margin noise
- number of noise features
- number of informative features
- slab or band width

Suggested comparisons:

- test accuracy
- balanced accuracy if classes become imbalanced
- AUC when scores are available
- proximity-based PHATE plots
- tree depth or node count as a rough efficiency diagnostic
