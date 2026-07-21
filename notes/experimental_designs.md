

1. Proximity-weighted predictions for RF-GAP-capable methods.
   1. (List all methods here).
2. Missing data imputation
   1. Mechanisms:
      1. MCAR
      2. MNAR
      3. MAR
   2. Percentatges:
      1. 5%
      2. 10%
      3. 25%
      4. 50%
      5. 75%
   3. Initializations:
      1. Stats (mean, median, mode)
      2. k-NN
      3. missForest?
      4. MICE?
3. Outlier Detection
   1. What are the key metrics? 
   2. Use of ground truth/simulations?
   3. Normalization
   4. Class-conditioning
   5. Quantizing Regression
4. Local Feature Importance
   1. How to measure fit here?
   2. Comparisons:
      1. SHAP
      2. LIME
      3. CLIQUE (Need Python adaptation?)
5. Manifold Learning
   1. RF-PHATE
      1. Revisit/reuse metrics from this paper?
   2. RF-AE?
6. Manifold Alignment
   1. Fosta
   2. RF-MA
7. Prototypes:
   1. What is a meaningful prototype?
   2. How to measure?
   3. What to compare to?
8. Clustering?