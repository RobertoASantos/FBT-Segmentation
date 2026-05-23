# RISKSEG PROJECT --- FULL TECHNICAL HANDOFF

------------------------------------------------------------------------

## 1. OVERVIEW

RiskSeg is a hybrid segmentation + machine learning framework designed
for: - High-performance ranking (AUC, KS, Lift) - Interpretability via
tree-based segmentation - Handling structured + time-series engineered
features

Core concept: \> Use tree splits to create local regimes, then apply
logistic models and combine via stacking.

------------------------------------------------------------------------

## 2. ARCHITECTURE

### 2.1 Pipeline

1.  Input data (full feature set)
2.  Feature engineering (lags, rolling stats, derivatives)
3.  Segmentation (tree-based)
4.  Node-level modeling
5.  Leaf-level modeling
6.  Global stacking

------------------------------------------------------------------------

### 2.2 Components

#### Screening Layer

-   Selects best variables for split candidates
-   Uses:
    -   factorial_lr (main)
    -   correlation / target (fast modes)

#### Split Optimization

-   Evaluates candidate splits using:
    -   AUC
    -   KS
    -   Lift
    -   Combined metric

#### Leaf Models

-   LogisticRegression
-   Weighted (balanced)

#### Final Combiner

-   Stacking (global + local)

------------------------------------------------------------------------

## 3. KEY CUSTOM FEATURE: SCREENING CONTROL

### Problem

Too many variables → explosion in split search space.

### Solution

Introduce:

``` python
screening_variables=None
```

### Behavior

-   Full dataset used for modeling
-   Only subset used for split decisions

### Internal Logic

``` python
def _get_allowed_screening_variables(self, X_seg_columns):
    if self.screening_variables is None:
        return list(X_seg_columns)
    return [c for c in self.screening_variables if c in X_seg_columns]
```

Applied in:

``` python
allowed_vars = self._get_allowed_screening_variables(X_seg.columns)
```

------------------------------------------------------------------------

## 4. PERFORMANCE INSIGHTS

### Best Observed Metrics

-   AUC ≈ 0.769
-   KS ≈ 0.43

### Key Finding

Balanced weights improve generalization.

------------------------------------------------------------------------

## 5. MAIN BOTTLENECK

Factorial expansion:

``` python
factorial_max_interaction_features=None
```

Issues: - Combinatorial explosion - Redundant interactions - Slower
training - Overfitting risk

------------------------------------------------------------------------

## 6. OPTIMIZATION STRATEGY

### Recommended Config

``` python
factorial_max_interaction_features=10
top_k_variables=2
min_samples_leaf=0.15
n_numeric_bins=3
use_grouping=False
```

------------------------------------------------------------------------

### Fast Debug Mode

``` python
screening_mode="target"
max_depth=2
factorial_max_interaction_features=5
```

------------------------------------------------------------------------

## 7. DATA CHARACTERISTICS

Dataset includes: - Lag features (12, 48, 96 windows) - Aggregations
(mean, std, var) - Strong multicollinearity

Implication: \> Limiting interaction space improves performance.

------------------------------------------------------------------------

## 8. BUGS FIXED

### 8.1 LogisticRegression Scope

Moved import to global scope.

### 8.2 screening_variables not recognized

Cause: wrong module loaded.

### 8.3 Pandas Fragmentation

Replaced column-by-column insertion with batch DataFrame creation.

------------------------------------------------------------------------

## 9. STRATEGIC DECISIONS

### KEEP

-   Full feature space
-   Balanced weights
-   Logistic regression
-   Stacking approach

### CONTROL

-   Split variables
-   Factorial complexity

### AVOID

-   Reducing dataset globally
-   Over-reliance on threshold metrics

------------------------------------------------------------------------

## 10. NEXT STEPS

1.  Tune factorial interactions (k=8--12)
2.  Extract variable importance per node
3.  Optimize threshold
4.  Cross-validation stability analysis

------------------------------------------------------------------------

## 11. MENTAL MODEL

Full dataset → model\
Restricted variables → splits\
Controlled factorial → generalization

------------------------------------------------------------------------

## 12. RESTART COMMAND

When starting new session:

> Continue RiskSeg with screening_variables and factorial tuning.

------------------------------------------------------------------------

END OF DOCUMENT
