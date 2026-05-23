# RISKSEG PROJECT --- SESSION HANDOFF

## Objective

Develop a segmentation-based predictive model (RiskSeg) optimized for
ranking (AUC, KS, Lift) using time-series features.

## Core Design

-   Tree segmentation + Logistic Regression
-   Screening → Split → Leaf models → Global stacking

## Key Decisions

-   Keep FULL dataset for modeling
-   Restrict only split variables via `screening_variables`
-   Use LogisticRegression with `class_weight='balanced'`
-   Use stacking (global + local)

## Screening Control

Implemented: - screening_variables parameter - helper:
\_get_allowed_screening_variables - filtering applied inside screening
step

## Performance Findings

Best result: - AUC ≈ 0.769 - KS ≈ 0.43

Balanced weights improved performance consistently.

## Optimization Strategy

Recommended: - factorial_max_interaction_features = 10 - top_k_variables
= 2 - min_samples_leaf = 0.15 - n_numeric_bins = 3

## Known Bottleneck

Factorial explosion due to: - many lag features - high collinearity

## Bugs Fixed

-   LogisticRegression scope issue
-   screening_variables not recognized
-   pandas fragmentation

## Next Steps

1.  Tune factorial interactions (k=8--12)
2.  Extract split variable importance
3.  Optimize threshold (not just AUC)
4.  Evaluate stability across folds

## Mental Model

Full features (model) + restricted features (splits) + controlled
factorial = stable model

## Resume Command

Continue RiskSeg optimization with screening control and factorial
tuning.
