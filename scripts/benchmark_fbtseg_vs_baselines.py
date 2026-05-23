#!/usr/bin/env python
"""Benchmark: fbtseg vs LogisticRegression, Linear Probability, MLP.

Replica a comparação do paper ICAI 2012 (Tabela 1) em uma base UCI
de escolha, com stratified k-fold cross-validation e métricas:
- error_rate
- auc (ROC)
- ks (KS estatístico)
- precision
- lift (@ top 10%)

Uso:

    python scripts/benchmark_fbtseg_vs_baselines.py --dataset chess --n-splits 3
    python scripts/benchmark_fbtseg_vs_baselines.py --dataset magic --n-splits 5
    python scripts/benchmark_fbtseg_vs_baselines.py --dataset adult --smoke
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from fbtseg import (
    FBTSeg,
    LinearProbabilityClassifier,
    article_uci_preset,
    load_article_dataset,
)
from fbtseg.datasets import get_spec

warnings.simplefilter("ignore", ConvergenceWarning)


def encode_categoricals(X, categorical_columns):
    """Encode categorical columns to integers."""
    X_copy = X.copy()
    encoders = {}
    for col in categorical_columns:
        if col in X_copy.columns:
            le = LabelEncoder()
            X_copy[col] = le.fit_transform(X_copy[col].astype(str))
            encoders[col] = le
    return X_copy, encoders


def evaluate_proba(y_true, y_proba):
    """Compute metrics from predicted probabilities."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix

    y_pred = (y_proba >= 0.5).astype(int)

    # Error rate
    error_rate = (y_pred != y_true).mean()

    # AUC (ROC)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    # KS statistic (máxima separação entre distribuições)
    ks = abs((y_proba[y_true == 1] <= 0.5).mean() - (y_proba[y_true == 0] <= 0.5).mean())

    # Precision
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Lift @ top 10%
    n_top_10pct = max(1, int(0.1 * len(y_true)))
    top_10pct_idx = np.argsort(y_proba)[-n_top_10pct:]
    lift_10pct = (y_true[top_10pct_idx].mean() / y_true.mean()) if y_true.mean() > 0 else 1.0

    return {
        "error_rate": error_rate,
        "auc": auc_score,
        "ks": ks,
        "precision": precision,
        "lift_10pct": lift_10pct,
    }


def benchmark_dataset(dataset_name: str, n_splits: int = 5, smoke: bool = False):
    """Benchmark fbtseg vs baselines on a single dataset."""

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load data
    spec = get_spec(dataset_name)
    X, y = load_article_dataset(spec)
    print(f"Shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Reduce for smoke test
    if smoke:
        idx = np.arange(min(500, len(X)))
        X, y = X.iloc[idx], y.iloc[idx]
        n_splits = 2
        print(f"[SMOKE] Reduced to {len(X)} samples, {n_splits} splits")

    # Prepare categorical encoding for baselines
    cat_cols = spec.categorical_columns
    X_encoded, encoders = encode_categoricals(X, cat_cols)

    # Storage for fold results
    results = {
        "fbtseg": [],
        "logistic": [],
        "linear_prob": [],
        "mlp": [],
    }

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        print(f"\n  Fold {fold_idx}/{n_splits}...", end=" ", flush=True)

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        X_tr_enc, X_te_enc = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]

        # --- fbtseg ---
        try:
            model_fbtseg = article_uci_preset(categorical_features=cat_cols)
            t0 = time.perf_counter()
            model_fbtseg.fit(X_tr, y_tr)
            t_fbtseg = time.perf_counter() - t0
            proba_fbtseg = model_fbtseg.predict_proba(X_te)[:, 1]
            metrics_fbtseg = evaluate_proba(y_te.values, proba_fbtseg)
            metrics_fbtseg["time"] = t_fbtseg
            results["fbtseg"].append(metrics_fbtseg)
            print(f"fbtseg({metrics_fbtseg['error_rate']:.4f})", end=" | ", flush=True)
        except Exception as e:
            print(f"fbtseg(ERROR: {str(e)[:20]})", end=" | ", flush=True)
            results["fbtseg"].append({})

        # --- LogisticRegression (no regularization, like paper) ---
        try:
            model_lr = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
            t0 = time.perf_counter()
            model_lr.fit(X_tr_enc, y_tr)
            t_lr = time.perf_counter() - t0
            proba_lr = model_lr.predict_proba(X_te_enc)[:, 1]
            metrics_lr = evaluate_proba(y_te.values, proba_lr)
            metrics_lr["time"] = t_lr
            results["logistic"].append(metrics_lr)
            print(f"logistic({metrics_lr['error_rate']:.4f})", end=" | ", flush=True)
        except Exception as e:
            print(f"logistic(ERROR)", end=" | ", flush=True)
            results["logistic"].append({})

        # --- LinearProbabilityClassifier ---
        try:
            model_lp = LinearProbabilityClassifier()
            t0 = time.perf_counter()
            model_lp.fit(X_tr_enc, y_tr)
            t_lp = time.perf_counter() - t0
            proba_lp = model_lp.predict_proba(X_te_enc)[:, 1]
            metrics_lp = evaluate_proba(y_te.values, proba_lp)
            metrics_lp["time"] = t_lp
            results["linear_prob"].append(metrics_lp)
            print(f"linear({metrics_lp['error_rate']:.4f})", end=" | ", flush=True)
        except Exception as e:
            print(f"linear(ERROR)", end=" | ", flush=True)
            results["linear_prob"].append({})

        # --- MLPClassifier ---
        try:
            model_mlp = MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=500,
                early_stopping=False,
                random_state=42,
            )
            t0 = time.perf_counter()
            model_mlp.fit(X_tr_enc, y_tr)
            t_mlp = time.perf_counter() - t0
            proba_mlp = model_mlp.predict_proba(X_te_enc)[:, 1]
            metrics_mlp = evaluate_proba(y_te.values, proba_mlp)
            metrics_mlp["time"] = t_mlp
            results["mlp"].append(metrics_mlp)
            print(f"mlp({metrics_mlp['error_rate']:.4f})", end="", flush=True)
        except Exception as e:
            print(f"mlp(ERROR)", end="", flush=True)
            results["mlp"].append({})

        print()

    # Aggregate results
    print(f"\n{'-'*70}")
    print(f"{'Method':<20} {'Error Rate':<15} {'AUC':<10} {'KS':<10} {'Precision':<12} {'Lift@10%':<10} {'Time (s)':<10}")
    print(f"{'-'*70}")

    summary = {}
    for method, metrics_list in results.items():
        if not metrics_list or not all(metrics_list):
            print(f"{method:<20} [FAILED]")
            continue

        valid_metrics = [m for m in metrics_list if m]
        if not valid_metrics:
            print(f"{method:<20} [NO VALID RESULTS]")
            continue

        avg_error = np.mean([m["error_rate"] for m in valid_metrics])
        avg_auc = np.mean([m["auc"] for m in valid_metrics])
        avg_ks = np.mean([m["ks"] for m in valid_metrics])
        avg_precision = np.mean([m["precision"] for m in valid_metrics])
        avg_lift = np.mean([m["lift_10pct"] for m in valid_metrics])
        avg_time = np.mean([m["time"] for m in valid_metrics])

        print(
            f"{method:<20} {avg_error*100:>6.2f}%        "
            f"{avg_auc:>6.4f}    {avg_ks:>6.4f}    {avg_precision:>6.4f}      "
            f"{avg_lift:>6.3f}      {avg_time:>6.3f}"
        )

        summary[method] = {
            "error_rate": avg_error,
            "auc": avg_auc,
            "ks": avg_ks,
            "precision": avg_precision,
            "lift_10pct": avg_lift,
            "time": avg_time,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fbtseg vs LogisticRegression, LinearProbability, MLP"
    )
    parser.add_argument(
        "--dataset",
        choices=["adult", "chess", "german", "magic", "spambase"],
        default="chess",
        help="Dataset to benchmark on (default: chess)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test with reduced data (500 samples, 2 splits)",
    )

    args = parser.parse_args()

    summary = benchmark_dataset(args.dataset, n_splits=args.n_splits, smoke=args.smoke)

    print(f"\n{'='*70}")
    print(f"Benchmark complete for {args.dataset.upper()}")
    print(f"{'='*70}\n")

    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
