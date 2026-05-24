"""Benchmark do RiskSegV2 vs LR baseline nos datasets do paper ICAI 2012.

Uso:
    python scripts/run_v2_benchmark.py
    python scripts/run_v2_benchmark.py --datasets adult german
    python scripts/run_v2_benchmark.py --n-splits 5 --output-dir artifacts/v2_smoke
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fbtseg import article_specs, article_uci_preset, load_article_dataset


def _baseline_pipeline(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scaler", MinMaxScaler())]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.0)


def baseline_logistic_predict(X_train, y_train, X_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pre = _baseline_pipeline(X_train)
        Z_train = pre.fit_transform(X_train, y_train)
        Z_test = pre.transform(X_test)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
        lr.fit(Z_train, y_train)
        return lr.predict_proba(Z_test)[:, 1]


def evaluate_v2(spec, X_train, y_train, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        t0 = time.perf_counter()
        model = article_uci_preset(categorical_features=spec.categorical_columns)
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = model.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t1
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "error": float(np.mean((proba >= 0.5).astype(int) != y_test.to_numpy())),
        "fit_s": fit_s,
        "pred_s": pred_s,
    }


def evaluate_lr(X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    proba = baseline_logistic_predict(X_train, y_train, X_test)
    fit_s = time.perf_counter() - t0
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "error": float(np.mean((proba >= 0.5).astype(int) != y_test.to_numpy())),
        "fit_s": fit_s,
        "pred_s": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["LR", "V2"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="artifacts/v2_benchmark")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = article_specs()
    if args.datasets:
        specs = [s for s in specs if s.name in args.datasets]

    rows = []
    for spec in specs:
        print(f"\n[DATASET] {spec.name}")
        X, y = load_article_dataset(spec)
        if args.max_rows is not None and len(X) > args.max_rows:
            X = X.iloc[: args.max_rows].reset_index(drop=True)
            y = y.iloc[: args.max_rows].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
            X_tr = X.iloc[tr].reset_index(drop=True)
            X_te = X.iloc[te].reset_index(drop=True)
            y_tr = y.iloc[tr].reset_index(drop=True)
            y_te = y.iloc[te].reset_index(drop=True)
            print(f"  fold={fold}  train={len(X_tr)}  test={len(X_te)}")

            if "LR" in args.models:
                m = evaluate_lr(X_tr, y_tr, X_te, y_te)
                m.update(dataset=spec.name, model="LR", fold=fold)
                rows.append(m)
                print(f"    LR  err={m['error']:.4f}  auc={m['auc']:.4f}  fit_s={m['fit_s']:.2f}")

            if "V2" in args.models:
                m = evaluate_v2(spec, X_tr, y_tr, X_te, y_te)
                m.update(dataset=spec.name, model="V2", fold=fold)
                rows.append(m)
                print(f"    V2  err={m['error']:.4f}  auc={m['auc']:.4f}  "
                      f"fit_s={m['fit_s']:.2f}  pred_s={m['pred_s']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "v2_benchmark_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "model"]).agg(
            error=("error", "mean"),
            auc=("auc", "mean"),
            fit_s=("fit_s", "mean"),
            pred_s=("pred_s", "mean"),
        ).reset_index().sort_values(["dataset", "error"])
    )
    summary.to_csv(out_dir / "v2_benchmark_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
