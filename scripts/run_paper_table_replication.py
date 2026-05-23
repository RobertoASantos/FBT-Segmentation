"""Replicacao da Tabela 1 do paper ICAI 2012.

Cruza 3 base learners (Linear, Logistic, MLP) com {Simple, V2(FBTSeg)}
sobre as 5 bases UCI usadas no artigo: chess, german, magic, adult, spambase.

Metrica: error rate (paper usa erro de classificacao). K-fold CV.

Uso:
    python scripts/run_paper_table_replication.py
    python scripts/run_paper_table_replication.py --datasets magic chess
    python scripts/run_paper_table_replication.py --n-splits 10
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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fbtseg import (
    LinearProbabilityClassifier,
    article_specs,
    article_uci_preset,
    load_article_dataset,
)


def preprocess(X_train, X_test, categorical_cols):
    """Numerico em [0,1] + dummies para categorico. Igual ao protocolo do paper."""
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    cat_cols = [c for c in X_train.columns if c in categorical_cols]

    Xtr = X_train.copy()
    Xte = X_test.copy()

    if num_cols:
        scaler = MinMaxScaler().fit(Xtr[num_cols].fillna(0).values)
        Xtr[num_cols] = scaler.transform(Xtr[num_cols].fillna(0).values)
        Xte[num_cols] = scaler.transform(Xte[num_cols].fillna(0).values)

    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(Xtr[cat_cols].astype(str).values)
        dummies_tr = ohe.transform(Xtr[cat_cols].astype(str).values)
        dummies_te = ohe.transform(Xte[cat_cols].astype(str).values)
        out_tr = np.hstack([Xtr[num_cols].values, dummies_tr]) if num_cols else dummies_tr
        out_te = np.hstack([Xte[num_cols].values, dummies_te]) if num_cols else dummies_te
    else:
        out_tr = Xtr[num_cols].values
        out_te = Xte[num_cols].values

    return out_tr, out_te


def err(y, pred):
    return float(np.mean(pred != y.to_numpy()))


def fit_simple(base_name, base_factory, Xtr_raw, ytr, Xte_raw, yte, cat_cols):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, cat_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = base_factory()
        model.fit(Xtr, ytr.values)
        pred = model.predict(Xte)
    return err(yte, pred)


def fit_v2(base_name, base_factory, Xtr, ytr, Xte, yte, cat_cols, max_depth=2):
    """V2 com base learner configuravel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        rs = article_uci_preset(
            categorical_features=cat_cols,
            base_estimator=base_factory(),
            max_depth=max_depth,
        )
        rs.fit(Xtr, ytr)
        pred = (rs.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    return err(yte, pred)


def base_factories(random_state: int = 42):
    return {
        "Linear": lambda: LinearProbabilityClassifier(),
        "Logistic": lambda: LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=2000, random_state=random_state
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(10,),
            activation="logistic",
            max_iter=200,
            early_stopping=False,  # evita falha em segmentos pequenos
            random_state=random_state,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="artifacts/v2_paper_table")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = article_specs()
    if args.datasets:
        specs = [s for s in specs if s.name in args.datasets]

    factories = base_factories()
    rows = []

    for spec in specs:
        print(f"\n[DATASET] {spec.name}")
        X, y = load_article_dataset(spec)
        if args.max_rows is not None and len(X) > args.max_rows:
            X = X.iloc[: args.max_rows].reset_index(drop=True)
            y = y.iloc[: args.max_rows].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            Xtr = X.iloc[tr].reset_index(drop=True)
            Xte = X.iloc[te].reset_index(drop=True)
            ytr = y.iloc[tr].reset_index(drop=True)
            yte = y.iloc[te].reset_index(drop=True)

            for base_name, factory in factories.items():
                t0 = time.perf_counter()
                err_simple = fit_simple(
                    base_name, factory, Xtr, ytr, Xte, yte, spec.categorical_columns
                )
                t_simple = time.perf_counter() - t0

                t1 = time.perf_counter()
                err_v2 = fit_v2(
                    base_name, factory, Xtr, ytr, Xte, yte,
                    spec.categorical_columns, max_depth=args.max_depth,
                )
                t_v2 = time.perf_counter() - t1

                rows.append({
                    "dataset": spec.name,
                    "fold": fold,
                    "base": base_name,
                    "err_Simple": err_simple,
                    "err_V2": err_v2,
                    "delta_err": err_v2 - err_simple,
                    "fit_s_Simple": t_simple,
                    "fit_s_V2": t_v2,
                })
                print(
                    f"  fold={fold} base={base_name:8s}  Simple={err_simple:.4f}  "
                    f"V2={err_v2:.4f}  delta={err_v2 - err_simple:+.4f}  "
                    f"t_V2={t_v2:.1f}s"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "paper_table_results.csv", index=False)

    # Pivot table — replica Tabela 1 do paper
    pivot = df.groupby(["dataset", "base"]).agg(
        Simple_mean=("err_Simple", "mean"),
        Simple_std=("err_Simple", "std"),
        V2_mean=("err_V2", "mean"),
        V2_std=("err_V2", "std"),
        delta_mean=("delta_err", "mean"),
    ).reset_index()
    pivot.to_csv(out / "paper_table_summary.csv", index=False)
    print("\n=== Tabela 1 reproduzida (mean error %) ===")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Replicacao da Tabela 1 em formato markdown
    md_lines = ["# Replicacao Tabela 1 do paper ICAI 2012\n"]
    md_lines.append(f"K-fold CV (k={args.n_splits}), max_depth={args.max_depth}\n")
    md_lines.append("| Dataset | Base | Simple | V2 (FBTSeg) | Delta |")
    md_lines.append("|---|---|---:|---:|---:|")
    for _, r in pivot.iterrows():
        md_lines.append(
            f"| {r['dataset']} | {r['base']} | "
            f"{r['Simple_mean']*100:.2f}% ± {r['Simple_std']*100:.2f} | "
            f"{r['V2_mean']*100:.2f}% ± {r['V2_std']*100:.2f} | "
            f"{r['delta_mean']*100:+.2f} |"
        )
    (out / "paper_table_replicated.md").write_text("\n".join(md_lines) + "\n")

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
