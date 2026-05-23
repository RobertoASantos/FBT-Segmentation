"""Debug rapido do gap V2 vs paper FBTSeg no Adult.

3-fold (rapido) testando ~5 configuracoes diferentes.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from benchmarks.article_suite import (
    build_article_preprocessor,
    get_article_dataset_specs,
    load_article_dataset,
)
from riskseg.v2 import article_uci_preset


def err(y, p):
    return float(np.mean((p >= 0.5).astype(int) != y.to_numpy()))


def main():
    spec = next(s for s in get_article_dataset_specs() if s.name == "adult")
    X, y = load_article_dataset(spec)
    print(f"Adult: {len(X)} rows, base_rate={y.mean():.4f}", flush=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    tr, te = splits[0]
    X_tr = X.iloc[tr].reset_index(drop=True)
    X_te = X.iloc[te].reset_index(drop=True)
    y_tr = y.iloc[tr].reset_index(drop=True)
    y_te = y.iloc[te].reset_index(drop=True)

    # baseline LR
    pre = build_article_preprocessor(X_tr)
    Ztr = pre.fit_transform(X_tr, y_tr)
    Zte = pre.transform(X_te)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
    lr.fit(Ztr, y_tr)
    p_lr = lr.predict_proba(Zte)[:, 1]
    print(f"\nLR baseline   err={err(y_te, p_lr):.4f}", flush=True)

    configs = [
        ("preset default",                 {}),
        ("max_depth=3",                    {"max_depth": 3}),
        ("max_depth=3, top_k=3",           {"max_depth": 3, "top_k_variables": 3}),
        ("max_depth=3, marginal_odds",     {"max_depth": 3, "combiner_method": "marginal_odds"}),
        ("max_depth=3, global_stacking",   {"max_depth": 3, "prediction_mode": "global_stacking"}),
        ("max_depth=3, cascade",           {"max_depth": 3, "prediction_mode": "cascade"}),
    ]
    for label, kw in configs:
        t0 = time.perf_counter()
        m = article_uci_preset(categorical_features=spec.categorical_columns, **kw).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        e = err(y_te, p)
        n_int = sum(1 for n in m.nodes_ if not n.is_leaf)
        vars_used = m.get_summary()["used_variables_unique"]
        print(
            f"{label:36s} err={e:.4f}  fit={time.perf_counter()-t0:5.1f}s  "
            f"n_internal={n_int}  vars={vars_used}",
            flush=True,
        )

    # arvore mais profunda
    print("\n=== Arvore do preset default (fold 1) ===", flush=True)
    m = article_uci_preset(categorical_features=spec.categorical_columns).fit(X_tr, y_tr)
    print(m.plot_model_tree(), flush=True)


if __name__ == "__main__":
    main()
