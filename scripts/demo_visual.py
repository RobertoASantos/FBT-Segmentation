#!/usr/bin/env python
"""Demo FBTSeg: métricas no terminal + árvore de segmentação visual.

Treina o modelo em Chess e Magic, exibe métricas comparativas no
terminal e salva a árvore de segmentação de cada base em PNG.

    pip install fbtseg matplotlib
    python scripts/demo_visual.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

from fbtseg import FBTSeg, load_article_dataset, plot_tree
from fbtseg.datasets import get_spec

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    ("chess", "Chess",  "36 variáveis categóricas  |  branco vs preto vence"),
    ("magic", "Magic",  "10 variáveis numéricas    |  partículas gamma vs hadrão"),
]
TEST_SIZE   = 0.30
RANDOM_SEED = 42
MAX_DEPTH   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_v = auc(fpr, tpr)
    ordem = np.argsort(y_proba)
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    ks = 0.0
    if n_pos > 0 and n_neg > 0:
        ks = float(np.max(np.abs(
            np.cumsum(y_true[ordem]) / n_pos -
            np.cumsum(1 - y_true[ordem]) / n_neg
        )))
    base  = y_true.mean()
    n_top = max(1, int(0.10 * len(y_true)))
    lift  = float(y_true[np.argsort(y_proba)[-n_top:]].mean() / base) if base else 1.0
    return dict(
        error=float((y_pred != y_true).mean()),
        accuracy=float((y_pred == y_true).mean()),
        auc=float(auc_v),
        ks=float(ks),
        lift10=float(lift),
        y_pred=y_pred,
    )


def encode_for_lr(X_tr, X_te, cat_cols):
    Xtr, Xte = X_tr.copy(), X_te.copy()
    for col in cat_cols:
        if col in Xtr.columns:
            le = LabelEncoder()
            Xtr[col] = le.fit_transform(Xtr[col].astype(str))
            known = {c: i for i, c in enumerate(le.classes_)}
            Xte[col] = Xte[col].astype(str).map(known).fillna(0).astype(int)
    return Xtr, Xte


# ─────────────────────────────────────────────────────────────────────────────
# Treino + avaliação
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 64)
print("  FBTSeg  —  Demo com métricas e árvore visual")
print("═" * 64)

models_trained = {}

for ds_key, ds_name, ds_desc in DATASETS:
    print(f"\n▶  {ds_name}  ({ds_desc})")

    spec    = get_spec(ds_key)
    X, y    = load_article_dataset(spec)
    cat     = spec.categorical_columns

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   treino={len(X_tr)}  teste={len(X_te)}")

    # FBTSeg
    model = FBTSeg(
        max_depth=MAX_DEPTH,
        min_samples_leaf=0.05,
        metric="error",
        top_k_variables=1,
        n_numeric_bins=4,
        validation_fraction=0.20,
        scale_numeric=True,
        combiner_method="stacking",
        categorical_features=cat,
        random_state=RANDOM_SEED,
    )
    model.fit(X_tr, y_tr)
    proba_f = model.predict_proba(X_te)[:, 1]
    m_f     = compute_metrics(y_te.values, proba_f)

    # Logistic Regression (baseline)
    X_tr_enc, X_te_enc = encode_for_lr(X_tr, X_te, cat)
    lr = LogisticRegression(penalty=None, max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_tr_enc, y_tr)
    proba_l = lr.predict_proba(X_te_enc)[:, 1]
    m_l     = compute_metrics(y_te.values, proba_l)

    models_trained[ds_key] = dict(name=ds_name, model=model)

    # ── Tabela de métricas no terminal ───────────────────────────────────────
    print(f"\n   {'Métrica':<14} {'FBTSeg':>10} {'Logistic':>10}  {'Δ':>8}")
    print(f"   {'-'*14} {'-'*10} {'-'*10}  {'-'*8}")
    for k, label in [("error","Erro%"), ("accuracy","Acurácia"),
                     ("auc","AUC"), ("ks","KS"), ("lift10","Lift@10%")]:
        vf, vl = m_f[k], m_l[k]
        delta  = vf - vl
        fmt_f  = f"{vf*100:.2f}%" if k in ("error","accuracy") else f"{vf:.4f}"
        fmt_l  = f"{vl*100:.2f}%" if k in ("error","accuracy") else f"{vl:.4f}"
        sign   = "▲" if (k == "error" and delta < 0) or (k != "error" and delta > 0) else "▼"
        d_str  = (f"{sign} {abs(delta)*100:.2f}%" if k in ("error","accuracy")
                  else f"{sign} {abs(delta):.4f}")
        print(f"   {label:<14} {fmt_f:>10} {fmt_l:>10}  {d_str:>8}")

    # ── Árvore em texto ──────────────────────────────────────────────────────
    print(f"\n   Árvore (texto):\n")
    for line in model.plot_model_tree().splitlines():
        print(f"   {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Figuras — uma por dataset, somente a árvore
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n  Gerando árvores visuais...")

try:
    import matplotlib.pyplot as plt

    for ds_key, entry in models_trained.items():
        fig = plot_tree(
            entry["model"],
            title=f"FBTSeg — Árvore de Segmentação  ({entry['name']})",
        )
        out = f"tree_{ds_key}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        print(f"  Salvo: {out}")
        plt.show()

    print()

except ImportError:
    print("  matplotlib não instalado — figuras não geradas.")
    print("  pip install matplotlib")
