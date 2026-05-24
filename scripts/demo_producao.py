#!/usr/bin/env python
"""Demonstração do pacote fbtseg em produção.

Compara FBTSeg com os principais modelos de classificação binária
em 3 bases clássicas do scikit-learn.

    pip install fbtseg
    python demo_producao.py

Bases utilizadas (nenhum download externo necessário):
  - Breast Cancer   (569 amostras | 30 variáveis | diagnóstico maligno/benigno)
  - Credit Approval (690 amostras | 15 variáveis | aprovação de crédito)
  - Spambase        (4.601 amostras | 57 variáveis | detecção de spam)

Modelos comparados:
  - FBTSeg               (pacote em avaliação)
  - Logistic Regression  (baseline clássico)
  - Random Forest        (ensemble estado da arte)
  - Gradient Boosting    (ensemble estado da arte)
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Garante UTF-8 no terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

# --------------------------------------------------------------------------- #
# Importa fbtseg (instalado via pip)                                          #
# --------------------------------------------------------------------------- #
try:
    from fbtseg import FBTSeg
    FBTSEG_VERSION = __import__("importlib.metadata", fromlist=["version"]).version("fbtseg")
except ImportError:
    print("ERRO: fbtseg nao instalado. Execute: pip install fbtseg")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Bases de dados                                                               #
# --------------------------------------------------------------------------- #

def load_datasets() -> list[dict]:
    """Carrega as 3 bases. Breast Cancer e Banknote via sklearn/UCI integrado."""

    datasets = []

    # 1. Breast Cancer (sklearn built-in)
    bc = load_breast_cancer()
    datasets.append({
        "name": "Breast Cancer",
        "desc": "569 amostras | 30 vars numericas | maligno vs benigno",
        "X": __import__("pandas").DataFrame(bc.data, columns=bc.feature_names),
        "y": __import__("pandas").Series(bc.target),
        "categorical": (),
        "scale": True,
    })

    # 2. Spambase (fbtseg built-in loader — download automatico na 1a vez)
    try:
        from fbtseg import load_article_dataset
        from fbtseg.datasets import get_spec
        spec = get_spec("spambase")
        X_sp, y_sp = load_article_dataset(spec)
        datasets.append({
            "name": "Spambase",
            "desc": "4.601 amostras | 57 vars numericas | spam vs ham",
            "X": X_sp,
            "y": y_sp,
            "categorical": (),
            "scale": True,
        })
    except Exception as e:
        print(f"  [aviso] Spambase nao carregado: {e}")

    # 3. German Credit (fbtseg built-in loader)
    try:
        from fbtseg import load_article_dataset
        from fbtseg.datasets import get_spec
        spec = get_spec("german")
        X_ge, y_ge = load_article_dataset(spec)
        datasets.append({
            "name": "German Credit",
            "desc": "1.000 amostras | 20 vars mistas | risco de credito",
            "X": X_ge,
            "y": y_ge,
            "categorical": spec.categorical_columns,
            "scale": False,
        })
    except Exception as e:
        print(f"  [aviso] German nao carregado: {e}")

    return datasets


# --------------------------------------------------------------------------- #
# Metricas                                                                     #
# --------------------------------------------------------------------------- #

def metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    error  = float((y_pred != y_true).mean())

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score   = float(auc(fpr, tpr))

    order   = np.argsort(y_proba)
    n_pos   = y_true.sum()
    n_neg   = len(y_true) - n_pos
    ks_stat = 0.0
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(y_true[order]) / n_pos
        cum_neg = np.cumsum(1 - y_true[order]) / n_neg
        ks_stat = float(np.max(np.abs(cum_pos - cum_neg)))

    base = y_true.mean()
    n_top = max(1, int(0.10 * len(y_true)))
    top_idx = np.argsort(y_proba)[-n_top:]
    lift = float(y_true[top_idx].mean() / base) if base > 0 else 1.0

    return {"error": error, "auc": auc_score, "ks": ks_stat, "lift10": lift}


def avg(metric_list: list[dict]) -> dict:
    return {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #

def make_models(cat_cols: tuple, n_samples: int = 5000) -> list[tuple[str, object]]:
    # Datasets pequenos precisam de folha mínima maior para evitar segmentos degenerados
    min_leaf = 0.10 if n_samples < 2000 else 0.05
    return [
        ("FBTSeg", FBTSeg(
            max_depth=3,
            min_samples_leaf=min_leaf,
            metric="error",
            categorical_features=cat_cols,
        )),
        ("Logistic Regression", LogisticRegression(
            penalty=None, max_iter=1000, random_state=42,
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, random_state=42,
        )),
    ]


# --------------------------------------------------------------------------- #
# Cross-validation                                                             #
# --------------------------------------------------------------------------- #

def run_cv(dataset: dict, n_splits: int = 5) -> list[dict]:
    """Roda CV estratificado para todos os modelos. Retorna lista de resultados."""
    X      = dataset["X"]
    y      = dataset["y"]
    cat    = dataset["categorical"]
    scale  = dataset["scale"]
    models = make_models(cat, n_samples=len(X))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Acumula fold-metrics por modelo
    fold_results = {name: [] for name, _ in models}
    fold_times   = {name: [] for name, _ in models}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}", end="", flush=True)

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

        # Escala numericas se necessario
        if scale:
            num_cols = [c for c in X_tr.columns if c not in cat]
            sc = StandardScaler()
            X_tr = X_tr.copy()
            X_te = X_te.copy()
            X_tr[num_cols] = sc.fit_transform(X_tr[num_cols])
            X_te[num_cols] = sc.transform(X_te[num_cols])

        for name, model_proto in models:
            import sklearn
            m = sklearn.clone(model_proto)
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            proba = m.predict_proba(X_te)[:, 1]
            fold_results[name].append(metrics(y_te, proba))
            fold_times[name].append(elapsed)
            print(f"  {name.split()[0]}={fold_results[name][-1]['error']*100:.1f}%",
                  end="", flush=True)
        print()

    return [
        {
            "model": name,
            **avg(fold_results[name]),
            "time": float(np.mean(fold_times[name])),
        }
        for name, _ in models
    ]


# --------------------------------------------------------------------------- #
# Saida formatada                                                              #
# --------------------------------------------------------------------------- #

W = 82
SEP = "=" * W

def print_table(results: list[dict]):
    best_error = min(r["error"] for r in results)
    print(f"\n  {'Modelo':<22} {'Erro%':>7} {'AUC':>7} {'KS':>7} {'Lift@10%':>9} {'Treino(s)':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*10}")
    for r in results:
        star = " *" if r["error"] == best_error else "  "
        print(
            f"  {r['model']:<22}"
            f" {r['error']*100:>6.2f}%"
            f" {r['auc']:>7.4f}"
            f" {r['ks']:>7.4f}"
            f" {r['lift10']:>9.3f}"
            f" {r['time']:>9.2f}s"
            f"{star}"
        )
    print(f"\n  * melhor erro medio nos {5} folds")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    print(f"\n{SEP}")
    print(f"  fbtseg v{FBTSEG_VERSION}  —  Benchmark de Producao")
    print(f"  FBTSeg vs Logistic Regression vs Random Forest vs Gradient Boosting")
    print(f"  5-Fold Stratified Cross-Validation")
    print(f"{SEP}")
    print(f"\n  Instalacao:  pip install fbtseg")
    print(f"  GitHub:      https://github.com/RobertoASantos/FBT-Segmentation\n")

    datasets = load_datasets()
    all_results = {}

    for ds in datasets:
        print(f"\n{SEP}")
        print(f"  {ds['name'].upper()}  —  {ds['desc']}")
        print(f"{SEP}\n")
        results = run_cv(ds)
        all_results[ds["name"]] = results
        print_table(results)

    # Resumo final
    print(f"\n{SEP}")
    print(f"  RESUMO — Erro Medio (%) por Base")
    print(f"{SEP}")
    model_names = [r["model"] for r in next(iter(all_results.values()))]
    print(f"\n  {'Base':<18}", end="")
    for mn in model_names:
        print(f"  {mn.split()[0]:>9}", end="")
    print(f"  {'Vencedor'}")
    print(f"  {'-'*18}", end="")
    for _ in model_names:
        print(f"  {'-'*9}", end="")
    print(f"  {'-'*12}")

    fbtseg_wins = 0
    for ds_name, results in all_results.items():
        erros = {r["model"]: r["error"] for r in results}
        winner = min(erros, key=erros.get)
        if "FBTSeg" in winner:
            fbtseg_wins += 1
        print(f"  {ds_name:<18}", end="")
        for mn in model_names:
            mark = " *" if mn == winner else "  "
            print(f"  {erros[mn]*100:>7.2f}%{mark}", end="")
        print(f"  {winner.split()[0]}")

    print(f"\n  FBTSeg venceu em {fbtseg_wins}/{len(all_results)} bases.\n")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
