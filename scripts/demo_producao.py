#!/usr/bin/env python
"""FBTSeg vs classificadores interpretáveis do scikit-learn.

    pip install fbtseg
    python scripts/demo_producao.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

try:
    import importlib.metadata
    from fbtseg import FBTSeg, load_article_dataset
    from fbtseg.datasets import get_spec
    VERSION = importlib.metadata.version("fbtseg")
except ImportError:
    print("Execute:  pip install fbtseg")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Bases                                                                        #
# --------------------------------------------------------------------------- #

BASES_CONFIG = [
    ("chess",    "Chess",   "3.196 amostras  |  36 vars categóricas  |  xadrez",    None),
    ("magic",    "Magic",   "19.020 amostras  |  10 vars numéricas  |  partículas", None),
    ("adult",    "Adult",   "10.000 amostras  |  14 vars mistas  |  renda >50K",    10_000),
]

def carregar_bases() -> list[dict]:
    bases = []
    for key, nome, desc, max_n in BASES_CONFIG:
        try:
            spec = get_spec(key)
            X, y = load_article_dataset(spec)
            if max_n and len(X) > max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), size=max_n, replace=False)
                idx.sort()
                X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
            bases.append({"nome": nome, "desc": desc, "X": X, "y": y,
                          "categoricas": spec.categorical_columns})
            print(f"  OK  {nome:<12} {X.shape[0]:>6} amostras, {X.shape[1]} variáveis")
        except Exception as e:
            print(f"  ERRO {nome}: {e}")
    return bases


# --------------------------------------------------------------------------- #
# Métricas                                                                     #
# --------------------------------------------------------------------------- #

def calcular(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    erro = float((y_pred != y_true).mean())
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_v = float(auc(fpr, tpr))
    ordem = np.argsort(y_proba)
    n_pos, n_neg = int(y_true.sum()), len(y_true) - int(y_true.sum())
    ks = 0.0
    if n_pos > 0 and n_neg > 0:
        cp = np.cumsum(y_true[ordem]) / n_pos
        cn = np.cumsum(1 - y_true[ordem]) / n_neg
        ks = float(np.max(np.abs(cp - cn)))
    base = y_true.mean()
    n_top = max(1, int(0.10 * len(y_true)))
    lift = float(y_true[np.argsort(y_proba)[-n_top:]].mean() / base) if base > 0 else 1.0
    return {"erro": erro, "auc": auc_v, "ks": ks, "lift10": lift}

def media(lista):
    return {k: float(np.mean([m[k] for m in lista])) for k in lista[0]}


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #

def criar_modelos(categoricas, n) -> list[tuple[str, object]]:
    min_leaf = 0.10 if n < 2000 else 0.05
    return [
        ("FBTSeg", FBTSeg(
            max_depth=3, min_samples_leaf=min_leaf, metric="error",
            top_k_variables=1, n_numeric_bins=4, validation_fraction=0.20,
            scale_numeric=True, combiner_method="stacking",
            categorical_features=categoricas, random_state=42,
        )),
        ("Logistic Regression", LogisticRegression(
            penalty=None, max_iter=1000, random_state=42,
        )),
        ("Decision Tree", DecisionTreeClassifier(
            max_depth=3, random_state=42,   # mesma profundidade do FBTSeg
        )),
        ("Naive Bayes", GaussianNB()),
    ]


# --------------------------------------------------------------------------- #
# Encoding para sklearn                                                        #
# --------------------------------------------------------------------------- #

def preparar_sklearn(X_tr, X_te, categoricas):
    num_cols = [c for c in X_tr.columns if c not in categoricas]
    Xtr, Xte = X_tr.copy(), X_te.copy()
    if num_cols:
        sc = StandardScaler()
        Xtr[num_cols] = sc.fit_transform(X_tr[num_cols])
        Xte[num_cols] = sc.transform(X_te[num_cols])
    for col in categoricas:
        if col in Xtr.columns:
            le = LabelEncoder()
            Xtr[col] = le.fit_transform(Xtr[col].astype(str))
            known = {c: i for i, c in enumerate(le.classes_)}
            Xte[col] = Xte[col].astype(str).map(known).fillna(0).astype(int)
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Cross-validation                                                             #
# --------------------------------------------------------------------------- #

def rodar_cv(base: dict, n_splits: int = 5) -> list[dict]:
    import sklearn
    X, y, cat = base["X"], base["y"], base["categoricas"]
    modelos = criar_modelos(cat, len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds_r = {n: [] for n, _ in modelos}
    folds_t = {n: [] for n, _ in modelos}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}", end="", flush=True)
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
        X_tr_sk, X_te_sk = preparar_sklearn(X_tr, X_te, cat)

        for nome, proto in modelos:
            m = sklearn.clone(proto)
            Xf_tr = X_tr if nome == "FBTSeg" else X_tr_sk
            Xf_te = X_te if nome == "FBTSeg" else X_te_sk
            t0 = time.perf_counter()
            m.fit(Xf_tr, y_tr)
            elapsed = time.perf_counter() - t0
            proba = m.predict_proba(Xf_te)[:, 1]
            folds_r[nome].append(calcular(y_te, proba))
            folds_t[nome].append(elapsed)
            print(f"  {nome.split()[0]}={folds_r[nome][-1]['erro']*100:.1f}%",
                  end="", flush=True)
        print()

    return [{"modelo": nome, **media(folds_r[nome]),
             "tempo": float(np.mean(folds_t[nome]))} for nome, _ in modelos]


# --------------------------------------------------------------------------- #
# Saída                                                                        #
# --------------------------------------------------------------------------- #

W = 80
SEP = "=" * W

def imprimir_tabela(resultados):
    melhor = min(r["erro"] for r in resultados)
    print(f"\n  {'Modelo':<22} {'Erro%':>7} {'Acurácia':>9} {'AUC':>7} {'KS':>7} {'Lift@10':>8} {'Tempo':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for r in resultados:
        tag = "  <--" if r["erro"] == melhor else ""
        print(f"  {r['modelo']:<22}"
              f" {r['erro']*100:>6.2f}%"
              f" {(1-r['erro'])*100:>8.2f}%"
              f" {r['auc']:>7.4f}"
              f" {r['ks']:>7.4f}"
              f" {r['lift10']:>8.3f}"
              f" {r['tempo']:>6.2f}s{tag}")


def main():
    print(f"\n{SEP}")
    print(f"  FBTSeg v{VERSION}  —  vs classificadores interpretáveis sklearn")
    print(f"  5-Fold Stratified Cross-Validation")
    print(f"{SEP}\n")

    print("  Carregando bases (download automático na 1ª vez)...\n")
    bases = carregar_bases()
    todos = {}

    for base in bases:
        print(f"\n{SEP}")
        print(f"  {base['nome'].upper()}  —  {base['desc']}")
        print(f"{SEP}\n")
        res = rodar_cv(base)
        todos[base["nome"]] = res
        imprimir_tabela(res)

    # Resumo
    print(f"\n{SEP}")
    print(f"  RESUMO  —  Erro Médio (%)  [ <-- melhor ]")
    print(f"{SEP}")
    nomes = [r["modelo"] for r in next(iter(todos.values()))]
    col = 13
    print(f"\n  {'Base':<14}" + "".join(f"  {n.split()[0]:>{col}}" for n in nomes))
    print(f"  {'-'*14}" + "".join(f"  {'-'*col}" for _ in nomes))

    fbtseg_wins = 0
    for base_nome, res in todos.items():
        erros = {r["modelo"]: r["erro"] for r in res}
        venc = min(erros, key=erros.get)
        if venc == "FBTSeg":
            fbtseg_wins += 1
        linha = f"  {base_nome:<14}"
        for nm in nomes:
            val = f"{erros[nm]*100:.2f}%"
            mark = " *" if nm == venc else "  "
            linha += f"  {val+mark:>{col}}"
        print(linha)

    print(f"\n  FBTSeg venceu em {fbtseg_wins}/{len(todos)} bases.\n")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
