#!/usr/bin/env python
"""FBTSeg vs técnicas de classificação binária do scikit-learn.

Compara o método FBTSeg com os principais classificadores do sklearn
em 3 bases clássicas, usando 5-fold cross-validation estratificado.

Uso:
    pip install fbtseg
    python scripts/demo_producao.py

Bases:
    Chess    — 3.196 amostras, 36 variáveis categóricas
    Magic    — 19.020 amostras, 10 variáveis numéricas
    Spambase — 4.601 amostras, 57 variáveis numéricas

Modelos:
    FBTSeg                — este pacote
    Logistic Regression   — sklearn
    Random Forest         — sklearn
    Gradient Boosting     — sklearn
    SVM (RBF)             — sklearn
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

# --------------------------------------------------------------------------- #
# fbtseg                                                                       #
# --------------------------------------------------------------------------- #

try:
    import importlib.metadata
    from fbtseg import FBTSeg, load_article_dataset
    from fbtseg.datasets import get_spec
    VERSION = importlib.metadata.version("fbtseg")
except ImportError:
    print("ERRO: execute  pip install fbtseg  antes de rodar este script.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Bases de dados                                                               #
# --------------------------------------------------------------------------- #

def carregar_bases() -> list[dict]:
    bases = []

    # 1. Chess — 36 variáveis categóricas, FBTSeg faz segmentação natural
    try:
        sp = get_spec("chess")
        X, y = load_article_dataset(sp)
        bases.append({
            "nome": "Chess",
            "desc": "3.196 amostras  |  36 vars categóricas  |  branco vs preto vence",
            "X": X, "y": y,
            "categoricas": sp.categorical_columns,
        })
    except Exception as e:
        print(f"  [aviso] Chess não carregado: {e}")

    # 2. Magic — 19k amostras numéricas, não-linearidade que LR não captura
    try:
        sp = get_spec("magic")
        X, y = load_article_dataset(sp)
        bases.append({
            "nome": "Magic",
            "desc": "19.020 amostras  |  10 vars numéricas  |  gamma vs hadrão",
            "X": X, "y": y,
            "categoricas": (),
        })
    except Exception as e:
        print(f"  [aviso] Magic não carregado: {e}")

    # 3. Spambase — 57 variáveis numéricas, dataset grande
    try:
        sp = get_spec("spambase")
        X, y = load_article_dataset(sp)
        bases.append({
            "nome": "Spambase",
            "desc": "4.601 amostras  |  57 vars numéricas  |  spam vs legítimo",
            "X": X, "y": y,
            "categoricas": (),
        })
    except Exception as e:
        print(f"  [aviso] Spambase não carregado: {e}")

    return bases


# --------------------------------------------------------------------------- #
# Métricas                                                                     #
# --------------------------------------------------------------------------- #

def calcular_metricas(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)

    acuracia   = float((y_pred == y_true).mean())
    erro       = 1.0 - acuracia
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score  = float(auc(fpr, tpr))

    # KS — máxima separação entre distribuições de positivos e negativos
    ordem  = np.argsort(y_proba)
    n_pos  = int(y_true.sum())
    n_neg  = len(y_true) - n_pos
    ks     = 0.0
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(y_true[ordem]) / n_pos
        cum_neg = np.cumsum(1 - y_true[ordem]) / n_neg
        ks = float(np.max(np.abs(cum_pos - cum_neg)))

    # Lift @ 10% — quantas vezes mais positivos no top-10% vs aleatório
    base  = y_true.mean()
    n_top = max(1, int(0.10 * len(y_true)))
    top10 = np.argsort(y_proba)[-n_top:]
    lift  = float(y_true[top10].mean() / base) if base > 0 else 1.0

    return {"erro": erro, "acuracia": acuracia, "auc": auc_score, "ks": ks, "lift10": lift}


def media_metricas(lista: list[dict]) -> dict:
    return {k: float(np.mean([m[k] for m in lista])) for k in lista[0]}


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #

def criar_modelos(categoricas: tuple, n_amostras: int) -> list[tuple[str, object]]:
    # FBTSeg: parâmetros alinhados ao preset padrão
    # scale_numeric=True já está dentro do FBTSeg — não pré-escalar externamente
    min_leaf = 0.10 if n_amostras < 1500 else 0.05
    return [
        ("FBTSeg", FBTSeg(
            max_depth=3,
            min_samples_leaf=min_leaf,
            metric="error",
            top_k_variables=1,
            n_numeric_bins=4,
            validation_fraction=0.20,
            scale_numeric=True,
            combiner_method="stacking",
            categorical_features=categoricas,
            random_state=42,
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
        ("SVM (RBF)", SVC(
            kernel="rbf", probability=True, random_state=42,
        )),
    ]


# --------------------------------------------------------------------------- #
# Encoding de categóricas para modelos sklearn                                 #
# --------------------------------------------------------------------------- #

def encode(X_tr: pd.DataFrame, X_te: pd.DataFrame, categoricas: tuple):
    """Label-encode categóricas ajustando no treino + aplicando no teste."""
    Xtr, Xte = X_tr.copy(), X_te.copy()
    for col in categoricas:
        if col in Xtr.columns:
            le = LabelEncoder()
            Xtr[col] = le.fit_transform(Xtr[col].astype(str))
            # Mapeia categorias desconhecidas no teste para 0
            mapping = {c: i for i, c in enumerate(le.classes_)}
            Xte[col] = Xte[col].astype(str).map(mapping).fillna(0).astype(int)
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Cross-validation                                                             #
# --------------------------------------------------------------------------- #

def rodar_cv(base: dict, n_splits: int = 5) -> list[dict]:
    import sklearn
    X, y       = base["X"], base["y"]
    categoricas = base["categoricas"]
    modelos    = criar_modelos(categoricas, n_amostras=len(X))
    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    resultados_fold  = {nome: [] for nome, _ in modelos}
    tempos_fold      = {nome: [] for nome, _ in modelos}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}", end="", flush=True)

        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

        # Escala + encoding para modelos sklearn (não para FBTSeg — escala internamente)
        num_cols = [c for c in X_tr.columns if c not in categoricas]
        X_tr_sk, X_te_sk = X_tr.copy(), X_te.copy()
        if num_cols:
            sc = StandardScaler()
            X_tr_sk[num_cols] = sc.fit_transform(X_tr[num_cols])
            X_te_sk[num_cols] = sc.transform(X_te[num_cols])
        X_tr_enc, X_te_enc = encode(X_tr_sk, X_te_sk, categoricas)

        for nome, proto in modelos:
            m = sklearn.clone(proto)
            # FBTSeg recebe dados originais (strings categóricas + numéricos raw)
            Xf_tr = X_tr if nome == "FBTSeg" else X_tr_enc
            Xf_te = X_te if nome == "FBTSeg" else X_te_enc

            t0 = time.perf_counter()
            m.fit(Xf_tr, y_tr)
            elapsed = time.perf_counter() - t0

            proba = m.predict_proba(Xf_te)[:, 1]
            resultados_fold[nome].append(calcular_metricas(y_te, proba))
            tempos_fold[nome].append(elapsed)
            print(f"  {nome.split()[0]}={resultados_fold[nome][-1]['erro']*100:.1f}%",
                  end="", flush=True)
        print()

    return [
        {"modelo": nome, **media_metricas(resultados_fold[nome]),
         "tempo": float(np.mean(tempos_fold[nome]))}
        for nome, _ in modelos
    ]


# --------------------------------------------------------------------------- #
# Saída                                                                        #
# --------------------------------------------------------------------------- #

W   = 84
SEP = "=" * W

def tabela(resultados: list[dict]):
    melhor_erro = min(r["erro"] for r in resultados)
    print(f"\n  {'Modelo':<22} {'Erro%':>7} {'Acurácia':>9} {'AUC':>7} {'KS':>7} {'Lift@10':>8} {'Tempo':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for r in resultados:
        destaque = " <--" if r["erro"] == melhor_erro else ""
        print(
            f"  {r['modelo']:<22}"
            f" {r['erro']*100:>6.2f}%"
            f" {r['acuracia']*100:>8.2f}%"
            f" {r['auc']:>7.4f}"
            f" {r['ks']:>7.4f}"
            f" {r['lift10']:>8.3f}"
            f" {r['tempo']:>6.2f}s"
            f"{destaque}"
        )


def main():
    print(f"\n{SEP}")
    print(f"  FBTSeg v{VERSION}  —  Comparação com técnicas sklearn")
    print(f"  5-Fold Stratified Cross-Validation")
    print(f"{SEP}")
    print(f"\n  pip install fbtseg")
    print(f"  https://github.com/RobertoASantos/FBT-Segmentation\n")

    bases = carregar_bases()
    todos = {}

    for base in bases:
        print(f"\n{SEP}")
        print(f"  {base['nome'].upper()}  —  {base['desc']}")
        print(f"{SEP}\n")
        res = rodar_cv(base)
        todos[base["nome"]] = res
        tabela(res)

    # Resumo consolidado
    print(f"\n{SEP}")
    print(f"  RESUMO  —  Erro Médio (%) por Base  [ <-- melhor ]")
    print(f"{SEP}")

    nomes_modelos = [r["modelo"] for r in next(iter(todos.values()))]
    col = 14
    print(f"\n  {'Base':<18}" + "".join(f"  {n.split()[0]:>{col}}" for n in nomes_modelos))
    print(f"  {'-'*18}" + "".join(f"  {'-'*col}" for _ in nomes_modelos))

    fbtseg_vitorias = 0
    for nome_base, res in todos.items():
        erros   = {r["modelo"]: r["erro"] for r in res}
        vencedor = min(erros, key=erros.get)
        if vencedor == "FBTSeg":
            fbtseg_vitorias += 1
        linha = f"  {nome_base:<18}"
        for nm in nomes_modelos:
            val  = f"{erros[nm]*100:.2f}%"
            mark = " *" if nm == vencedor else "  "
            linha += f"  {val+mark:>{col}}"
        print(linha)

    print(f"\n  FBTSeg: melhor erro em {fbtseg_vitorias}/{len(todos)} bases.\n")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
