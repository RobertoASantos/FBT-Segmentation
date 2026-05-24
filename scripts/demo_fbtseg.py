#!/usr/bin/env python
"""Demonstração do FBTSeg vs Regressão Logística em 3 bases UCI.

Reproduz o espírito da Tabela 1 do artigo ICAI 2012 (Santos & Barros, 2012):
compara o método de segmentação FBTSeg com o baseline simples (Regressão
Logística sem regularização) em 3 bases clássicas de risco/classificação.

Métricas avaliadas por 5-fold cross-validation estratificado:
  - Error Rate   (% de predições erradas)
  - AUC          (área sob a curva ROC — 1 = perfeito)
  - KS           (Kolmogorov-Smirnov — separação das distribuições)
  - Lift @ 10%   (quantas vezes mais positivos no top-10% vs random)

Bases utilizadas:
  - Chess    (3.196 inst | 36 vars categóricas | alvo: branco/preto vence)
  - German   (1.000 inst | 20 vars | risco de crédito)
  - Magic    (19.020 inst | 10 vars numéricas | partículas gama vs hadrão)

Uso:
  python scripts/demo_fbtseg.py

Referências:
  SANTOS, R. A. F.; BARROS, R. S. M. Comparing Segmentation Methods with
  Different Base Classifiers. ICAI 2012, Las Vegas, USA.
"""

from __future__ import annotations

import sys
import time
import warnings

# Garante UTF-8 no terminal Windows (evita erro de encoding em acentos)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from fbtseg import article_uci_preset, load_article_dataset
from fbtseg.datasets import get_spec

warnings.simplefilter("ignore", ConvergenceWarning)


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #

DATASETS = ["chess", "german", "magic"]
N_SPLITS = 5
RANDOM_STATE = 42

# Resultados publicados no paper para referência (Error Rate %)
PAPER_RESULTS = {
    "chess":  {"simple": 2.60, "fbtseg": 1.02},
    "german": {"simple": 27.50, "fbtseg": 24.90},
    "magic":  {"simple": 20.98, "fbtseg": 18.69},
}


# --------------------------------------------------------------------------- #
# Métricas                                                                     #
# --------------------------------------------------------------------------- #

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcula error rate, AUC, KS e Lift@10%."""
    y_pred = (y_proba >= 0.5).astype(int)

    # Error Rate
    error_rate = float((y_pred != y_true).mean())

    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = float(auc(fpr, tpr))

    # KS (máxima separação entre CDFs de positivos e negativos)
    order = np.argsort(y_proba)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        ks_score = 0.0
    else:
        cum_pos = np.cumsum(y_true[order]) / n_pos
        cum_neg = np.cumsum(1 - y_true[order]) / n_neg
        ks_score = float(np.max(np.abs(cum_pos - cum_neg)))

    # Lift @ 10%
    base_rate = y_true.mean()
    if base_rate == 0:
        lift = 1.0
    else:
        n_top = max(1, int(0.10 * len(y_true)))
        top_idx = np.argsort(y_proba)[-n_top:]
        lift = float(y_true[top_idx].mean() / base_rate)

    return {
        "error_rate": error_rate,
        "auc": auc_score,
        "ks": ks_score,
        "lift_10": lift,
    }


def mean_metrics(metric_list: list[dict]) -> dict:
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


# --------------------------------------------------------------------------- #
# Encoding para a Regressão Logística                                         #
# --------------------------------------------------------------------------- #

def fit_encoders(X, cat_cols) -> dict:
    """Ajusta LabelEncoders no dataset completo (evita unseen labels no CV)."""
    encoders = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            encoders[col] = le
    return encoders


def encode_X(X, encoders):
    """Aplica encoders pré-ajustados a um DataFrame."""
    Xc = X.copy()
    for col, le in encoders.items():
        if col in Xc.columns:
            Xc[col] = le.transform(Xc[col].astype(str))
    return Xc


# --------------------------------------------------------------------------- #
# Cross-validation de um dataset                                              #
# --------------------------------------------------------------------------- #

def run_cv(dataset_name: str) -> tuple[dict, dict, float, float]:
    """Executa 5-fold CV; devolve (metrics_fbtseg, metrics_lr, t_fbtseg, t_lr)."""
    spec = get_spec(dataset_name)
    X, y = load_article_dataset(spec)
    cat_cols = spec.categorical_columns

    # Encoders ajustados no dataset completo (evita unseen labels no CV)
    encoders = fit_encoders(X, cat_cols)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_fbtseg, fold_lr = [], []
    time_fbtseg, time_lr = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{N_SPLITS}...", end=" ", flush=True)

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values

        # --- FBTSeg ---
        model = article_uci_preset(categorical_features=cat_cols)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        t1 = time.perf_counter()
        proba = model.predict_proba(X_te)[:, 1]
        fold_fbtseg.append(compute_metrics(y_te, proba))
        time_fbtseg.append(t1 - t0)

        # --- Logistic Regression (sem regularização, igual ao paper) ---
        Xtr_enc = encode_X(X_tr, encoders)
        Xte_enc = encode_X(X_te, encoders)
        lr = LogisticRegression(penalty=None, max_iter=1000, random_state=RANDOM_STATE)
        t0 = time.perf_counter()
        lr.fit(Xtr_enc, y_tr)
        t1 = time.perf_counter()
        proba_lr = lr.predict_proba(Xte_enc)[:, 1]
        fold_lr.append(compute_metrics(y_te, proba_lr))
        time_lr.append(t1 - t0)

        print(
            f"fbtseg={fold_fbtseg[-1]['error_rate']*100:.1f}%  "
            f"lr={fold_lr[-1]['error_rate']*100:.1f}%"
        )

    return (
        mean_metrics(fold_fbtseg),
        mean_metrics(fold_lr),
        float(np.mean(time_fbtseg)),
        float(np.mean(time_lr)),
    )


# --------------------------------------------------------------------------- #
# Formatação da tabela final                                                  #
# --------------------------------------------------------------------------- #

W = 80

def section(title: str):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def header_row():
    print(
        f"  {'Dataset':<10}  {'Método':<22}  "
        f"{'Err%':>6}  {'AUC':>6}  {'KS':>6}  {'Lift@10%':>8}  {'Treino(s)':>9}"
    )
    print(f"  {'-'*8}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}")

def data_row(dataset, method, m, t, note=""):
    print(
        f"  {dataset:<10}  {method:<22}  "
        f"{m['error_rate']*100:>5.2f}%  "
        f"{m['auc']:>6.4f}  "
        f"{m['ks']:>6.4f}  "
        f"{m['lift_10']:>8.3f}  "
        f"{t:>9.2f}  {note}"
    )

def paper_ref_row(dataset):
    pr = PAPER_RESULTS.get(dataset)
    if not pr:
        return
    print(
        f"  {'':10}  {'  [paper FBTSeg]':<22}  "
        f"{pr['fbtseg']:>5.2f}%  {'---':>6}  {'---':>6}  {'---':>8}  {'---':>9}"
    )
    print(
        f"  {'':10}  {'  [paper Logistic]':<22}  "
        f"{pr['simple']:>5.2f}%  {'---':>6}  {'---':>6}  {'---':>8}  {'---':>9}"
    )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    print()
    print("=" * W)
    print("  FBTSeg  —  Find Best Tree Segmentation")
    print("  Comparação com Regressão Logística | 5-Fold Cross-Validation")
    print("  Bases: Chess, German, Magic  (UCI Repository)")
    print("=" * W)
    print(f"\n  Referência: Santos & Barros, ICAI 2012")
    print(f"  Pacote:     pip install fbtseg")
    print(f"  GitHub:     https://github.com/RobertoASantos/FBT-Segmentation")

    all_results = {}

    for ds in DATASETS:
        section(f"Dataset: {ds.upper()}")
        spec = get_spec(ds)
        X, y = load_article_dataset(spec)
        print(f"  {X.shape[0]:,} instâncias | {X.shape[1]} variáveis | "
              f"{len(spec.categorical_columns)} categóricas")
        print()
        m_fbtseg, m_lr, t_fbtseg, t_lr = run_cv(ds)
        all_results[ds] = (m_fbtseg, m_lr, t_fbtseg, t_lr)

    # Tabela final
    section("RESULTADO FINAL — Média dos 5 Folds")
    header_row()

    for ds in DATASETS:
        m_fbtseg, m_lr, t_fbtseg, t_lr = all_results[ds]

        # Ganho em error_rate
        ganho = (m_lr["error_rate"] - m_fbtseg["error_rate"]) * 100
        sinal = f"{'(+' if ganho > 0 else '('}{abs(ganho):.2f}% vs LR)"

        data_row(ds.capitalize(), "FBTSeg", m_fbtseg, t_fbtseg, sinal)
        data_row("", "Logistic Regression", m_lr, t_lr)
        paper_ref_row(ds)
        print()

    # Resumo comparativo
    section("RESUMO — Ganho Médio do FBTSeg sobre Regressão Logística")
    print()
    ganhos_error = []
    ganhos_auc   = []
    ganhos_ks    = []
    ganhos_lift  = []

    for ds in DATASETS:
        m_f, m_l, _, _ = all_results[ds]
        ganhos_error.append((m_l["error_rate"] - m_f["error_rate"]) * 100)
        ganhos_auc.append(m_f["auc"] - m_l["auc"])
        ganhos_ks.append(m_f["ks"] - m_l["ks"])
        ganhos_lift.append(m_f["lift_10"] - m_l["lift_10"])
        won = ganhos_error[-1] > 0
        marker = "V" if won else "X"
        print(f"  [{marker}] {ds.capitalize():<8}  "
              f"error rate {'+' if ganhos_error[-1]>0 else ''}{ganhos_error[-1]:.2f}%  |  "
              f"AUC {'+' if ganhos_auc[-1]>0 else ''}{ganhos_auc[-1]:.4f}  |  "
              f"KS {'+' if ganhos_ks[-1]>0 else ''}{ganhos_ks[-1]:.4f}  |  "
              f"Lift {'+' if ganhos_lift[-1]>0 else ''}{ganhos_lift[-1]:.3f}")

    print()
    print(f"  Ganho médio em Error Rate:  {np.mean(ganhos_error):+.2f}%  "
          f"({'melhora' if np.mean(ganhos_error) > 0 else 'piora'})")
    print(f"  Ganho médio em AUC:         {np.mean(ganhos_auc):+.4f}")
    print(f"  Ganho médio em KS:          {np.mean(ganhos_ks):+.4f}")
    print(f"  Ganho médio em Lift@10%:    {np.mean(ganhos_lift):+.3f}")
    print()
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
