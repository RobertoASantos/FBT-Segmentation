"""Métricas D do RiskSeg conforme Capítulo 4 da tese.

Convenção: todas as funções de pontuação retornam um valor onde
**maior é melhor**. Para métricas naturalmente de minimização (erro,
1-AUC), são retornadas como o complemento.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def error_rate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Taxa de erro de classificação binária."""
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return float(np.mean(pred != y_true))


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if y_true.min() == y_true.max():
        return 0.5
    return float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float)))


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov clássico entre as distribuições de positivos e negativos.

    Coincide com a estatística do teste KS de duas amostras (KS2) citada
    na tese: max |F_1(s) - F_0(s)|.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    grid = np.sort(np.concatenate([pos, neg]))
    f_pos = np.searchsorted(np.sort(pos), grid, side="right") / pos.size
    f_neg = np.searchsorted(np.sort(neg), grid, side="right") / neg.size
    return float(np.max(np.abs(f_pos - f_neg)))


def lift_at_top(y_true: np.ndarray, y_prob: np.ndarray, top_rate: float = 0.10) -> float:
    """Lift no topo dos `top_rate` percentuais de maior escore."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_true.size
    if n == 0:
        return 0.0
    base = y_true.mean()
    if base <= 0:
        return 0.0
    k = max(1, int(np.ceil(n * top_rate)))
    order = np.argsort(-y_prob)
    top_rate_pos = y_true[order[:k]].mean()
    return float(top_rate_pos / base)


def precision_at_top(y_true: np.ndarray, y_prob: np.ndarray, top_rate: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_true.size
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(n * top_rate)))
    order = np.argsort(-y_prob)
    return float(y_true[order[:k]].mean())


def odds_ratio_bands(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bands: int = 10,
    eps: float = 1e-9,
) -> float:
    """Razão de chances (OR) entre a banda superior e a inferior do escore.

    Banda superior = `n_bands`-ésimo decil; inferior = primeiro decil.
    `OR = odds(banda_top) / odds(banda_bottom)`, com `odds = bad / good`.
    Métrica descrita na tese como D para estudos de risco de crédito.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 1.0
    quantiles = np.linspace(0, 1, n_bands + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if edges.size < 3:
        return 1.0
    band = np.clip(np.searchsorted(edges[1:-1], y_prob, side="right"), 0, edges.size - 2)
    top = band == band.max()
    bot = band == band.min()
    if not top.any() or not bot.any():
        return 1.0
    p_top = y_true[top].mean()
    p_bot = y_true[bot].mean()
    odds_top = p_top / max(1 - p_top, eps)
    odds_bot = p_bot / max(1 - p_bot, eps)
    return float(odds_top / max(odds_bot, eps))


METRIC_REGISTRY = {
    "error": lambda y, p: -error_rate(y, p),
    "auc": auc,
    "ks": ks_statistic,
    "ks2": ks_statistic,
    "lift": lift_at_top,
    "precision": precision_at_top,
    "odds_ratio": odds_ratio_bands,
}


def metric_score(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pontuação canônica (maior é melhor) para a métrica nomeada."""
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"metric '{metric}' não suportada. "
            f"Opções: {sorted(METRIC_REGISTRY.keys())}"
        )
    return float(METRIC_REGISTRY[metric](y_true, y_prob))


def all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Bloco completo de métricas para diagnóstico."""
    return {
        "error": error_rate(y_true, y_prob),
        "auc": auc(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "lift": lift_at_top(y_true, y_prob),
        "precision": precision_at_top(y_true, y_prob),
        "odds_ratio": odds_ratio_bands(y_true, y_prob),
    }
