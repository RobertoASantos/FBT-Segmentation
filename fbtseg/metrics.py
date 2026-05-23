"""Métricas D do FBTSeg conforme Capítulo 4 da tese de Santos (2010).

A métrica `D` é o critério de desempenho usado pelo FBTSeg para:

1. Ranquear variáveis candidatas no **screening fatorial**
   (`_screen_candidate_variables` em `estimator.py`).
2. Comparar splits candidatos contra o baseline do nó pai
   (`_best_split_for_variable`).
3. Decidir se o ganho/perda do split é aceitável
   (`min_gain_pct` / `max_loss_pct`).

A tese (Cap. 4, Seção 4.2.1.6, parâmetro `D`) menciona quatro métricas
explicitamente como `D` válidas: **erro de classificação**, **KS2**
(Kolmogorov-Smirnov de duas amostras), **ROC/AUC** e **Odds Ratio** por
faixas de escore (esta última especialmente útil para risco de crédito).

Convenção interna: todas as funções de pontuação retornam um valor
onde **maior é melhor**. Para métricas naturalmente de minimização
(erro, 1-AUC), são retornadas como o complemento. Isso simplifica o
código de seleção: sempre `argmax`.

Referências (`docs/references.md`):
- SANTOS, 2010 — Cap. 4 da tese (define `D`).
- CONOVER, 1999 — *Practical Nonparametric Statistics* (KS2).
- FAWCETT, 2006 — *An Introduction to ROC Analysis*.
- THOMAS, EDELMAN & CROOK, 2002 — Odds Ratio em credit scoring.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def error_rate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Taxa de erro de classificação binária com corte em `threshold`.

    No paper ICAI 2012, esta é a métrica usada em todos os experimentos
    (Seção 3.2: "The metric used to evaluate the results of the
    experiments was the classification error").

    Parameters
    ----------
    y_true : array-like de inteiros em {0, 1}.
    y_prob : array-like com escore/probabilidade da classe positiva.
    threshold : ponto de corte (default 0.5 = decisão de Bayes para
        prior balanceado).

    Returns
    -------
    Erro de classificação no intervalo [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return float(np.mean(pred != y_true))


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Área sob a curva ROC (Fawcett, 2006).

    A tese cita `ROC` como uma das métricas `D` válidas (Seção 4.2.1.6).
    Retorna `0.5` se há apenas uma classe (evita exceção do sklearn).
    """
    y_true = np.asarray(y_true, dtype=int)
    if y_true.min() == y_true.max():
        return 0.5  # caso degenerado — sem AUC bem definida
    return float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float)))


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov clássico entre as distribuições de positivos e negativos.

    Calcula `max |F_1(s) - F_0(s)|`, onde:
    - `F_1` é a CDF empírica dos escores das observações positivas,
    - `F_0` é a CDF empírica dos escores das negativas.

    Esta é exatamente a **estatística do teste KS de duas amostras**
    (Conover, 1999), citada na tese como **KS2**. Em risco de crédito,
    KS é a métrica padrão da indústria para separação de classes
    (Thomas, Edelman & Crook, 2002).

    Implementação vetorizada: ordena os escores positivos e negativos,
    avalia ambas as CDFs num único grid (a união dos pontos) e tira o
    máximo do |delta|.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    # Grid = todos os pontos onde alguma CDF pode "saltar"
    grid = np.sort(np.concatenate([pos, neg]))
    # CDFs avaliadas no grid (vetorizado via searchsorted)
    f_pos = np.searchsorted(np.sort(pos), grid, side="right") / pos.size
    f_neg = np.searchsorted(np.sort(neg), grid, side="right") / neg.size
    return float(np.max(np.abs(f_pos - f_neg)))


def lift_at_top(y_true: np.ndarray, y_prob: np.ndarray, top_rate: float = 0.10) -> float:
    """Lift no topo dos `top_rate * 100` % de maior escore.

    `lift = taxa_de_positivos_no_top / taxa_global_de_positivos`.

    Métrica clássica de marketing/risco. Não está listada explicitamente
    na tese como `D`, mas é útil quando a aplicação prioriza identificar
    bem a cauda direita do escore (ex.: oferecer condições especiais
    para os melhores clientes — Cap. 4, Seção 4.2.1.6 do `D`).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_true.size
    if n == 0:
        return 0.0
    base = y_true.mean()
    if base <= 0:
        return 0.0  # sem positivos => lift indefinido => devolve 0
    k = max(1, int(np.ceil(n * top_rate)))
    order = np.argsort(-y_prob)  # decrescente para pegar o topo
    top_rate_pos = y_true[order[:k]].mean()
    return float(top_rate_pos / base)


def precision_at_top(y_true: np.ndarray, y_prob: np.ndarray, top_rate: float = 0.10) -> float:
    """Precisão (fração de positivos) nos `top_rate * 100` % maiores escores.

    Complementar ao `lift_at_top`: enquanto o lift escala pela taxa
    base, a precisão é absoluta.
    """
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

    Particiona o escore em `n_bands` faixas de quantis (decis se
    `n_bands=10`) e calcula:

        OR = odds(banda_top) / odds(banda_bottom)

    onde `odds(banda) = p(banda) / (1 - p(banda))` e `p(banda)` é a
    taxa de positivos na banda.

    Esta é a versão da métrica Odds Ratio mencionada na tese
    (Cap. 4, Seção 4.2.1.6, parâmetro `D`) e em Thomas, Edelman &
    Crook (2002), especialmente útil para estudos de risco de crédito
    quando se quer maximizar a separação entre o "topo bom" e o
    "fundo ruim" do escore.

    `eps` evita divisão por zero quando uma das bandas tem p ≈ 1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 1.0
    quantiles = np.linspace(0, 1, n_bands + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if edges.size < 3:
        # escores quase constantes — sem banda significativa
        return 1.0
    # `band[i]` é o índice da banda em que `y_prob[i]` caiu
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


# Registro central: nome -> função "maior é melhor".
# Note que `error` é negativado para que o argmax escolha o menor erro.
METRIC_REGISTRY = {
    "error": lambda y, p: -error_rate(y, p),
    "auc": auc,
    "ks": ks_statistic,
    "ks2": ks_statistic,  # alias — o "KS" da tese é o KS2 estatístico
    "lift": lift_at_top,
    "precision": precision_at_top,
    "odds_ratio": odds_ratio_bands,
}


def metric_score(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pontuação canônica (maior é melhor) para a métrica nomeada.

    Esta função é chamada em vários pontos do estimador
    (`_screen`, `_best_split_for_variable`) sempre que precisamos
    comparar candidatos. O contrato "maior é melhor" deixa o código
    de seleção trivial: `best = argmax(scores)`.
    """
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"metric '{metric}' não suportada. "
            f"Opções: {sorted(METRIC_REGISTRY.keys())}"
        )
    return float(METRIC_REGISTRY[metric](y_true, y_prob))


def all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Bloco completo de métricas para diagnóstico e relatórios.

    Diferente de `metric_score`, aqui as métricas naturais são
    retornadas em sua escala original (ex.: `error` em [0, 1], não
    negativado). Útil para `model.evaluate(X, y)` e para logs.
    """
    return {
        "error": error_rate(y_true, y_prob),
        "auc": auc(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "lift": lift_at_top(y_true, y_prob),
        "precision": precision_at_top(y_true, y_prob),
        "odds_ratio": odds_ratio_bands(y_true, y_prob),
    }
