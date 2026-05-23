"""Base learners adicionais para uso no `FBTSeg`.

O paper ICAI 2012 (Santos & Barros, 2012a) compara FBTSeg cruzado com
**três** base learners diferentes:

- **Regressão Logística** (LR sem regularização) — disponível
  diretamente via `sklearn.linear_model.LogisticRegression(penalty=None)`.
- **Regressão Linear** (probabilidade linear, com clip em [0, 1]) —
  exportada aqui como `LinearProbabilityClassifier`.
- **MLP Neural Network** — usar `sklearn.neural_network.MLPClassifier`
  diretamente. O `FBTSeg._fit_clone` tolera estimadores que não
  aceitam `sample_weight` (cai no caminho sem peso).

A intuição por trás de oferecer Regressão Linear como base: é o pior
classificador entre os três no agregado, MAS é exatamente onde o
FBTSeg mostra os ganhos mais dramáticos (Magic 32% → 16% no paper,
porque a segmentação compensa a incapacidade da regressão linear de
modelar interações).

Referências (`docs/references.md`):
- SANTOS & BARROS, 2012 (ICAI) — Seção 2, Seção 3 (base learners).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression


class LinearProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Regressão linear ajustada contra `y` em {0, 1}, com escore truncado.

    Replica a configuração "Linear Regression" do artigo ICAI 2012,
    onde o alvo dicotômico é tratado como contínuo num modelo linear
    e a probabilidade prevista é o output cru, **truncado em [0, 1]**
    para virar uma probabilidade válida (modelo de probabilidade
    linear, MPL).

    Limitações conhecidas:
    - extrapola para fora de [0, 1] sem o clip;
    - heterocedástico por construção;
    - mas é o que o paper usou e é por isso que está aqui.

    Compatível com a API sklearn: `fit`, `predict_proba`, `predict`,
    `decision_function`, suporta `clone()`.
    """

    def __init__(self):
        # Sem hiperparâmetros — manter `__init__` vazio é importante
        # para que `sklearn.base.clone()` funcione sem warning.
        pass

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Convenção sklearn: `classes_` ordenado.
        self.classes_ = np.array(sorted(np.unique(y).astype(int)))
        if self.classes_.size < 2:
            # Caso degenerado: só uma classe. Vira "classificador constante".
            self.constant_ = float(y.mean()) if y.size else 0.5
            self._fitted = "constant"
            return self
        self.model_ = LinearRegression()
        if sample_weight is not None:
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)
        self._fitted = "linear"
        return self

    def predict_proba(self, X):
        """Devolve `[1-p, p]` com `p = clip(linear(X), 0, 1)`."""
        X = np.asarray(X, dtype=float)
        if self._fitted == "constant":
            n = X.shape[0]
            return np.full((n, 2), [1 - self.constant_, self.constant_], dtype=float)
        # Escore cru da regressão linear, truncado em [0, 1].
        p1 = np.clip(self.model_.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        """Escore (probabilidade da classe positiva) — sem truncamento explícito."""
        return self.predict_proba(X)[:, 1]
