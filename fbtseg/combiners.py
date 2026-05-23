"""Funções de combinação `f` descritas no Capítulo 4 da tese de Santos (2010).

A "função de combinação" `f` (parâmetro `combiner_method` do
estimador) recebe os escores produzidos por dois modelos especialistas
— um para cada segmento da divisão binária — e devolve um escore
**alinhado** ao alvo. Ela serve a dois propósitos:

1. Permite que a métrica `D` seja calculada sobre uma escala comum,
   viabilizando a comparação entre o par (esquerda, direita) e o
   modelo simples do nó pai.
2. Permite que segmentos treinados com **técnicas diferentes**
   (ex.: logística num lado, MLP no outro) produzam escores numa
   mesma escala probabilística — uma das vantagens didáticas que a
   tese enfatiza (Cap. 4, Seção 4.2.1.2).

Duas implementações são oferecidas, exatamente como na tese (Cap. 4,
Seção 4.2.1.3):

* `StackingCombiner` — Wolpert (1992).
  Constrói a matriz Z = [score_segmento_pertencente, 0]: cada linha
  tem o escore do modelo do segmento ao qual o registro pertence
  numa coluna e zero na outra. Uma regressão logística é ajustada
  sobre Z contra o alvo, produzindo o escore final.

* `MarginalOddsCombiner` — Thomas, Edelman & Crook (2002).
  Recalibra os escores via regressão logística por segmento, escolhe
  um segmento como **referência** e usa os coeficientes dessa
  logística para mapear o escore do outro segmento, alinhando ambos
  numa mesma escala de log-odds.

Referências (`docs/references.md`):
- WOLPERT, 1992 — Stacked Generalization.
- THOMAS, EDELMAN & CROOK, 2002 — Credit Scoring and its Applications.
- SANTOS, 2010 — Tese, Cap. 4, Seção 4.2.1.3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


# Tolerância para evitar log(0) / log(1) ao computar logits.
_EPS = 1e-9


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Logit numericamente estável: clipa p em [eps, 1-eps] antes do log."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _safe_sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide com clipping em |z| <= 50 para evitar overflow no exp()."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))


@dataclass
class StackingCombiner:
    """Stacking de Wolpert via regressão logística sobre `[score_segmento, 0]`.

    Construção da matriz Z (vide tese, Cap. 4, Seção 4.2.1.3):
        Para cada observação i:
            Z[i, 0] = score_left(i) se i pertence ao segmento esquerdo, 0 c.c.
            Z[i, 1] = score_right(i) se i pertence ao segmento direito, 0 c.c.
        (Exatamente uma das colunas é não-zero por linha.)

    Ajusta `LogisticRegression` sobre `(Z, y)`. O escore final é
    `model.predict_proba(Z)[:, 1]`.

    `penalty=None` (default) reproduz o comportamento de uma regressão
    logística "pura" (máxima verossimilhança sem regularização),
    análogo ao `PROC LOGISTIC` do SAS Enterprise Miner usado no paper.
    """

    penalty: str | None = None
    C: float = 1e6
    max_iter: int = 2000
    random_state: int = 42
    # Atributos pós-fit:
    model_: LogisticRegression | None = None
    is_constant_: bool = False
    constant_p_: float = 0.5

    def fit(
        self,
        score_left: np.ndarray,
        score_right: np.ndarray,
        membership: np.ndarray,
        y: np.ndarray,
    ) -> "StackingCombiner":
        """Ajusta a logística do stacking.

        Parameters
        ----------
        score_left : escore predito pelo modelo do segmento esquerdo,
            válido apenas onde `membership == 0`.
        score_right : escore predito pelo modelo do segmento direito,
            válido apenas onde `membership == 1`.
        membership : vetor em {0, 1} indicando a qual segmento cada
            observação pertence.
        y : alvo binário em {0, 1}.
        """
        membership = np.asarray(membership, dtype=int)
        y = np.asarray(y, dtype=int)
        n = membership.size

        # Constrói Z: cada linha tem só uma coluna não-zero.
        Z = np.zeros((n, 2), dtype=float)
        Z[membership == 0, 0] = np.asarray(score_left, dtype=float)[membership == 0]
        Z[membership == 1, 1] = np.asarray(score_right, dtype=float)[membership == 1]

        # Proteção: se y é constante (uma única classe), nem cabe LR.
        if np.unique(y).size < 2:
            self.is_constant_ = True
            self.constant_p_ = float(y.mean()) if y.size else 0.5
            return self

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            # quando penalty=None, C é ignorado pelo sklearn (>=1.4)
            C=self.C if self.penalty else 1.0,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        # `lbfgs` pode emitir `ConvergenceWarning` em segmentos pequenos —
        # silenciamos porque o estimador chamador trata isso uma vez no fit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(Z, y)
        return self

    def predict_proba(
        self,
        score_left: np.ndarray,
        score_right: np.ndarray,
        membership: np.ndarray,
    ) -> np.ndarray:
        """Aplica o stacking treinado e devolve `[1-p, p]` para cada linha."""
        membership = np.asarray(membership, dtype=int)
        n = membership.size
        Z = np.zeros((n, 2), dtype=float)
        Z[membership == 0, 0] = np.asarray(score_left, dtype=float)[membership == 0]
        Z[membership == 1, 1] = np.asarray(score_right, dtype=float)[membership == 1]
        if self.is_constant_ or self.model_ is None:
            return np.full((n, 2), [1 - self.constant_p_, self.constant_p_], dtype=float)
        return self.model_.predict_proba(Z)


@dataclass
class MarginalOddsCombiner:
    """Recalibração por Marginal Odds com segmento de referência.

    Procedimento (Thomas, Edelman & Crook, 2002, Cap. 8; citado na
    tese de Santos, Cap. 4, Seção 4.2.1.3):

    1. Para cada segmento `s ∈ {left, right}`:
       ajusta `LR(logit(score_s) → y_s)` produzindo `(a_s, b_s)`.
       Esse passo equivale a estimar a log-odds verdadeira em função
       do escore bruto: `logit(P(y=1|score)) ≈ a_s + b_s * logit(score)`.
    2. Escolhe um segmento de **referência** (default `auto` = o maior).
    3. Em predição:
       - segmento de referência: aplica seus próprios `(a, b)`;
       - segmento "outro": aplica os `(a, b)` da referência ao seu
         escore bruto, alinhando as duas escalas de log-odds.

    A diferença para o `StackingCombiner` é importante: o stacking
    aprende um peso conjunto para os dois escores (assume que ambos
    estão na mesma escala); o Marginal Odds **força** que ambos
    estejam na escala da referência aplicando os mesmos coeficientes.

    Em código antigo (V1), este combiner ajustava duas logísticas
    independentes e simplesmente concatenava os resultados — perdendo
    o passo de alinhamento. Esta implementação corrige esse desvio.
    """

    max_iter: int = 2000
    random_state: int = 42
    reference: str = "auto"  # 'left' | 'right' | 'auto' (maior n)

    # Coeficientes da logística por segmento, preenchidos no fit:
    intercept_left_: float = 0.0
    slope_left_: float = 1.0
    intercept_right_: float = 0.0
    slope_right_: float = 1.0
    # Flags de segmento degenerado (uma única classe):
    is_constant_left_: bool = False
    is_constant_right_: bool = False
    constant_p_left_: float = 0.5
    constant_p_right_: float = 0.5
    # Resolução final do segmento de referência ('left' ou 'right'):
    reference_resolved_: str = "left"

    def _fit_segment(self, score, y):
        """Ajusta `LR(logit(score) → y)` num segmento isolado.

        Devolve a tupla `(is_constant, constant_p, intercept, slope)`.
        Se o segmento é degenerado (sem amostras ou só uma classe),
        marca como constante e devolve `p = média(y)`.
        """
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=int)
        if y.size == 0 or np.unique(y).size < 2:
            return True, float(y.mean()) if y.size else 0.5, 0.0, 1.0
        # logit(score) como única feature → R^1 mapeado para R^1.
        x = _safe_logit(score).reshape(-1, 1)
        m = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(x, y)
        return False, 0.0, float(m.intercept_[0]), float(m.coef_[0, 0])

    def fit(
        self,
        score_left: np.ndarray,
        score_right: np.ndarray,
        membership: np.ndarray,
        y: np.ndarray,
    ) -> "MarginalOddsCombiner":
        """Ajusta a calibração por segmento e escolhe o segmento de referência."""
        membership = np.asarray(membership, dtype=int)
        y = np.asarray(y, dtype=int)
        score_left = np.asarray(score_left, dtype=float)
        score_right = np.asarray(score_right, dtype=float)

        left_mask = membership == 0
        right_mask = membership == 1

        # Cada segmento aprende sua própria reta (intercept, slope) no
        # espaço de logit(score) -> y.
        (
            self.is_constant_left_,
            self.constant_p_left_,
            self.intercept_left_,
            self.slope_left_,
        ) = self._fit_segment(score_left[left_mask], y[left_mask])
        (
            self.is_constant_right_,
            self.constant_p_right_,
            self.intercept_right_,
            self.slope_right_,
        ) = self._fit_segment(score_right[right_mask], y[right_mask])

        # Escolha do segmento de referência: por default, o que tem
        # mais observações (mais robusto). O usuário pode forçar.
        if self.reference == "auto":
            self.reference_resolved_ = "left" if left_mask.sum() >= right_mask.sum() else "right"
        else:
            self.reference_resolved_ = self.reference
        return self

    def predict_proba(
        self,
        score_left: np.ndarray,
        score_right: np.ndarray,
        membership: np.ndarray,
    ) -> np.ndarray:
        """Aplica a recalibração alinhada e devolve `[1-p, p]`.

        - Segmento de referência: usa seus próprios `(a, b)`.
        - Segmento "outro": **aplica os `(a, b)` da referência** ao
          logit do seu escore, projetando-o na escala do segmento de
          referência.
        """
        membership = np.asarray(membership, dtype=int)
        n = membership.size
        p = np.full(n, 0.5, dtype=float)

        left_mask = membership == 0
        right_mask = membership == 1

        ref = self.reference_resolved_
        # `(a, b)` = coeficientes da REFERÊNCIA.
        # `(a_other, b_other)` = coeficientes do segmento NÃO-referência
        # (mantidos para diagnóstico, mas NÃO usados na projeção).
        if ref == "left":
            a, b = self.intercept_left_, self.slope_left_
            a_other, b_other = self.intercept_right_, self.slope_right_
            is_const_ref = self.is_constant_left_
            const_ref = self.constant_p_left_
            is_const_other = self.is_constant_right_
            const_other = self.constant_p_right_
        else:
            a, b = self.intercept_right_, self.slope_right_
            a_other, b_other = self.intercept_left_, self.slope_left_
            is_const_ref = self.is_constant_right_
            const_ref = self.constant_p_right_
            is_const_other = self.is_constant_left_
            const_other = self.constant_p_left_

        # Aplica a transformação alinhada em cada lado.
        if ref == "left":
            # Segmento de referência (esquerdo) — usa próprios coefs.
            if is_const_ref:
                p[left_mask] = const_ref
            else:
                p[left_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_left[left_mask])
                )
            # Outro segmento (direito) — aplica coefs da REFERÊNCIA
            # sobre o escore do segmento direito. Esse é o passo de
            # "alinhamento" descrito por Thomas, Edelman & Crook (2002).
            if is_const_other:
                p[right_mask] = const_other
            else:
                p[right_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_right[right_mask])
                )
        else:
            # Mesma lógica, com 'right' como referência.
            if is_const_ref:
                p[right_mask] = const_ref
            else:
                p[right_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_right[right_mask])
                )
            if is_const_other:
                p[left_mask] = const_other
            else:
                p[left_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_left[left_mask])
                )

        return np.column_stack([1 - p, p])


def build_combiner(name: str, random_state: int = 42):
    """Factory: instancia o combiner indicado por `name`.

    Aceita `'stacking'` ou `'marginal_odds'` (case-insensitive),
    correspondendo aos dois caminhos descritos no Capítulo 4 da tese.
    """
    name = (name or "").lower()
    if name == "stacking":
        return StackingCombiner(random_state=random_state)
    if name == "marginal_odds":
        return MarginalOddsCombiner(random_state=random_state)
    raise ValueError(f"combiner_method '{name}' não reconhecido.")
