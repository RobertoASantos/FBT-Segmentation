"""Funções de combinação `f` descritas no Capítulo 4 da tese.

* `StackingCombiner` (Wolpert): a matriz Z tem o escore do segmento ao
  qual o registro pertence numa coluna e zero na outra. Uma regressão
  logística é ajustada sobre `Z` contra o alvo, produzindo o escore
  final.

* `MarginalOddsCombiner` (Thomas et al., 2002): ajusta uma logística
  por segmento, escolhe um segmento de **referência** e usa os
  coeficientes dessa logística para recalibrar o escore do outro
  segmento, alinhando ambos numa mesma escala de log-odds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


_EPS = 1e-9


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _safe_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))


@dataclass
class StackingCombiner:
    """Stacking via regressão logística sobre `[score_segmento, 0]`."""

    penalty: str | None = None
    C: float = 1e6
    max_iter: int = 2000
    random_state: int = 42
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
        membership = np.asarray(membership, dtype=int)
        y = np.asarray(y, dtype=int)
        n = membership.size

        Z = np.zeros((n, 2), dtype=float)
        Z[membership == 0, 0] = np.asarray(score_left, dtype=float)[membership == 0]
        Z[membership == 1, 1] = np.asarray(score_right, dtype=float)[membership == 1]

        if np.unique(y).size < 2:
            self.is_constant_ = True
            self.constant_p_ = float(y.mean()) if y.size else 0.5
            return self

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C if self.penalty else 1.0,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
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

    Para cada segmento, ajusta `LR(score → y)` em log-odds. Em predição:
    o segmento de referência usa seus próprios coeficientes; o outro
    segmento aplica os coeficientes da referência sobre seu escore,
    alinhando as duas escalas (procedimento descrito em Thomas, Edelman
    & Crook, 2002).
    """

    max_iter: int = 2000
    random_state: int = 42
    reference: str = "auto"  # 'left' | 'right' | 'auto' (maior n)
    intercept_left_: float = 0.0
    slope_left_: float = 1.0
    intercept_right_: float = 0.0
    slope_right_: float = 1.0
    is_constant_left_: bool = False
    is_constant_right_: bool = False
    constant_p_left_: float = 0.5
    constant_p_right_: float = 0.5
    reference_resolved_: str = "left"

    def _fit_segment(self, score, y):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=int)
        if y.size == 0 or np.unique(y).size < 2:
            return True, float(y.mean()) if y.size else 0.5, 0.0, 1.0
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
        membership = np.asarray(membership, dtype=int)
        y = np.asarray(y, dtype=int)
        score_left = np.asarray(score_left, dtype=float)
        score_right = np.asarray(score_right, dtype=float)

        left_mask = membership == 0
        right_mask = membership == 1

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
        membership = np.asarray(membership, dtype=int)
        n = membership.size
        p = np.full(n, 0.5, dtype=float)

        left_mask = membership == 0
        right_mask = membership == 1

        ref = self.reference_resolved_
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

        # segmento de referência: usa seus próprios coeficientes
        if ref == "left":
            if is_const_ref:
                p[left_mask] = const_ref
            else:
                p[left_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_left[left_mask])
                )
            # outro segmento: aplica coeficientes do segmento de referência
            if is_const_other:
                p[right_mask] = const_other
            else:
                # alinhamento: usa slope da referência sobre o escore do outro
                p[right_mask] = _safe_sigmoid(
                    a + b * _safe_logit(score_right[right_mask])
                )
        else:
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
    name = (name or "").lower()
    if name == "stacking":
        return StackingCombiner(random_state=random_state)
    if name == "marginal_odds":
        return MarginalOddsCombiner(random_state=random_state)
    raise ValueError(f"combiner_method '{name}' não reconhecido.")
