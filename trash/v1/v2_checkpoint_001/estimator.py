"""Estimador principal do RiskSeg V2.

Implementação fiel ao Capítulo 4 da tese de Roberto Angelo Fernandes
Santos, com correções dos desvios identificados na revisão da V1:

1. Modelos descendentes treinam **sem** a variável de split (efeito
   eliminado, como descrito na tese).
2. Marginal Odds usa segmento de **referência** real (`combiners.py`).
3. `grouping_features` é **lista por variável** (= `rUsaBlocos`);
   `max_group_size` (= `rQtdeBlocos`).
4. `min_gain_pct` / `max_loss_pct` em **percentual** do baseline.
5. `prediction_mode="leaf"` é o default (sem `global_stacking`).
6. Baseline `LogisticRegression(penalty=None)` por default (paper-like).
7. Predição totalmente vetorizada por folha.
8. Screening fatorial vetorizado em numpy.
9. Métrica `odds_ratio` disponível.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted

from .combiners import build_combiner
from .metrics import metric_score, all_metrics
from .tree import Node, collect_leaves, collect_nodes, route_observations, route_pair_parent
from .views import ModelView, SegView


_DEFAULT_MAX_ITER = 2000


def _default_base_estimator(random_state: int = 42) -> LogisticRegression:
    """Logística sem regularização — aproxima `PROC LOGISTIC` do SAS."""
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=_DEFAULT_MAX_ITER,
        random_state=random_state,
    )


def _fit_clone(estimator, X, y):
    """Treina uma cópia do estimador, com proteção a classe única."""
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if classes.size < 2:
        return _ConstantClassifier(float(y.mean()) if y.size else 0.5)
    m = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X, y)
    return m


@dataclass
class _ConstantClassifier:
    p_: float

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full((n, 2), [1 - self.p_, self.p_], dtype=float)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RiskSegV2(ClassifierMixin, BaseEstimator):
    """RiskSeg V2 — segmentação binária recursiva por modelo fatorial.

    Parâmetros principais (terminologia da tese entre parênteses):

    - `metric` (`D`): métrica usada para escolher splits.
    - `max_depth`, `min_samples_leaf` (`B`): critérios de parada.
    - `min_gain_pct`, `max_loss_pct`: ganho mínimo / perda máxima em
      percentual sobre o baseline do nó.
    - `top_k_variables` (`rQtdeVarTeste`): variáveis candidatas a teste
      de categoria após o screening fatorial.
    - `validation_fraction` (`rTamValidação`): fração do treino do nó
      usada como validação interna.
    - `n_numeric_bins` (`rQtdeDivisões`): nº de faixas para variáveis
      numéricas (segmentação).
    - `grouping_features` (`rUsaBlocos`): lista de variáveis que
      admitem agrupamento de categorias nos testes de quebra.
    - `max_group_size` (`rQtdeBlocos`): tamanho máximo do bloco.
    - `combiner_method` (`f`): `'stacking'` ou `'marginal_odds'`.
    - `drop_split_feature_in_children`: se True, modelos descendentes
      não enxergam a variável usada na divisão (recomendação da tese).
    - `prediction_mode`:
        - `'leaf'`: predição da folha alcançada (default).
        - `'pair_combiner'`: aplica o combiner do nó pai.
    - `base_estimator`: estimador sklearn para os modelos do nó/segmento.
      Default = `LogisticRegression(penalty=None)`.

    A V2 expõe ainda `factorial_max_interaction_features` para limitar
    o custo do screening em bases largas; defina `None` para a forma
    plena descrita na tese.
    """

    def __init__(
        self,
        base_estimator=None,
        screening_estimator=None,
        categorical_features=None,
        metric: str = "error",
        max_depth: int = 2,
        min_samples_leaf: float | int = 0.05,
        min_gain_pct: float = 0.0,
        max_loss_pct: float = 0.0,
        top_k_variables: int = 1,
        validation_fraction: float = 0.35,
        n_numeric_bins: int = 4,
        grouping_features: tuple | list | None = None,
        max_group_size: int = 2,
        ordered_groups_for_numeric: bool = True,
        combiner_method: str = "stacking",
        prediction_mode: str = "leaf",
        drop_split_feature_in_children: bool = True,
        scale_numeric: bool = True,
        factorial_max_interaction_features: int | None = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.base_estimator = base_estimator
        self.screening_estimator = screening_estimator
        self.categorical_features = categorical_features
        self.metric = metric
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_pct = min_gain_pct
        self.max_loss_pct = max_loss_pct
        self.top_k_variables = top_k_variables
        self.validation_fraction = validation_fraction
        self.n_numeric_bins = n_numeric_bins
        self.grouping_features = grouping_features
        self.max_group_size = max_group_size
        self.ordered_groups_for_numeric = ordered_groups_for_numeric
        self.combiner_method = combiner_method
        self.prediction_mode = prediction_mode
        self.drop_split_feature_in_children = drop_split_feature_in_children
        self.scale_numeric = scale_numeric
        self.factorial_max_interaction_features = factorial_max_interaction_features
        self.random_state = random_state
        self.verbose = verbose

    # =========================================================================
    # API sklearn
    # =========================================================================

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_input(X, y)
        self._rng = np.random.default_rng(self.random_state)

        cats = tuple(self.categorical_features or ())
        self.seg_view_ = SegView(
            n_numeric_bins=self.n_numeric_bins,
            categorical_features=cats,
        ).fit(X)
        self.model_view_ = ModelView(
            scale_numeric=self.scale_numeric,
            categorical_features=cats,
        ).fit(X)

        seg_codes = self.seg_view_.transform(X)
        model_mat = self.model_view_.transform(X)
        y_arr = y.to_numpy(dtype=int)

        self._base = self.base_estimator or _default_base_estimator(self.random_state)
        self._screening = self.screening_estimator or self._base

        self._node_counter = 0
        self._log(1, f"[FIT] n={len(y_arr)} features={X.shape[1]} model_cols={model_mat.shape[1]}")

        self.root_ = self._grow(
            idx=np.arange(len(y_arr), dtype=np.int64),
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y_arr,
            depth=0,
            used_features=tuple(),
        )

        self.leaves_ = collect_leaves(self.root_)
        self.nodes_ = collect_nodes(self.root_)
        self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        self._log(
            1,
            f"[FIT] tree built: nodes={len(self.nodes_)} leaves={len(self.leaves_)} "
            f"depth_reached={max(n.depth for n in self.nodes_)}",
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = self._as_dataframe(X)
        seg_codes = self.seg_view_.transform(X)
        model_mat = self.model_view_.transform(X)

        if self.prediction_mode == "leaf":
            return self._predict_proba_leaf(seg_codes, model_mat)
        if self.prediction_mode == "pair_combiner":
            return self._predict_proba_pair(seg_codes, model_mat)
        raise ValueError(f"prediction_mode '{self.prediction_mode}' não suportado.")

    def predict(self, X):
        proba = self.predict_proba(X)
        encoded = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[encoded]

    def score(self, X, y):
        y_enc = self._encode_y(y)
        return -np.mean((self.predict(X) != self.classes_[y_enc]).astype(float)) + 1.0

    # =========================================================================
    # Crescimento da árvore
    # =========================================================================

    def _grow(self, idx, seg_codes, model_mat, y, depth, used_features):
        node_id = self._next_node_id()
        used_features = tuple(used_features)

        cols_to_use = self._cols_for_features(used_features)
        node_model = _fit_clone(self._base, model_mat[idx][:, cols_to_use], y[idx])

        node = Node(
            node_id=node_id,
            depth=depth,
            is_leaf=True,
            node_model=node_model,
            node_columns=cols_to_use,
            n_train=int(idx.size),
            n_pos=int(np.sum(y[idx] == 1)),
            used_features=used_features,
        )

        min_leaf = self._resolve_min_leaf(idx.size)
        if (
            depth >= self.max_depth
            or idx.size < 2 * min_leaf
            or np.unique(y[idx]).size < 2
        ):
            self._log(2, f"[NODE {node_id}] folha por parada (depth={depth}, n={idx.size})")
            return node

        tr_rel, va_rel = self._internal_train_val_split(y[idx])
        tr_abs = idx[tr_rel]
        va_abs = idx[va_rel]

        baseline_model = _fit_clone(self._base, model_mat[tr_abs][:, cols_to_use], y[tr_abs])
        baseline_proba = baseline_model.predict_proba(model_mat[va_abs][:, cols_to_use])[:, 1]
        baseline_obj = metric_score(self.metric, y[va_abs], baseline_proba)
        node.baseline_obj = baseline_obj

        screening = self._screen(
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y,
            tr_abs=tr_abs,
            va_abs=va_abs,
            cols_to_use=cols_to_use,
            used_features=used_features,
        )
        if not screening:
            self._log(2, f"[NODE {node_id}] screening vazio → folha")
            return node

        top_vars = [v for v, _ in screening[: max(1, self.top_k_variables)]]
        self._log(
            3,
            f"[NODE {node_id}] screening top-{len(top_vars)}: "
            + ", ".join(f"{v}={obj:.4f}" for v, obj in screening[:5]),
        )

        best = None
        for var_name in top_vars:
            best = self._best_split_for_variable(
                var_name=var_name,
                seg_codes=seg_codes,
                model_mat=model_mat,
                y=y,
                tr_abs=tr_abs,
                va_abs=va_abs,
                idx=idx,
                cols_to_use=cols_to_use,
                used_features=used_features,
                current_best=best,
            )

        if best is None:
            self._log(2, f"[NODE {node_id}] sem split candidato válido → folha")
            return node

        denom = abs(baseline_obj) if baseline_obj != 0 else 1.0
        gain_pct = (best["obj"] - baseline_obj) / denom
        node.gain_pct = gain_pct
        node.split_obj = best["obj"]

        if gain_pct >= 0:
            accept = gain_pct >= self.min_gain_pct
        else:
            accept = (-gain_pct) <= self.max_loss_pct
        if not accept:
            self._log(
                2,
                f"[NODE {node_id}] split rejeitado: gain_pct={gain_pct:.4f} "
                f"(min={self.min_gain_pct}, max_loss={self.max_loss_pct})",
            )
            return node

        node.is_leaf = False
        node.split_variable = best["var_name"]
        node.split_col = best["var_col"]
        node.split_codes = best["codes"]
        node.split_group_text = best["group_text"]
        node.left_model = best["left_model"]
        node.right_model = best["right_model"]
        node.combiner = best["combiner"]
        node.segment_columns = best["seg_columns"]

        self._log(
            2,
            f"[NODE {node_id}] ACEITO | var={best['var_name']} grupo={best['group_text']} "
            f"gain_pct={gain_pct:.4f} obj={best['obj']:.4f} baseline={baseline_obj:.4f}",
        )

        new_used = tuple(list(used_features) + [best["var_name"]]) if self.drop_split_feature_in_children else used_features
        left_idx = idx[best["left_mask_idx"]]
        right_idx = idx[best["right_mask_idx"]]

        node.left = self._grow(
            idx=left_idx,
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y,
            depth=depth + 1,
            used_features=new_used,
        )
        node.right = self._grow(
            idx=right_idx,
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y,
            depth=depth + 1,
            used_features=new_used,
        )
        return node

    # =========================================================================
    # Screening fatorial
    # =========================================================================

    def _screen(self, seg_codes, model_mat, y, tr_abs, va_abs, cols_to_use, used_features):
        results = []
        candidate_vars = [c for c in self.seg_view_.columns_ if c not in used_features]

        main_train = model_mat[tr_abs][:, cols_to_use]
        main_val = model_mat[va_abs][:, cols_to_use]

        max_inter = self.factorial_max_interaction_features
        if max_inter is None or max_inter >= main_train.shape[1]:
            inter_cols = np.arange(main_train.shape[1])
        else:
            variances = main_train.var(axis=0)
            inter_cols = np.argsort(-variances)[:max_inter]
        sub_train = main_train[:, inter_cols] if inter_cols.size else main_train[:, :0]
        sub_val = main_val[:, inter_cols] if inter_cols.size else main_val[:, :0]

        for var_name in candidate_vars:
            var_col = self.seg_view_.column_index(var_name)
            codes_tr = seg_codes[tr_abs, var_col]
            codes_va = seg_codes[va_abs, var_col]
            present = np.unique(codes_tr[codes_tr >= 0])
            if present.size < 2:
                continue

            dummies_tr = (codes_tr[:, None] == present[None, :]).astype(np.float64)
            dummies_va = (codes_va[:, None] == present[None, :]).astype(np.float64)

            if sub_train.size and dummies_tr.shape[1] > 0:
                inter_tr = (dummies_tr[:, :, None] * sub_train[:, None, :]).reshape(
                    dummies_tr.shape[0], -1
                )
                inter_va = (dummies_va[:, :, None] * sub_val[:, None, :]).reshape(
                    dummies_va.shape[0], -1
                )
            else:
                inter_tr = np.zeros((dummies_tr.shape[0], 0))
                inter_va = np.zeros((dummies_va.shape[0], 0))

            Z_tr = np.concatenate([dummies_tr, main_train, inter_tr], axis=1)
            Z_va = np.concatenate([dummies_va, main_val, inter_va], axis=1)

            model = _fit_clone(self._screening, Z_tr, y[tr_abs])
            p_va = model.predict_proba(Z_va)[:, 1]
            obj = metric_score(self.metric, y[va_abs], p_va)
            results.append((var_name, obj))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # =========================================================================
    # Avaliação de splits para uma variável
    # =========================================================================

    def _best_split_for_variable(
        self,
        var_name,
        seg_codes,
        model_mat,
        y,
        tr_abs,
        va_abs,
        idx,
        cols_to_use,
        used_features,
        current_best,
    ):
        var_col = self.seg_view_.column_index(var_name)
        node_codes = seg_codes[idx, var_col]
        codes_present = np.unique(node_codes[node_codes >= 0])
        if codes_present.size < 2:
            return current_best

        is_numeric = self.seg_view_.is_numeric_column(var_name)
        groups = self._enumerate_groups(var_name, codes_present, is_numeric)
        if not groups:
            return current_best

        seg_cols = self.model_view_.columns_excluding(
            list(used_features) + ([var_name] if self.drop_split_feature_in_children else [])
        )
        if seg_cols.size == 0:
            seg_cols = self.model_view_.all_columns()

        codes_tr = seg_codes[tr_abs, var_col]
        codes_va = seg_codes[va_abs, var_col]
        min_leaf = self._resolve_min_leaf(idx.size)

        best = current_best

        for group_codes in groups:
            left_mask_tr = np.isin(codes_tr, group_codes)
            right_mask_tr = ~left_mask_tr
            left_mask_va = np.isin(codes_va, group_codes)
            right_mask_va = ~left_mask_va

            if (
                left_mask_tr.sum() < min_leaf
                or right_mask_tr.sum() < min_leaf
                or left_mask_va.sum() < 1
                or right_mask_va.sum() < 1
            ):
                continue

            y_tr_left = y[tr_abs][left_mask_tr]
            y_tr_right = y[tr_abs][right_mask_tr]
            if np.unique(y_tr_left).size < 2 or np.unique(y_tr_right).size < 2:
                continue

            X_tr_left = model_mat[tr_abs[left_mask_tr]][:, seg_cols]
            X_tr_right = model_mat[tr_abs[right_mask_tr]][:, seg_cols]
            X_va_left = model_mat[va_abs[left_mask_va]][:, seg_cols]
            X_va_right = model_mat[va_abs[right_mask_va]][:, seg_cols]

            left_model = _fit_clone(self._base, X_tr_left, y_tr_left)
            right_model = _fit_clone(self._base, X_tr_right, y_tr_right)

            score_tr_left = np.zeros(tr_abs.size, dtype=float)
            score_tr_right = np.zeros(tr_abs.size, dtype=float)
            score_tr_left[left_mask_tr] = left_model.predict_proba(X_tr_left)[:, 1]
            score_tr_right[right_mask_tr] = right_model.predict_proba(X_tr_right)[:, 1]
            membership_tr = right_mask_tr.astype(int)

            combiner = build_combiner(self.combiner_method, random_state=self.random_state)
            combiner.fit(
                score_left=score_tr_left,
                score_right=score_tr_right,
                membership=membership_tr,
                y=y[tr_abs],
            )

            score_va_left = np.zeros(va_abs.size, dtype=float)
            score_va_right = np.zeros(va_abs.size, dtype=float)
            score_va_left[left_mask_va] = left_model.predict_proba(X_va_left)[:, 1]
            score_va_right[right_mask_va] = right_model.predict_proba(X_va_right)[:, 1]
            membership_va = right_mask_va.astype(int)

            p_va = combiner.predict_proba(
                score_left=score_va_left,
                score_right=score_va_right,
                membership=membership_va,
            )[:, 1]
            obj = metric_score(self.metric, y[va_abs], p_va)

            node_codes_full = seg_codes[idx, var_col]
            left_idx_rel = np.where(np.isin(node_codes_full, group_codes))[0]
            right_idx_rel = np.where(~np.isin(node_codes_full, group_codes))[0]

            candidate = {
                "var_name": var_name,
                "var_col": var_col,
                "codes": np.asarray(group_codes, dtype=np.int32),
                "group_text": self._format_group(var_name, group_codes),
                "obj": obj,
                "left_model": left_model,
                "right_model": right_model,
                "combiner": combiner,
                "seg_columns": seg_cols,
                "left_mask_idx": left_idx_rel,
                "right_mask_idx": right_idx_rel,
            }
            if best is None or candidate["obj"] > best["obj"]:
                best = candidate

        return best

    # =========================================================================
    # Predição vetorizada
    # =========================================================================

    def _predict_proba_leaf(self, seg_codes, model_mat):
        n = seg_codes.shape[0]
        leaf_ids = route_observations(self.root_, seg_codes)
        out = np.full((n, 2), [0.5, 0.5], dtype=float)
        leaf_by_id = {leaf.node_id: leaf for leaf in self.leaves_}
        for leaf_id in np.unique(leaf_ids):
            if leaf_id < 0:
                continue
            mask = leaf_ids == leaf_id
            leaf = leaf_by_id[int(leaf_id)]
            X = model_mat[mask][:, leaf.node_columns]
            out[mask] = leaf.node_model.predict_proba(X)
        return out

    def _predict_proba_pair(self, seg_codes, model_mat):
        n = seg_codes.shape[0]
        parent_ids, sides = route_pair_parent(self.root_, seg_codes)
        out = np.full((n, 2), [0.5, 0.5], dtype=float)
        node_by_id = {node.node_id: node for node in self.nodes_}
        for parent_id in np.unique(parent_ids):
            if parent_id < 0:
                # observação chega numa raiz que é folha
                mask = parent_ids < 0
                X = model_mat[mask][:, self.root_.node_columns]
                out[mask] = self.root_.node_model.predict_proba(X)
                continue
            parent = node_by_id[int(parent_id)]
            seg_cols = parent.segment_columns
            mask = parent_ids == parent_id
            local_sides = sides[mask]
            X = model_mat[mask][:, seg_cols]
            score_left = np.zeros(X.shape[0], dtype=float)
            score_right = np.zeros(X.shape[0], dtype=float)
            left_local = local_sides == 0
            right_local = local_sides == 1
            if left_local.any():
                score_left[left_local] = parent.left_model.predict_proba(X[left_local])[:, 1]
            if right_local.any():
                score_right[right_local] = parent.right_model.predict_proba(X[right_local])[:, 1]
            p = parent.combiner.predict_proba(
                score_left=score_left,
                score_right=score_right,
                membership=local_sides.astype(int),
            )[:, 1]
            out[mask] = np.column_stack([1 - p, p])
        return out

    # =========================================================================
    # Utilidades
    # =========================================================================

    def _validate_input(self, X, y):
        X_df = self._as_dataframe(X)
        if y is None:
            raise ValueError("RiskSegV2.fit exige y.")
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        if y_series.size != X_df.shape[0]:
            raise ValueError("X e y com tamanhos inconsistentes.")
        if y_series.isna().any():
            raise ValueError("y contém NaN.")

        classes = np.unique(y_series.to_numpy())
        if classes.size != 2:
            raise ValueError(
                f"RiskSegV2 é binário; recebeu {classes.size} classes: {classes!r}"
            )
        self.classes_ = classes
        y_enc = (y_series.to_numpy() == classes[1]).astype(int)
        return X_df, pd.Series(y_enc, index=X_df.index, name=y_series.name)

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X = np.asarray(X)
        cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def _encode_y(self, y):
        y = np.asarray(y)
        return (y == self.classes_[1]).astype(int)

    def _next_node_id(self):
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _resolve_min_leaf(self, n):
        if isinstance(self.min_samples_leaf, float) and self.min_samples_leaf < 1:
            return max(2, int(np.ceil(n * self.min_samples_leaf)))
        return int(self.min_samples_leaf)

    def _internal_train_val_split(self, y):
        rng = self._rng
        n = y.size
        n_val = max(2, int(np.round(n * self.validation_fraction)))
        n_val = min(n - 2, n_val)
        if n_val < 1 or np.unique(y).size < 2:
            order = rng.permutation(n)
            return order[: max(1, n - n_val)], order[max(1, n - n_val) :]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_val,
            random_state=self.random_state,
        )
        tr, va = next(sss.split(np.zeros(n), y))
        return tr, va

    def _cols_for_features(self, used_features):
        if self.drop_split_feature_in_children and used_features:
            cols = self.model_view_.columns_excluding(used_features)
            return cols if cols.size > 0 else self.model_view_.all_columns()
        return self.model_view_.all_columns()

    def _enumerate_groups(self, var_name, codes_present, is_numeric):
        codes = list(map(int, codes_present))
        n = len(codes)
        if n < 2:
            return []

        groupings = self.grouping_features
        uses_grouping = groupings is None or var_name in (groupings or [])
        max_block = max(1, self.max_group_size) if uses_grouping else 1

        groups: list[tuple[int, ...]] = []
        if is_numeric and self.ordered_groups_for_numeric:
            for start in range(n):
                for end in range(start + 1, n + 1):
                    if end - start >= n:
                        continue
                    if end - start > max_block:
                        continue
                    groups.append(tuple(codes[start:end]))
        else:
            for k in range(1, max_block + 1):
                if k >= n:
                    break
                for combo in itertools.combinations(codes, k):
                    groups.append(tuple(sorted(combo)))

        # remove complementos duplicados
        seen = set()
        dedup = []
        for g in groups:
            comp = tuple(sorted(set(codes) - set(g)))
            key = frozenset((frozenset(g), frozenset(comp)))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(g)
        return dedup

    def _format_group(self, var_name, group_codes):
        if self.seg_view_.is_numeric_column(var_name):
            return f"bins[{','.join(str(c) for c in group_codes)}]"
        idx = self.seg_view_.category_index_[var_name]
        inv = {v: k for k, v in idx.items()}
        labels = [inv.get(int(c), str(c)) for c in group_codes]
        return "{" + ", ".join(labels) + "}"

    def _log(self, level, msg):
        if self.verbose >= level:
            print(msg)

    # =========================================================================
    # Diagnóstico
    # =========================================================================

    def evaluate(self, X, y):
        y_enc = self._encode_y(y)
        p = self.predict_proba(X)[:, 1]
        return all_metrics(y_enc, p)

    def get_tree_summary(self) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        rows = []
        for node in self.nodes_:
            rows.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "is_leaf": node.is_leaf,
                    "n_train": node.n_train,
                    "n_pos": node.n_pos,
                    "split_variable": node.split_variable,
                    "split_group": node.split_group_text,
                    "baseline_obj": node.baseline_obj,
                    "split_obj": node.split_obj,
                    "gain_pct": node.gain_pct,
                }
            )
        return pd.DataFrame(rows)


# =============================================================================
# Presets
# =============================================================================


def _thesis_defaults() -> dict:
    return {
        "metric": "error",
        "max_depth": 3,
        "min_samples_leaf": 0.05,
        "validation_fraction": 0.25,
        "top_k_variables": 1,
        "n_numeric_bins": 4,
        "max_group_size": 2,
        "combiner_method": "stacking",
        "prediction_mode": "leaf",
        "drop_split_feature_in_children": True,
        "scale_numeric": True,
        "min_gain_pct": 0.0,
        "max_loss_pct": 0.02,
        "factorial_max_interaction_features": None,
        "verbose": 0,
    }


def _article_uci_defaults() -> dict:
    return {
        "metric": "error",
        "max_depth": 2,
        "min_samples_leaf": 0.05,
        "validation_fraction": 0.35,
        "top_k_variables": 1,
        "n_numeric_bins": 4,
        "max_group_size": 2,
        "combiner_method": "stacking",
        "prediction_mode": "leaf",
        "drop_split_feature_in_children": True,
        "scale_numeric": True,
        "min_gain_pct": 0.0,
        "max_loss_pct": 0.0,
        "factorial_max_interaction_features": 16,
        "verbose": 0,
    }


def _article_synthetic_defaults() -> dict:
    return {
        "metric": "error",
        "max_depth": 3,
        "min_samples_leaf": 0.10,
        "validation_fraction": 0.25,
        "top_k_variables": 1,
        "n_numeric_bins": 4,
        "max_group_size": 2,
        "combiner_method": "stacking",
        "prediction_mode": "leaf",
        "drop_split_feature_in_children": True,
        "scale_numeric": True,
        "min_gain_pct": 0.0,
        "max_loss_pct": 0.0,
        "factorial_max_interaction_features": None,
        "verbose": 0,
    }


def thesis_preset(**overrides) -> RiskSegV2:
    """Preset alinhado ao Capítulo 4 da tese (default mais fiel)."""
    params = {**_thesis_defaults(), **overrides}
    return RiskSegV2(**params)


def article_uci_preset(**overrides) -> RiskSegV2:
    """Preset alinhado ao artigo ICAI 2012 (UCI: Chess, German, Magic, Adult, Spambase)."""
    params = {**_article_uci_defaults(), **overrides}
    return RiskSegV2(**params)


def article_synthetic_preset(**overrides) -> RiskSegV2:
    """Preset alinhado ao artigo ICTAI 2012 (datasets sintéticos)."""
    params = {**_article_synthetic_defaults(), **overrides}
    return RiskSegV2(**params)
