import math
import itertools
import numbers
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse as scipy_sparse

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def _check_no_complex_data(X):
    if isinstance(X, pd.DataFrame):
        for dtype in X.dtypes:
            if np.issubdtype(dtype, np.complexfloating):
                raise ValueError("Complex data not supported")
        values = X.to_numpy()
    else:
        values = np.asarray(X)
    if np.iscomplexobj(values):
        raise ValueError("Complex data not supported")


def _check_not_sparse(X):
    if scipy_sparse.issparse(X):
        raise TypeError("Sparse input is not supported. Convert X to a dense array before fitting RiskSegOptimizer.")


def _check_no_nan_inf(X):
    if isinstance(X, pd.DataFrame):
        if X.isna().to_numpy().any():
            raise ValueError("Input X contains NaN.")
        numeric = X.select_dtypes(include=[np.number])
        if len(numeric.columns) and np.isinf(numeric.to_numpy()).any():
            raise ValueError("Input X contains inf.")
        return

    values = np.asarray(X)
    if np.issubdtype(values.dtype, np.number):
        if np.isnan(values).any():
            raise ValueError("Input X contains NaN.")
        if np.isinf(values).any():
            raise ValueError("Input X contains inf.")
        return

    if values.dtype == object:
        for value in values.ravel():
            try:
                if bool(pd.isna(value)):
                    raise ValueError("Input X contains NaN.")
            except (TypeError, ValueError) as exc:
                if str(exc) == "Input X contains NaN.":
                    raise
            if isinstance(value, numbers.Number) and np.isinf(value):
                raise ValueError("Input X contains inf.")


def _check_supported_object_data(X):
    if isinstance(X, pd.DataFrame):
        if not any(dtype == object for dtype in X.dtypes):
            return
        values = X.to_numpy(dtype=object)
    else:
        values = np.asarray(X)
        if values.dtype != object:
            return

    for value in values.ravel():
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass

        if isinstance(value, (str, bytes, numbers.Number, np.bool_)):
            continue

        try:
            hash(value)
        except TypeError as exc:
            raise TypeError("argument must be a string or number") from exc


def _as_dataframe(X, feature_names=None):
    _check_not_sparse(X)
    _check_no_complex_data(X)
    _check_no_nan_inf(X)
    _check_supported_object_data(X)
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            "Expected 2D array, got 1D array instead. Reshape your data either "
            "using array.reshape(-1, 1) if your data has a single feature or "
            "array.reshape(1, -1) if it contains a single sample."
        )
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    elif X.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X.shape[1]} features, but RiskSegOptimizer is expecting "
            f"{len(feature_names)} features as input."
        )
    return pd.DataFrame(X, columns=list(feature_names))


def _as_series(y, name="target"):
    if isinstance(y, pd.Series):
        s = y.copy()
    else:
        y = np.asarray(y)
        if y.ndim != 1:
            if y.ndim == 2 and y.shape[1] == 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was expected.",
                    DataConversionWarning,
                    stacklevel=2,
                )
            y = y.ravel()
        s = pd.Series(y, name=name)
    return s


def _safe_logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _safe_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _group_text(group):
    vals = sorted(list(group))
    return "{" + ", ".join(map(str, vals)) + "}"


def _dedupe_complementary_groups(groups, universe):
    seen = set()
    kept = []
    universe = frozenset(universe)

    for g in groups:
        comp = universe - g
        key = tuple(sorted(g)) if tuple(sorted(g)) <= tuple(sorted(comp)) else tuple(sorted(comp))
        if key not in seen:
            seen.add(key)
            kept.append(g)
    return kept


def _powerset(values, max_size=2):
    vals = list(values)
    out = []
    for r in range(1, min(max_size, len(vals)) + 1):
        for comb in itertools.combinations(vals, r):
            s = frozenset(comb)
            if 0 < len(s) < len(vals):
                out.append(s)
    return out


# ==============================================================================
# MODELOS AUXILIARES
# ==============================================================================

class ConstantModel:
    def __init__(self, prob_1):
        self.prob_1 = float(np.clip(prob_1, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.prob_1, dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        return np.full(len(X), _safe_logit(self.prob_1), dtype=float)


class LogisticStackingCombiner:
    def __init__(self, random_state=42, C=1.0, max_iter=5000):
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.had_convergence_warning_ = False

    def fit(self, Z, y, sample_weight=None):
        self.model_ = LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            if sample_weight is not None:
                self.model_.fit(Z, y, sample_weight=sample_weight)
            else:
                self.model_.fit(Z, y)

            self.had_convergence_warning_ = any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

        return self

    def predict_proba(self, Z):
        return self.model_.predict_proba(Z)


class MarginalOddsCombiner:
    def __init__(self, random_state=42, max_iter=5000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.had_convergence_warning_ = False

    def _fit_one(self, score, y, sample_weight=None):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y).astype(int)

        if len(score) == 0:
            return ConstantModel(float(np.mean(y)) if len(y) > 0 else 0.5)
        if len(np.unique(y)) < 2:
            return ConstantModel(float(np.mean(y)))

        X = _ensure_2d(_safe_logit(score))
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)

            self.had_convergence_warning_ = any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

        return model

    def fit(self, Z, y, membership, sample_weight=None):
        membership = np.asarray(membership).astype(int)
        y = np.asarray(y).astype(int)

        left_idx = membership == 0
        right_idx = membership == 1

        sw_left = sample_weight[left_idx] if sample_weight is not None else None
        sw_right = sample_weight[right_idx] if sample_weight is not None else None

        self.left_model_ = self._fit_one(Z[left_idx, 0], y[left_idx], sw_left)
        self.right_model_ = self._fit_one(Z[right_idx, 1], y[right_idx], sw_right)
        return self

    



    def predict_proba(self, Z, membership=None):
        if membership is None:
            raise ValueError("MarginalOddsCombiner exige membership.")

        membership = np.asarray(membership).astype(int)
        out = np.zeros((len(Z), 2), dtype=float)

        left_idx = membership == 0
        right_idx = membership == 1

        if np.any(left_idx):
            Xl = _ensure_2d(_safe_logit(Z[left_idx, 0]))
            if hasattr(self.left_model_, "predict_proba"):
                pl = self.left_model_.predict_proba(Xl)[:, 1]
            else:
                pl = np.full(np.sum(left_idx), self.left_model_.prob_1, dtype=float)
            out[left_idx, 1] = pl
            out[left_idx, 0] = 1 - pl

        if np.any(right_idx):
            Xr = _ensure_2d(_safe_logit(Z[right_idx, 1]))
            if hasattr(self.right_model_, "predict_proba"):
                pr = self.right_model_.predict_proba(Xr)[:, 1]
            else:
                pr = np.full(np.sum(right_idx), self.right_model_.prob_1, dtype=float)
            out[right_idx, 1] = pr
            out[right_idx, 0] = 1 - pr

        return out


# ==============================================================================
# NÓ
# ==============================================================================

@dataclass
class RiskSegNode:
    node_id: int
    depth: int
    train_index: np.ndarray

    is_leaf: bool = True
    split_variable: str = None
    split_group: frozenset = None
    split_rule_text: str = None

    left_child: "RiskSegNode" = None
    right_child: "RiskSegNode" = None

    node_model: object = None
    left_model: object = None
    right_model: object = None
    combiner: object = None

    global_metrics_train: dict = field(default_factory=dict)
    global_metrics_val: dict = field(default_factory=dict)
    split_metrics_val: dict = field(default_factory=dict)
    split_gain: float = None

    left_support: dict = field(default_factory=dict)
    right_support: dict = field(default_factory=dict)

    screening_table: pd.DataFrame = None
    split_candidates_table: pd.DataFrame = None


# ==============================================================================
# RISKSEG
# ==============================================================================

class RiskSegOptimizer(ClassifierMixin, BaseEstimator):
    """
    Implementação do RISKSEG aderente ao Capítulo 4:
    - screening por variável com modelo fatorial/interativo
    - segmentação binária por categoria/grupo de categorias
    - combiner por nó (stacking ou marginal odds)
    - modelo global final como stacking das folhas

    Ajuste importante nesta versão:
    - o combiner local passa a ser treinado no TREINO do nó e validado na VALIDAÇÃO,
      evitando usar a validação para ajustar o próprio combiner.
    - foi acrescentado verbose detalhado em níveis.
    - foi acrescentada avaliação out-of-sample sem remover o summary in-sample.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = False
        return tags

    def __init__(
        self,
        base_estimator=None,                  # compatibilidade retroativa
        screening_estimator=None,
        screening_variables=None,
        categorical_features=None,
        node_estimator=None,
        segment_estimator=None,
        local_combiner_estimator=None,
        global_combiner_estimator=None,
        screening_mode="factorial_target_interactions",
        combiner_method="stacking",
        prediction_mode="global_stacking",
        metric="combined",
        top_rate=0.05,
        metric_weights=(0.7, 0.3),
        classification_threshold=0.5,
        max_depth=3,
        min_samples_leaf=0.10,
        min_gain=0.0,
        validation_fraction=0.25,
        validation_mode="random_holdout",
        recent_zone_size=None,
        n_recent_windows=4,
        recent_window_val_ratio=0.5,
        gap_size=0,
        random_state=42,
        stratify=True,
        auto_bin_numeric=True,
        n_numeric_bins=4,
        numeric_binning="quantile",
        group_binned_numeric=False,
        scale_model_numeric=False,
        model_numeric_scaling="minmax",
        factorial_max_interaction_features=None,
        factorial_feature_selector="variance",
        factorial_include_main_effects=True,
        factorial_drop_first=False,
        use_grouping=True,
        max_group_size=2,
        top_k_variables=3,
        use_top_k_variables=True,
        use_validation_to_accept_split=True,
        global_only_if_no_gain=True,
        global_stacking_C=1.0,
        verbose=4
    ):
        self.base_estimator = base_estimator   # compatibilidade retroativa
        self.screening_estimator = screening_estimator
        self.screening_variables = screening_variables
        self.categorical_features = categorical_features
        self.node_estimator = node_estimator
        self.segment_estimator = segment_estimator
        self.local_combiner_estimator = local_combiner_estimator
        self.global_combiner_estimator = global_combiner_estimator
        
        self.screening_mode = screening_mode
        self.combiner_method = combiner_method
        self.prediction_mode = prediction_mode
        self.metric = metric
        self.top_rate = top_rate
        self.metric_weights = metric_weights
        self.classification_threshold = classification_threshold
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.validation_fraction = validation_fraction
        self.validation_mode = validation_mode
        self.recent_zone_size = recent_zone_size
        self.n_recent_windows = n_recent_windows
        self.recent_window_val_ratio = recent_window_val_ratio
        self.gap_size = gap_size
        self.random_state = random_state
        self.stratify = stratify
        self.auto_bin_numeric = auto_bin_numeric
        self.n_numeric_bins = n_numeric_bins
        self.numeric_binning = numeric_binning
        self.group_binned_numeric = group_binned_numeric
        self.scale_model_numeric = scale_model_numeric
        self.model_numeric_scaling = model_numeric_scaling
        self.factorial_max_interaction_features = factorial_max_interaction_features
        self.factorial_feature_selector = factorial_feature_selector
        self.factorial_include_main_effects = factorial_include_main_effects
        self.factorial_drop_first = factorial_drop_first
        self.use_grouping = use_grouping
        self.max_group_size = max_group_size
        self.top_k_variables = top_k_variables
        self.use_top_k_variables = use_top_k_variables
        self.use_validation_to_accept_split = use_validation_to_accept_split
        self.global_only_if_no_gain = global_only_if_no_gain
        self.global_stacking_C = global_stacking_C
        self.verbose = verbose

    @classmethod
    def paper_params(cls, **overrides):
        params = {
            "screening_mode": "factorial_target_interactions",
            "combiner_method": "stacking",
            "prediction_mode": "global_stacking",
            "metric": "error",
            "max_depth": 2,
            "min_samples_leaf": 0.05,
            "validation_fraction": 0.35,
            "auto_bin_numeric": True,
            "n_numeric_bins": 4,
            "numeric_binning": "quantile",
            "group_binned_numeric": True,
            "scale_model_numeric": True,
            "model_numeric_scaling": "minmax",
            "use_grouping": False,
            "top_k_variables": 1,
            "use_top_k_variables": True,
            "factorial_max_interaction_features": 8,
            "use_validation_to_accept_split": True,
            "global_only_if_no_gain": True,
            "verbose": 0,
        }
        params.update(overrides)
        return params

    @classmethod
    def paper_preset(cls, **overrides):
        return cls(**cls.paper_params(**overrides))

    def _evaluate_dataset_internal(self, X, y, dataset_name="internal"):
        """
        Versão interna de avaliação para uso durante o fit,
        sem exigir check_is_fitted.
        """
        y_series = _as_series(y)
        p = self._predict_proba_internal(X)[:, 1]
        metrics = self._calculate_metrics(y_series, p)

        result = {
            "dataset_name": dataset_name,
            "n_observations": int(len(y_series)),
            "objective": float(self._objective_from_metrics(metrics)),
            "auc": float(metrics["auc"]),
            "ks": float(metrics["ks"]),
            "lift": float(metrics["lift"]),
            "precision": float(metrics["precision"]),
            "error": float(metrics["error"]),
            "rocmin": float(metrics["rocmin"]),
            "prediction_mode": self.prediction_mode,
            "combiner_method": self.combiner_method,
            "screening_mode": self.screening_mode,
            "factorial_max_interaction_features": self.factorial_max_interaction_features,
            "factorial_feature_selector": self.factorial_feature_selector,
            "factorial_include_main_effects": self.factorial_include_main_effects,
            "factorial_drop_first": self.factorial_drop_first,
            "max_depth_reached": int(self.node_table_["depth"].max()) if hasattr(self, "node_table_") and len(self.node_table_) > 0 else None,
            "n_nodes": int(len(self.node_table_)) if hasattr(self, "node_table_") else None,
            "n_leaves": int((self.node_table_["is_leaf"]).sum()) if hasattr(self, "node_table_") and len(self.node_table_) > 0 else None,
            "used_variables_unique": sorted(set(self.node_table_.loc[~self.node_table_["is_leaf"], "split_variable"].dropna().tolist())) if hasattr(self, "node_table_") and len(self.node_table_) > 0 else [],
            "used_variables_sequence": self.node_table_.loc[~self.node_table_["is_leaf"], "split_variable"].dropna().tolist() if hasattr(self, "node_table_") and len(self.node_table_) > 0 else [],
            "screening_estimator_name": self.estimator_metadata_["screening_estimator_name"] if hasattr(self, "estimator_metadata_") else None,
            "node_estimator_name": self.estimator_metadata_["node_estimator_name"] if hasattr(self, "estimator_metadata_") else None,
            "segment_estimator_name": self.estimator_metadata_["segment_estimator_name"] if hasattr(self, "estimator_metadata_") else None,
            "local_combiner_estimator_name": self.estimator_metadata_["local_combiner_estimator_name"] if hasattr(self, "estimator_metadata_") else None,
            "global_combiner_estimator_name": self.estimator_metadata_["global_combiner_estimator_name"] if hasattr(self, "estimator_metadata_") else None
        }

        return result

    def _next_node_id(self):
        self._node_counter_ += 1
        return self._node_counter_

    def _fit_estimator_with_warning_control(self, estimator, X, y, sample_weight=None, context=""):
        """
        Ajusta um estimador controlando ConvergenceWarning para não poluir a saída.
        Retorna o estimador ajustado e um boolean informando se houve warning.
        """
        had_convergence_warning = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            if sample_weight is not None:
                estimator.fit(X, y, sample_weight=sample_weight)
            else:
                estimator.fit(X, y)

            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    had_convergence_warning = True

        if had_convergence_warning:
            self._log(3, f"[WARNING] ConvergenceWarning durante ajuste: {context}", indent=2)

        return estimator, had_convergence_warning

    # --------------------------------------------------------------------------
    # LOGGING HELPERS
    # --------------------------------------------------------------------------

    def _get_allowed_screening_variables(self, X_seg_columns):
        all_cols = list(X_seg_columns)

        if self.screening_variables is None:
            return all_cols

        allowed = [c for c in self.screening_variables if c in all_cols]
        missing = [c for c in self.screening_variables if c not in all_cols]

        if len(missing) > 0:
            self._log(1, f"[WARNING] screening_variables ausentes: {missing}")

        if len(allowed) == 0:
            raise ValueError("Nenhuma variável válida em screening_variables.")

        return allowed

    def _log(self, level, msg, indent=0):
        if self.verbose >= level:
            print(f"{'    ' * indent}{msg}")

    def _fmt_metrics(self, metrics):
        if metrics is None or len(metrics) == 0:
            return "metrics=NA"
        parts = []
        try:
            parts.append(f"obj={self._objective_from_metrics(metrics):.6f}")
        except Exception:
            pass
        for k in ["auc", "ks", "lift", "precision", "error", "rocmin"]:
            if k in metrics and metrics[k] is not None:
                try:
                    parts.append(f"{k}={float(metrics[k]):.6f}")
                except Exception:
                    parts.append(f"{k}={metrics[k]}")
        return " | ".join(parts)

    def _build_recent_temporal_split_indices(self, n_obs):
        if n_obs < 4:
            raise ValueError("Número insuficiente de observações para validação temporal recente.")

        requested_recent_zone = self.recent_zone_size
        if requested_recent_zone is None:
            requested_recent_zone = max(int(round(n_obs * self.validation_fraction)), 2)

        recent_total = min(int(requested_recent_zone), n_obs - 2)
        if recent_total < 2:
            raise ValueError("Faixa recente insuficiente para montar treino/validação.")

        recent_start = n_obs - recent_total
        base_train_idx = np.arange(recent_start, dtype=int)
        recent_idx = np.arange(recent_start, n_obs, dtype=int)

        n_windows = max(int(self.n_recent_windows), 1)
        blocks = [np.asarray(b, dtype=int) for b in np.array_split(recent_idx, n_windows) if len(b) > 0]

        train_parts = [base_train_idx]
        val_parts = []

        for block in blocks:
            block_len = len(block)
            if block_len < 2:
                continue

            gap = max(int(self.gap_size), 0)
            gap = min(gap, max(block_len - 2, 0))

            raw_val_size = int(round(block_len * float(self.recent_window_val_ratio)))
            val_size = max(1, raw_val_size)
            max_val_allowed = block_len - gap - 1
            val_size = min(val_size, max_val_allowed)

            train_local_size = block_len - gap - val_size
            if train_local_size <= 0:
                raise ValueError("Configuração inválida: treino local vazio nas janelas temporais recentes.")

            train_parts.append(block[:train_local_size])
            val_parts.append(block[train_local_size + gap: train_local_size + gap + val_size])

        if len(val_parts) == 0:
            raise ValueError("Nenhuma mini-janela de validação foi gerada.")

        train_idx = np.concatenate(train_parts).astype(int)
        val_idx = np.concatenate(val_parts).astype(int)

        train_idx = np.unique(train_idx)
        val_idx = np.unique(val_idx)
        overlap = np.intersect1d(train_idx, val_idx)
        if len(overlap) > 0:
            raise ValueError("Treino e validação internos ficaram sobrepostos.")
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Treino ou validação internos ficaram vazios.")

        return train_idx, val_idx

    def _build_internal_train_validation_indices(self, yn, node_id, depth):
        if self.validation_mode == "recent_temporal_windows":
            try:
                return self._build_recent_temporal_split_indices(len(yn))
            except Exception as e:
                self._log(1, f"[WARNING] fallback para random_holdout no node {node_id}: {e}", indent=1)

        stratify_y = yn if self.stratify else None
        try:
            return train_test_split(
                np.arange(len(yn)),
                test_size=self.validation_fraction,
                random_state=self.random_state + node_id + depth,
                stratify=stratify_y
            )
        except Exception:
            return train_test_split(
                np.arange(len(yn)),
                test_size=self.validation_fraction,
                random_state=self.random_state + node_id + depth,
                stratify=None
            )

    def _log_section(self, title, level=1):
        self._log(level, "=" * 110)
        self._log(level, title)
        self._log(level, "=" * 110)

    def _log_node_header(self, node_id, depth, y):
        y_arr = np.asarray(y)
        self._log(1, "=" * 100)
        self._log(
            1,
            f"NODE {node_id} | depth={depth} | n={len(y_arr)} | pos={int(np.sum(y_arr == 1))} | neg={int(np.sum(y_arr == 0))}"
        )

    def _log_screening_result(self, var, Ztr_shape=None, Zva_shape=None, metrics=None, indent=1):
        msg = f"[SCREEN] variável={var}"
        if Ztr_shape is not None and Zva_shape is not None:
            msg += f" | Ztr={Ztr_shape} | Zva={Zva_shape}"
        if metrics is not None:
            msg += " | " + self._fmt_metrics(metrics)
        self._log(2, msg, indent=indent)

    def _log_screening_top(self, screening_table, top_n=10, indent=1):
        if screening_table is None or screening_table.empty:
            self._log(2, "[SCREEN] nenhuma variável válida.", indent=indent)
            return
        self._log(2, f"[SCREEN] top {min(top_n, len(screening_table))} variáveis:", indent=indent)
        for _, row in screening_table.head(top_n).iterrows():
            self._log(
                2,
                f"rank={int(row['candidate_rank'])} | var={row['variable']} | obj={row['objective']:.6f} | "
                f"auc={row['auc']:.6f} | ks={row['ks']:.6f} | lift={row['lift']:.6f} | "
                f"precision={row['precision']:.6f} | error={row['error']:.6f} | rocmin={row['rocmin']:.6f}",
                indent=indent + 1
            )

    def _log_split_candidate(self, var, group, left_n=None, right_n=None, metrics=None, indent=2):
        msg = f"[SPLIT] var={var} | group={_group_text(group)}"
        if left_n is not None and right_n is not None:
            msg += f" | left_n={left_n} | right_n={right_n}"
        if metrics is not None:
            msg += " | " + self._fmt_metrics(metrics)
        self._log(3, msg, indent=indent)

    def _log_split_rejected(self, var, group, reason, indent=2):
        self._log(3, f"[REJECT] var={var} | group={_group_text(group)} | motivo={reason}", indent=indent)

    def _log_best_split(self, best_row, gain, indent=1):
        self._log(
            1,
            f"[BEST SPLIT] var={best_row['variable']} | group={best_row['group_text']} | objective={best_row['objective']:.6f} | gain={gain:.6f}",
            indent=indent
        )

    def _log_leaf_stacking_info(self, Z, leaf_ids):
        self._log(1, "[GLOBAL STACKING] iniciando alinhamento global das folhas")
        self._log(1, f"[GLOBAL STACKING] matriz Z shape={Z.shape}", indent=1)
        self._log(2, f"[GLOBAL STACKING] folhas usadas={leaf_ids}", indent=1)

    def _estimator_name(self, estimator):
        if estimator is None:
            return None
        return type(estimator).__name__

    def _estimator_repr(self, estimator):
        if estimator is None:
            return None
        try:
            return repr(estimator)
        except Exception:
            return type(estimator).__name__

    @staticmethod
    def help():
        return (
            "RiskSegOptimizer help\n"
            "Main estimators:\n"
            "- screening_estimator: model used in the screening/factorial step.\n"
            "- node_estimator: model used for the node/global benchmark.\n"
            "- segment_estimator: model used for left/right specialists and leaf models.\n"
            "- local_combiner_estimator: template/hyperparameters for the local combiner.\n"
            "- global_combiner_estimator: estimator used in global leaf stacking.\n"
            "Screening modes:\n"
            "- factorial_lr: backward-compatible alias for factorial_full.\n"
            "- factorial_full: dummies(var) + main effects + all dummy(var)*X interactions.\n"
            "- factorial_main_effects: dummies(var) + main effects only.\n"
            "- factorial_target_interactions: dummies(var) + main effects + interactions with a selected subset of X columns.\n"
            "- factorial_segment_only: only dummies(var), no main effects and no interactions.\n"
            "Factorial controls:\n"
            "- factorial_max_interaction_features: maximum number of X columns used in factorial_target_interactions.\n"
            "- factorial_feature_selector: currently 'variance' or fallback to column order.\n"
            "- factorial_include_main_effects: keep main effects in the factorial design.\n"
            "- factorial_drop_first: passed to get_dummies for the candidate segmentation variable.\n"
            "Important behavioral params:\n"
            "- combiner_method: stacking or marginal_odds.\n"
            "- prediction_mode: leaf, cascade, local_combiner, global_stacking.\n"
            "- metric: combined, lift, ks, precision, auc, error, rocmin.\n"
            "- auto_bin_numeric / n_numeric_bins / numeric_binning: numeric segmentation bins.\n"
            "- use_grouping / max_group_size: category grouping in split candidates.\n"
            "- top_k_variables / use_top_k_variables: how many screened variables are tested.\n"
            "- max_depth / min_samples_leaf / min_gain / validation_fraction: tree growth controls.\n"
            "Usage example:\n"
            "riskseg = RiskSegOptimizer(\n"
            "    screening_estimator=LogisticRegression(...),\n"
            "    node_estimator=LogisticRegression(...),\n"
            "    segment_estimator=LogisticRegression(...),\n"
            "    local_combiner_estimator=LogisticRegression(C=0.5,...),\n"
            "    global_combiner_estimator=LogisticRegression(C=0.2,...),\n"
            "    screening_mode='factorial_target_interactions',\n"
            "    factorial_max_interaction_features=12,\n"
            "    metric='combined',\n"
            "    combiner_method='stacking',\n"
            "    prediction_mode='global_stacking'\n"
            ")\n"
        )


    # --------------------------------------------------------------------------
    # API
    # --------------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        if y is None:
            raise ValueError("RiskSegOptimizer requires y to be passed, but the target y is None.")

        X_raw = _as_dataframe(X)
        y_raw = _as_series(y)

        if X_raw.shape[0] == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X_raw.shape}) while a minimum of 1 is required."
            )

        if X_raw.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape={X_raw.shape}) while a minimum of 1 is required."
            )

        if X_raw.shape[0] == 1:
            raise ValueError("RiskSegOptimizer requires at least 2 samples; got 1 sample.")

        if len(y_raw) != len(X_raw):
            raise ValueError("Found input variables with inconsistent numbers of samples.")

        if y_raw.isna().any():
            raise ValueError("Input y contains NaN.")

        target_type = type_of_target(y_raw)
        if target_type.startswith("continuous"):
            raise ValueError(f"Unknown label type: {target_type}")
        if target_type == "unknown":
            raise ValueError(f"Unknown label type: {target_type}")
        if target_type != "binary":
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {target_type}."
            )

        classes, y_encoded = np.unique(y_raw.to_numpy(), return_inverse=True)

        if len(classes) != 2:
            raise ValueError("RiskSegOptimizer requires a binary target with exactly two classes.")

        y = pd.Series(y_encoded.astype(int), name=y_raw.name, index=y_raw.index)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight deve ter o mesmo tamanho de y.")

        self.classes_ = classes
        self.n_features_in_ = X_raw.shape[1]
        self.feature_names_in_ = np.array(X_raw.columns, dtype=object)

        self.screening_estimator_ = self._build_screening_estimator()
        self.node_estimator_ = self._build_node_estimator()
        self.segment_estimator_ = self._build_segment_estimator()
        self.local_combiner_estimator_ = self._build_local_combiner_estimator()
        self.global_combiner_estimator_ = self._build_global_combiner_estimator()

        self.estimator_metadata_ = {
            "screening_estimator_name": self._estimator_name(self.screening_estimator_),
            "node_estimator_name": self._estimator_name(self.node_estimator_),
            "segment_estimator_name": self._estimator_name(self.segment_estimator_),
            "local_combiner_estimator_name": self._estimator_name(self.local_combiner_estimator_),
            "global_combiner_estimator_name": self._estimator_name(self.global_combiner_estimator_),
            "screening_estimator_repr": self._estimator_repr(self.screening_estimator_),
            "node_estimator_repr": self._estimator_repr(self.node_estimator_),
            "segment_estimator_repr": self._estimator_repr(self.segment_estimator_),
            "local_combiner_estimator_repr": self._estimator_repr(self.local_combiner_estimator_),
            "global_combiner_estimator_repr": self._estimator_repr(self.global_combiner_estimator_),
        }
        
        self._log(1, f"screening_estimator_={type(self.screening_estimator_).__name__}")
        self._log(1, f"node_estimator_={type(self.node_estimator_).__name__}")
        self._log(1, f"segment_estimator_={type(self.segment_estimator_).__name__}")
        self._log(1, f"local_combiner_estimator_={type(self.local_combiner_estimator_).__name__}")
        self._log(1, f"global_combiner_estimator_={type(self.global_combiner_estimator_).__name__}")

        self.model_view_, self.seg_view_ = self._prepare_views_fit(X_raw)
        self.global_prior_ = float(np.mean(y))
        self._node_counter_ = 0

        self._train_target_series_ = y.copy()
        self._train_original_X_ = X_raw.copy()

        self._log_section("RISKSEG | INÍCIO DO FIT", level=1)
        self._log(1, f"n_observations={len(y)}")
        self._log(1, f"n_original_variables={self.n_features_in_}")
        self._log(1, f"n_model_features_after_encoding={self.model_view_.shape[1]}")
        self._log(1, f"prediction_mode={self.prediction_mode}")
        self._log(1, f"screening_mode={self.screening_mode}")
        self._log(1, f"combiner_method={self.combiner_method}")
        self._log(1, f"metric={self.metric}")
        self._log(1, f"top_rate={self.top_rate}")
        self._log(1, f"metric_weights={self.metric_weights}")
        self._log(1, f"max_depth={self.max_depth}")
        self._log(1, f"min_samples_leaf={self.min_samples_leaf}")
        self._log(1, f"min_gain={self.min_gain}")
        self._log(1, f"validation_fraction={self.validation_fraction}")
        self._log(1, f"validation_mode={self.validation_mode}")
        self._log(1, f"recent_zone_size={self.recent_zone_size}")
        self._log(1, f"n_recent_windows={self.n_recent_windows}")
        self._log(1, f"recent_window_val_ratio={self.recent_window_val_ratio}")
        self._log(1, f"gap_size={self.gap_size}")
        self._log(1, f"auto_bin_numeric={self.auto_bin_numeric}")
        self._log(1, f"n_numeric_bins={self.n_numeric_bins}")
        self._log(1, f"numeric_binning={self.numeric_binning}")
        self._log(1, f"scale_model_numeric={self.scale_model_numeric}")
        self._log(1, f"model_numeric_scaling={self.model_numeric_scaling}")
        self._log(1, f"factorial_max_interaction_features={self.factorial_max_interaction_features}")
        self._log(1, f"factorial_feature_selector={self.factorial_feature_selector}")
        self._log(1, f"factorial_include_main_effects={self.factorial_include_main_effects}")
        self._log(1, f"factorial_drop_first={self.factorial_drop_first}")
        self._log(1, f"use_grouping={self.use_grouping}")
        self._log(1, f"max_group_size={self.max_group_size}")
        self._log(1, f"top_k_variables={self.top_k_variables}")
        self._log(1, f"screening_variables={self.screening_variables }")

        full_index = np.arange(len(y))
        self.root_ = self._fit_node(
            node_index=full_index,
            X_model=self.model_view_,
            X_seg=self.seg_view_,
            y=y,
            sample_weight=sample_weight,
            depth=0
        )

        self.leaf_nodes_ = self._collect_leaf_nodes(self.root_)
        self.node_table_ = self._collect_node_table(self.root_)
        self.split_table_ = self._collect_split_table(self.root_)

        self.global_simple_model_ = self._fit_node_model_safe(self.model_view_, y, sample_weight)
        p_global_simple = self.global_simple_model_.predict_proba(self.model_view_)[:, 1]
        self.global_simple_metrics_ = self._calculate_metrics(y, p_global_simple)
        self._log(1, "[GLOBAL SIMPLE] " + self._fmt_metrics(self.global_simple_metrics_))

        self.global_stacking_model_ = self._fit_global_leaf_stacking(
            X_model=self.model_view_,
            X_seg=self.seg_view_,
            y=y,
            sample_weight=sample_weight
        )
        p_global_stack = self.global_stacking_model_.predict_proba(
            self._build_global_leaf_matrix(self.model_view_, self.seg_view_)
        )[:, 1]
        self.global_stacking_metrics_ = self._calculate_metrics(y, p_global_stack)
        self._log(1, "[GLOBAL STACKING FINAL] " + self._fmt_metrics(self.global_stacking_metrics_))

        self.train_prediction_summary_ = self._evaluate_dataset_internal(
            X=self._train_original_X_,
            y=self._train_target_series_,
            dataset_name="train_final_prediction"
        )
        self._log(
            1,
            "[TRAIN FINAL PREDICTION] "
            f"objective={self.train_prediction_summary_['objective']:.6f} | "
            f"AUC={self.train_prediction_summary_['auc']:.6f} | "
            f"KS={self.train_prediction_summary_['ks']:.6f} | "
            f"Lift={self.train_prediction_summary_['lift']:.6f} | "
            f"Precision={self.train_prediction_summary_['precision']:.6f} | "
            f"Error={self.train_prediction_summary_['error']:.6f} | "
            f"ROCMIN={self.train_prediction_summary_['rocmin']:.6f}"
        )

        self.summary_ = self._build_summary_dict()
        self._log_section("RISKSEG | RESUMO FINAL", level=1)
        self._log(1, f"used_variables_unique={self.summary_['used_variables_unique']}")
        self._log(1, f"used_variables_sequence={self.summary_['used_variables_sequence']}")
        self._log(1, f"max_depth_reached={self.summary_['max_depth_reached']}")
        self._log(1, f"n_nodes={self.summary_['n_nodes']}")
        self._log(1, f"n_internal_nodes={self.summary_['n_internal_nodes']}")
        self._log(1, f"n_leaves={self.summary_['n_leaves']}")
        self._log(1, f"screening_estimator_name={self.summary_['screening_estimator_name']}")
        self._log(1, f"node_estimator_name={self.summary_['node_estimator_name']}")
        self._log(1, f"segment_estimator_name={self.summary_['segment_estimator_name']}")
        self._log(1, f"local_combiner_estimator_name={self.summary_['local_combiner_estimator_name']}")
        self._log(1, f"global_combiner_estimator_name={self.summary_['global_combiner_estimator_name']}")
        self._log(1, f"GLOBAL SIMPLE   | AUC={self.summary_['global_simple_auc']:.6f} | KS={self.summary_['global_simple_ks']:.6f} | Lift={self.summary_['global_simple_lift']:.6f} | Error={self.summary_['global_simple_error']:.6f}")
        self._log(1, f"RISKSEG FINAL   | AUC={self.summary_['riskseg_global_stacking_auc']:.6f} | KS={self.summary_['riskseg_global_stacking_ks']:.6f} | Lift={self.summary_['riskseg_global_stacking_lift']:.6f} | Error={self.summary_['riskseg_global_stacking_error']:.6f}")
        self._log(1, f"TRAIN RE-PRED   | AUC={self.summary_['train_final_prediction_auc']:.6f} | KS={self.summary_['train_final_prediction_ks']:.6f} | Lift={self.summary_['train_final_prediction_lift']:.6f} | Error={self.summary_['train_final_prediction_error']:.6f}")
        self._log(1, f"DELTA           | AUC={self.summary_['delta_auc_vs_global_simple']:.6f} | KS={self.summary_['delta_ks_vs_global_simple']:.6f} | Lift={self.summary_['delta_lift_vs_global_simple']:.6f} | Error={self.summary_['delta_error_vs_global_simple']:.6f}")

        self.is_fitted_ = True
        return self

    def _predict_proba_internal(self, X):
        """
        Versão interna de predict_proba para uso durante o fit,
        sem check_is_fitted.
        """
        X_raw = _as_dataframe(X, feature_names=self.feature_names_in_)
        X_model, X_seg = self._prepare_views_predict(X_raw)

        if self.prediction_mode == "leaf":
            out = np.zeros((len(X_model), 2), dtype=float)
            for i in range(len(X_model)):
                out[i] = self._predict_one_leaf(X_model.iloc[[i]], X_seg.iloc[[i]])
            return out

        if self.prediction_mode == "cascade":
            out = np.zeros((len(X_model), 2), dtype=float)
            for i in range(len(X_model)):
                out[i] = self._predict_one_cascade(X_model.iloc[[i]], X_seg.iloc[[i]], self.root_)
            return out

        if self.prediction_mode == "local_combiner":
            out = np.zeros((len(X_model), 2), dtype=float)
            for i in range(len(X_model)):
                out[i] = self._predict_one_local_combiner(X_model.iloc[[i]], X_seg.iloc[[i]], self.root_)
            return out

        if self.prediction_mode == "global_stacking":
            Z = self._build_global_leaf_matrix(X_model, X_seg)
            return self.global_stacking_model_.predict_proba(Z)

        raise ValueError("prediction_mode deve ser 'leaf', 'cascade', 'local_combiner' ou 'global_stacking'.")    


    def predict_proba(self, X):
            check_is_fitted(self, "is_fitted_")
            return self._predict_proba_internal(X)



    def predict(self, X):
        encoded = (self.predict_proba(X)[:, 1] >= self.classification_threshold).astype(int)
        return self.classes_[encoded]

    def decision_function(self, X):
        return _safe_logit(self.predict_proba(X)[:, 1])

    def _encode_target_for_metrics(self, y):
        y_series = _as_series(y)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        encoded = y_series.map(class_to_index)
        if encoded.isna().any():
            unknown = sorted(set(y_series[encoded.isna()].tolist()))
            raise ValueError(f"y contains previously unseen classes: {unknown}")
        return encoded.astype(int)

    def score(self, X, y):
        y = self._encode_target_for_metrics(y)
        p = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, p)

    def get_node_table(self):
        check_is_fitted(self, "is_fitted_")
        return self.node_table_.copy()

    def get_split_table(self):
        check_is_fitted(self, "is_fitted_")
        return self.split_table_.copy()

    def get_summary(self):
        check_is_fitted(self, "is_fitted_")
        return self.summary_.copy()

    def evaluate_dataset(self, X, y, dataset_name="external"):
        """
        Avalia qualquer base externa (teste, validação, holdout etc.)
        sem alterar o summary_ atual de treino.
        """
        check_is_fitted(self, "is_fitted_")

        y_series = self._encode_target_for_metrics(y)
        p = self.predict_proba(X)[:, 1]
        metrics = self._calculate_metrics(y_series, p)

        result = {
            "dataset_name": dataset_name,
            "n_observations": int(len(y_series)),
            "objective": float(self._objective_from_metrics(metrics)),
            "auc": float(metrics["auc"]),
            "ks": float(metrics["ks"]),
            "lift": float(metrics["lift"]),
            "precision": float(metrics["precision"]),
            "error": float(metrics["error"]),
            "rocmin": float(metrics["rocmin"]),
            "prediction_mode": self.prediction_mode,
            "combiner_method": self.combiner_method,
            "screening_mode": self.screening_mode,
            "factorial_max_interaction_features": self.factorial_max_interaction_features,
            "factorial_feature_selector": self.factorial_feature_selector,
            "factorial_include_main_effects": self.factorial_include_main_effects,
            "factorial_drop_first": self.factorial_drop_first,
            "max_depth_reached": int(self.summary_["max_depth_reached"]),
            "n_nodes": int(self.summary_["n_nodes"]),
            "n_leaves": int(self.summary_["n_leaves"]),
            "used_variables_unique": list(self.summary_["used_variables_unique"]),
            "used_variables_sequence": list(self.summary_["used_variables_sequence"]),
            "screening_estimator_name": self.estimator_metadata_["screening_estimator_name"],
            "node_estimator_name": self.estimator_metadata_["node_estimator_name"],
            "segment_estimator_name": self.estimator_metadata_["segment_estimator_name"],
            "local_combiner_estimator_name": self.estimator_metadata_["local_combiner_estimator_name"],
            "global_combiner_estimator_name": self.estimator_metadata_["global_combiner_estimator_name"],
            "screening_estimator_repr": self.estimator_metadata_["screening_estimator_repr"],
            "node_estimator_repr": self.estimator_metadata_["node_estimator_repr"],
            "segment_estimator_repr": self.estimator_metadata_["segment_estimator_repr"],
            "local_combiner_estimator_repr": self.estimator_metadata_["local_combiner_estimator_repr"],
            "global_combiner_estimator_repr": self.estimator_metadata_["global_combiner_estimator_repr"]
        }

        return result

    def evaluate_train_dataset(self):
        """
        Avaliação explícita da base de treino pela mesma rota de predição atual.
        """
        check_is_fitted(self, "is_fitted_")
        return self.evaluate_dataset(
            X=self._rebuild_original_like_X_from_views(),
            y=self._train_target_series_,
            dataset_name="train_repredicted"
        )

    def print_evaluation_summary(self, summary_dict, prefix="EVAL"):
        """
        Imprime resumo compacto de avaliação externa.
        """
        self._log(1, "=" * 110)
        self._log(1, f"{prefix} | dataset_name={summary_dict['dataset_name']}")
        self._log(1, f"{prefix} | n_observations={summary_dict['n_observations']}")
        self._log(
            1,
            f"{prefix} | objective={summary_dict['objective']:.6f} | "
            f"AUC={summary_dict['auc']:.6f} | "
            f"KS={summary_dict['ks']:.6f} | "
            f"Lift={summary_dict['lift']:.6f} | "
            f"Precision={summary_dict['precision']:.6f} | "
            f"Error={summary_dict['error']:.6f} | "
            f"ROCMIN={summary_dict['rocmin']:.6f}"
        )
        self._log(1, f"{prefix} | used_variables_unique={summary_dict['used_variables_unique']}")
        self._log(1, "=" * 110)

    def _rebuild_original_like_X_from_views(self):
        """
        Helper simples para avaliação da própria base de treino sem depender do X original externo.
        """
        X_back = self.seg_view_.copy()
        X_back = X_back.reindex(columns=self.feature_names_in_)
        return X_back

    # --------------------------------------------------------------------------
    # TREINO DOS NÓS
    # --------------------------------------------------------------------------

    def _fit_node(self, node_index, X_model, X_seg, y, sample_weight, depth):
        node_id = self._next_node_id()
        idx = np.asarray(node_index, dtype=int)

        node = RiskSegNode(
            node_id=node_id,
            depth=depth,
            train_index=idx
        )

        Xn_model = X_model.iloc[idx].reset_index(drop=True)
        Xn_seg = X_seg.iloc[idx].reset_index(drop=True)
        yn = y.iloc[idx].reset_index(drop=True)
        wn = sample_weight[idx] if sample_weight is not None else None

        self._log_node_header(node_id, depth, yn)

        node.node_model = self._fit_node_model_safe(Xn_model, yn, wn)
        p_train = node.node_model.predict_proba(Xn_model)[:, 1]
        node.global_metrics_train = self._calculate_metrics(yn, p_train)
        self._log(1, f"[NODE {node_id}] GLOBAL(train) | {self._fmt_metrics(node.global_metrics_train)}", indent=1)

        if depth >= self.max_depth:
            self._log(1, f"[NODE {node_id}] STOP | max_depth atingido.", indent=1)
            node.is_leaf = True
            return node

        if not self._can_split_node(yn):
            self._log(1, f"[NODE {node_id}] STOP | massa insuficiente para split.", indent=1)
            node.is_leaf = True
            return node

        idx_train_rel, idx_val_rel = self._build_internal_train_validation_indices(yn, node_id, depth)

        Xtr_model = Xn_model.iloc[idx_train_rel].reset_index(drop=True)
        Xva_model = Xn_model.iloc[idx_val_rel].reset_index(drop=True)
        Xtr_seg = Xn_seg.iloc[idx_train_rel].reset_index(drop=True)
        Xva_seg = Xn_seg.iloc[idx_val_rel].reset_index(drop=True)
        ytr = yn.iloc[idx_train_rel].reset_index(drop=True)
        yva = yn.iloc[idx_val_rel].reset_index(drop=True)
        wtr = wn[idx_train_rel] if wn is not None else None
        wva = wn[idx_val_rel] if wn is not None else None

        global_val_model = self._fit_node_model_safe(Xtr_model, ytr, wtr)
        p_global_val = global_val_model.predict_proba(Xva_model)[:, 1]
        node.global_metrics_val = self._calculate_metrics(yva, p_global_val)
        global_val_obj = self._objective_from_metrics(node.global_metrics_val)
        self._log(1, f"[NODE {node_id}] GLOBAL(val)   | {self._fmt_metrics(node.global_metrics_val)}", indent=1)

        screening_table = self._screen_candidate_variables(
            X_model_train=Xtr_model,
            X_seg_train=Xtr_seg,
            y_train=ytr,
            X_model_val=Xva_model,
            X_seg_val=Xva_seg,
            y_val=yva,
            sample_weight_train=wtr,
            allowed_vars=self._get_allowed_screening_variables(Xtr_seg.columns)
        )
        node.screening_table = screening_table.copy()
        self._log_screening_top(screening_table, top_n=min(10, len(screening_table)) if len(screening_table) > 0 else 0, indent=1)

        if screening_table.empty:
            self._log(1, f"[NODE {node_id}] STOP | screening vazio.", indent=1)
            node.is_leaf = True
            return node

        if self.use_top_k_variables:
            candidate_vars = screening_table["variable"].head(self.top_k_variables).tolist()
        else:
            candidate_vars = screening_table["variable"].tolist()
        self._log(1, f"[NODE {node_id}] variáveis candidatas ao split: {candidate_vars}", indent=1)

        split_table = self._evaluate_candidate_splits(
            candidate_vars=candidate_vars,
            X_model_train=Xtr_model,
            X_seg_train=Xtr_seg,
            y_train=ytr,
            X_model_val=Xva_model,
            X_seg_val=Xva_seg,
            y_val=yva,
            sample_weight_train=wtr,
            sample_weight_val=wva
        )
        node.split_candidates_table = split_table.copy()

        if split_table is not None and not split_table.empty:
            self._log(2, f"[NODE {node_id}] top candidatos de split:", indent=1)
            for i, (_, row) in enumerate(split_table.head(min(10, len(split_table))).iterrows(), start=1):
                self._log(
                    2,
                    f"#{i:02d} | var={row['variable']} | group={row['group_text']} | obj={row['objective']:.6f} | auc={row['auc']:.6f} | ks={row['ks']:.6f} | lift={row['lift']:.6f} | left_n={int(row['left_n'])} | right_n={int(row['right_n'])}",
                    indent=2
                )

        if split_table.empty:
            self._log(1, f"[NODE {node_id}] STOP | split_table vazia.", indent=1)
            node.is_leaf = True
            return node

        best = split_table.iloc[0].to_dict()
        gain = best["objective"] - global_val_obj
        self._log_best_split(best, gain, indent=1)

        accept_split = True
        if self.use_validation_to_accept_split:
            accept_split = gain > self.min_gain

        if self.global_only_if_no_gain and not accept_split:
            self._log(1, f"[NODE {node_id}] STOP | melhor split não superou min_gain={self.min_gain:.6f}.", indent=1)
            node.is_leaf = True
            return node

        var = best["variable"]
        group = best["group"]

        node.is_leaf = False
        node.split_variable = var
        node.split_group = group
        node.split_rule_text = f"{var} in {_group_text(group)} vs {var} not in {_group_text(group)}"
        node.split_gain = gain
        node.split_metrics_val = {
            "objective": float(best["objective"]),
            "lift": float(best["lift"]),
            "ks": float(best["ks"]),
            "precision": float(best["precision"]),
            "auc": float(best["auc"]),
            "error": float(best["error"]),
            "rocmin": float(best["rocmin"])
        }

        left_mask_full = Xn_seg[var].astype(str).isin(group).to_numpy()
        right_mask_full = ~left_mask_full

        left_idx_abs = idx[left_mask_full]
        right_idx_abs = idx[right_mask_full]

        y_left = y.iloc[left_idx_abs]
        y_right = y.iloc[right_idx_abs]

        node.left_support = {
            "n": int(len(left_idx_abs)),
            "positives": int(np.sum(y_left == 1)),
            "negatives": int(np.sum(y_left == 0))
        }
        node.right_support = {
            "n": int(len(right_idx_abs)),
            "positives": int(np.sum(y_right == 1)),
            "negatives": int(np.sum(y_right == 0))
        }
        self._log(1, f"[NODE {node_id}] ACCEPTED SPLIT | {node.split_rule_text}", indent=1)
        self._log(1, f"[NODE {node_id}] LEFT  | {node.left_support}", indent=2)
        self._log(1, f"[NODE {node_id}] RIGHT | {node.right_support}", indent=2)

        X_left_model = X_model.iloc[left_idx_abs]
        X_right_model = X_model.iloc[right_idx_abs]
        y_left_full = y.iloc[left_idx_abs]
        y_right_full = y.iloc[right_idx_abs]
        w_left_full = sample_weight[left_idx_abs] if sample_weight is not None else None
        w_right_full = sample_weight[right_idx_abs] if sample_weight is not None else None

        node.left_model = self._fit_segment_model_safe(X_left_model, y_left_full, w_left_full)
        node.right_model = self._fit_segment_model_safe(X_right_model, y_right_full, w_right_full)

        # IMPORTANTE: combiner local ajustado no TREINO do nó, validado na VALIDAÇÃO.
        left_mask_tr = Xtr_seg[var].astype(str).isin(group).to_numpy()
        right_mask_tr = ~left_mask_tr
        Ztr_comb, mtr = self._build_combiner_matrix(
            left_model=node.left_model,
            right_model=node.right_model,
            X_val=Xtr_model,
            left_mask=left_mask_tr,
            right_mask=right_mask_tr
        )
        node.combiner = self._build_combiner()
        self._fit_combiner(node.combiner, Ztr_comb, ytr.to_numpy(), membership=mtr, sample_weight=wtr)
        self._log(2, f"[NODE {node_id}] combiner local treinado no treino interno | Ztr_comb={Ztr_comb.shape}", indent=1)

        node.left_child = self._fit_node(
            node_index=left_idx_abs,
            X_model=X_model,
            X_seg=X_seg,
            y=y,
            sample_weight=sample_weight,
            depth=depth + 1
        )
        node.right_child = self._fit_node(
            node_index=right_idx_abs,
            X_model=X_model,
            X_seg=X_seg,
            y=y,
            sample_weight=sample_weight,
            depth=depth + 1
        )

        return node

    # --------------------------------------------------------------------------
    # SCREENING POR VARIÁVEL
    # --------------------------------------------------------------------------

    def _screen_candidate_variables(
        self,
        X_model_train,
        X_seg_train,
        y_train,
        X_model_val,
        X_seg_val,
        y_val,
        sample_weight_train=None,
        allowed_vars=None
    ):
        rows = []

        if allowed_vars is None:
            allowed_vars = list(X_seg_train.columns)

        for var in allowed_vars:
            self._log(2, f"[SCREEN] iniciando avaliação da variável candidata: {var}", indent=1)
            try:
                Ztr, Zva = self._build_factorial_design_for_variable(
                    var_name=var,
                    X_model_train=X_model_train,
                    X_seg_train=X_seg_train,
                    X_model_val=X_model_val,
                    X_seg_val=X_seg_val
                )

                if Ztr.shape[1] == 0:
                    continue

                self._log(4, f"[SCREEN] var={var} | desenho fatorial montado | Ztr={Ztr.shape} | Zva={Zva.shape}", indent=2)

                model = clone(self.screening_estimator_)
                model, had_warning = self._fit_estimator_with_warning_control(
                    estimator=model,
                    X=Ztr,
                    y=y_train,
                    sample_weight=sample_weight_train,
                    context=f"screening_var={var}"
                )

                if had_warning:
                    self._log(2, f"[SCREEN] variável={var} ajustada com ConvergenceWarning", indent=2)

                pva = model.predict_proba(Zva)[:, 1]
                met = self._calculate_metrics(y_val, pva)
                obj = self._objective_from_metrics(met)

                rows.append({
                    "variable": var,
                    "objective": obj,
                    "lift": met["lift"],
                    "ks": met["ks"],
                    "precision": met["precision"],
                    "auc": met["auc"],
                    "error": met["error"],
                    "rocmin": met["rocmin"],
                    "n_factorial_features": int(Ztr.shape[1])
                })
                self._log_screening_result(var=var, Ztr_shape=Ztr.shape, Zva_shape=Zva.shape, metrics=met, indent=2)

            except Exception as e:
                self._log(3, f"[SCREEN][ERRO] var={var} | erro={e}", indent=2)

        if len(rows) == 0:
            return pd.DataFrame(columns=[
                "variable", "objective", "lift", "ks", "precision", "auc", "error", "rocmin", "n_factorial_features"
            ])

        df = pd.DataFrame(rows).sort_values(
            ["objective", "lift", "ks", "precision", "auc"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
        df["candidate_rank"] = np.arange(1, len(df) + 1)
        return df

    def _select_factorial_interaction_columns(self, X_model_train):
        cols = list(X_model_train.columns)

        if self.factorial_max_interaction_features is None:
            return cols

        k = int(self.factorial_max_interaction_features)
        if k <= 0:
            return []

        if self.factorial_feature_selector == "variance":
            variances = X_model_train.var(axis=0).sort_values(ascending=False)
            return variances.head(min(k, len(variances))).index.tolist()

        return cols[:min(k, len(cols))]

    def _build_factorial_design_for_variable(
        self,
        var_name,
        X_model_train,
        X_seg_train,
        X_model_val,
        X_seg_val
    ):
        """
        Modos de screening fatorial:

        - factorial_full:
            dummies(var) + main effects + interações dummy(var) * X_model

        - factorial_main_effects:
            dummies(var) + main effects

        - factorial_target_interactions:
            dummies(var) + main effects + interações com subconjunto de colunas

        - factorial_segment_only:
            apenas dummies(var)
        """

        valid_modes = {
            "factorial_lr",
            "factorial_full",
            "factorial_main_effects",
            "factorial_target_interactions",
            "factorial_segment_only"
        }

        if self.screening_mode not in valid_modes:
            raise ValueError(
                "screening_mode deve ser um de: "
                "'factorial_lr', 'factorial_full', 'factorial_main_effects', "
                "'factorial_target_interactions', 'factorial_segment_only'."
            )

        mode = self.screening_mode
        if mode == "factorial_lr":
            mode = "factorial_full"

        xi_train = X_seg_train[var_name].astype(str)
        xi_val = X_seg_val[var_name].astype(str)

        xi_train_dummies = pd.get_dummies(
            xi_train,
            prefix=f"{var_name}__seg",
            drop_first=self.factorial_drop_first
        )
        xi_val_dummies = pd.get_dummies(
            xi_val,
            prefix=f"{var_name}__seg",
            drop_first=self.factorial_drop_first
        )

        for col in xi_train_dummies.columns:
            if col not in xi_val_dummies.columns:
                xi_val_dummies[col] = 0
        for col in xi_val_dummies.columns:
            if col not in xi_train_dummies.columns:
                xi_train_dummies[col] = 0

        xi_train_dummies = xi_train_dummies[xi_train_dummies.columns.sort_values()]
        xi_val_dummies = xi_val_dummies[xi_train_dummies.columns]

        main_train = X_model_train.reset_index(drop=True)
        main_val = X_model_val.reset_index(drop=True)

        dummy_train = xi_train_dummies.reset_index(drop=True).to_numpy(dtype=float)
        dummy_val = xi_val_dummies.reset_index(drop=True).to_numpy(dtype=float)

        main_train_np = main_train.to_numpy(dtype=float)
        main_val_np = main_val.to_numpy(dtype=float)

        if mode == "factorial_segment_only":
            return dummy_train, dummy_val

        parts_train = [dummy_train]
        parts_val = [dummy_val]

        if self.factorial_include_main_effects or mode in {"factorial_main_effects", "factorial_full", "factorial_target_interactions"}:
            parts_train.append(main_train_np)
            parts_val.append(main_val_np)

        if mode == "factorial_main_effects":
            Ztr = np.column_stack(parts_train)
            Zva = np.column_stack(parts_val)
            return Ztr, Zva

        if mode == "factorial_full":
            interaction_cols = list(main_train.columns)
        elif mode == "factorial_target_interactions":
            interaction_cols = self._select_factorial_interaction_columns(main_train)
        else:
            interaction_cols = []

        if len(interaction_cols) > 0:
            inter_train_parts = []
            inter_val_parts = []

            sub_train = main_train[interaction_cols].to_numpy(dtype=float)
            sub_val = main_val[interaction_cols].to_numpy(dtype=float)

            for j in range(dummy_train.shape[1]):
                dtr = dummy_train[:, j].reshape(-1, 1)
                dva = dummy_val[:, j].reshape(-1, 1)

                inter_train_parts.append(sub_train * dtr)
                inter_val_parts.append(sub_val * dva)

            if len(inter_train_parts) > 0:
                parts_train.extend(inter_train_parts)
                parts_val.extend(inter_val_parts)

        Ztr = np.column_stack(parts_train)
        Zva = np.column_stack(parts_val)
        return Ztr, Zva

    # --------------------------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------------------------

    def _evaluate_candidate_splits(
        self,
        candidate_vars,
        X_model_train,
        X_seg_train,
        y_train,
        X_model_val,
        X_seg_val,
        y_val,
        sample_weight_train=None,
        sample_weight_val=None
    ):
        rows = []

        for candidate_rank, var in enumerate(candidate_vars, start=1):
            self._log(2, f"[SPLIT] avaliando grupos para variável: {var}", indent=1)
            groups = self._generate_split_groups(X_seg_train[var].astype(str))

            for group in groups:
                self._log(4, f"[SPLIT] testando group={_group_text(group)}", indent=2)
                try:
                    left_mask_tr = X_seg_train[var].astype(str).isin(group).to_numpy()
                    right_mask_tr = ~left_mask_tr
                    left_mask_va = X_seg_val[var].astype(str).isin(group).to_numpy()
                    right_mask_va = ~left_mask_va

                    if not self._valid_masks(y_train, left_mask_tr, right_mask_tr):
                        self._log_split_rejected(var, group, "máscara inválida no treino", indent=2)
                        continue
                    if not self._valid_masks(y_val, left_mask_va, right_mask_va):
                        self._log_split_rejected(var, group, "máscara inválida na validação", indent=2)
                        continue

                    left_model = self._fit_segment_model_safe(
                        X_model_train.iloc[left_mask_tr],
                        y_train.iloc[left_mask_tr],
                        sample_weight_train[left_mask_tr] if sample_weight_train is not None else None
                    )
                    right_model = self._fit_segment_model_safe(
                        X_model_train.iloc[right_mask_tr],
                        y_train.iloc[right_mask_tr],
                        sample_weight_train[right_mask_tr] if sample_weight_train is not None else None
                    )

                    # IMPORTANTE: combiner ajustado no TREINO, não na validação.
                    Ztr, mtr = self._build_combiner_matrix(
                        left_model=left_model,
                        right_model=right_model,
                        X_val=X_model_train,
                        left_mask=left_mask_tr,
                        right_mask=right_mask_tr
                    )
                    combiner = self._build_combiner()
                    self._fit_combiner(
                        combiner,
                        Ztr,
                        y_train.to_numpy(),
                        membership=mtr,
                        sample_weight=sample_weight_train
                    )

                    Zva, mva = self._build_combiner_matrix(
                        left_model=left_model,
                        right_model=right_model,
                        X_val=X_model_val,
                        left_mask=left_mask_va,
                        right_mask=right_mask_va
                    )

                    pva = self._combiner_predict_proba(combiner, Zva, membership=mva)[:, 1]
                    met = self._calculate_metrics(y_val, pva)
                    obj = self._objective_from_metrics(met)

                    rows.append({
                        "variable": var,
                        "group": group,
                        "group_text": _group_text(group),
                        "candidate_rank": candidate_rank,
                        "objective": obj,
                        "lift": met["lift"],
                        "ks": met["ks"],
                        "precision": met["precision"],
                        "auc": met["auc"],
                        "error": met["error"],
                        "rocmin": met["rocmin"],
                        "left_n": int(np.sum(left_mask_tr)),
                        "right_n": int(np.sum(right_mask_tr)),
                        "left_pos": int(np.sum(y_train.iloc[left_mask_tr] == 1)),
                        "right_pos": int(np.sum(y_train.iloc[right_mask_tr] == 1))
                    })

                    self._log_split_candidate(
                        var=var,
                        group=group,
                        left_n=int(np.sum(left_mask_tr)),
                        right_n=int(np.sum(right_mask_tr)),
                        metrics=met,
                        indent=2
                    )

                except Exception as e:
                    self._log(3, f"[SPLIT][ERRO] var={var} | group={_group_text(group)} | erro={e}", indent=2)

        if len(rows) == 0:
            return pd.DataFrame(columns=[
                "variable", "group", "group_text", "candidate_rank", "objective",
                "lift", "ks", "precision", "auc", "error", "rocmin",
                "left_n", "right_n", "left_pos", "right_pos"
            ])

        df = pd.DataFrame(rows).sort_values(
            ["objective", "lift", "ks", "precision", "auc"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
        return df

    def _build_combiner(self):
        if self.combiner_method == "stacking":
            return LogisticStackingCombiner(
                random_state=self.random_state,
                C=getattr(self.local_combiner_estimator_, "C", 1.0) if self.local_combiner_estimator_ is not None else 1.0,
                max_iter=getattr(self.local_combiner_estimator_, "max_iter", 5000) if self.local_combiner_estimator_ is not None else 5000
            )

        if self.combiner_method == "marginal_odds":
            return MarginalOddsCombiner(
                random_state=self.random_state,
                max_iter=getattr(self.local_combiner_estimator_, "max_iter", 5000) if self.local_combiner_estimator_ is not None else 5000
            )

        raise ValueError("combiner_method deve ser 'stacking' ou 'marginal_odds'.")

    def _fit_combiner(self, combiner, Z, y, membership=None, sample_weight=None):
        if self.combiner_method == "marginal_odds":
            combiner.fit(Z, y, membership=membership, sample_weight=sample_weight)
            if getattr(combiner, "had_convergence_warning_", False):
                self._log(3, "[WARNING] ConvergenceWarning no combiner marginal_odds", indent=2)
            return combiner

        combiner.fit(Z, y, sample_weight=sample_weight)

        if getattr(combiner, "had_convergence_warning_", False):
            self._log(3, "[WARNING] ConvergenceWarning no combiner stacking", indent=2)

        return combiner

    def _combiner_predict_proba(self, combiner, Z, membership=None):
        if self.combiner_method == "marginal_odds":
            return combiner.predict_proba(Z, membership=membership)
        return combiner.predict_proba(Z)

    def _build_combiner_matrix(self, left_model, right_model, X_val, left_mask, right_mask):
        Z = np.zeros((len(X_val), 2), dtype=float)
        membership = np.full(len(X_val), -1, dtype=int)

        if np.any(left_mask):
            p_left = left_model.predict_proba(X_val.iloc[left_mask])[:, 1]
            Z[left_mask, 0] = p_left
            membership[left_mask] = 0

        if np.any(right_mask):
            p_right = right_model.predict_proba(X_val.iloc[right_mask])[:, 1]
            Z[right_mask, 1] = p_right
            membership[right_mask] = 1

        return Z, membership

    # --------------------------------------------------------------------------
    # GLOBAL STACKING DAS FOLHAS
    # --------------------------------------------------------------------------

    def _collect_leaf_nodes(self, root):
        leaves = []

        def walk(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                walk(node.left_child)
                walk(node.right_child)

        walk(root)
        return leaves

    def _route_to_leaf_id(self, x_seg):
        node = self.root_
        while not node.is_leaf:
            val = str(x_seg.iloc[0][node.split_variable])
            node = node.left_child if val in node.split_group else node.right_child
        return node.node_id

    def _build_global_leaf_matrix(self, X_model, X_seg):
        leaf_ids = [leaf.node_id for leaf in self.leaf_nodes_]
        leaf_pos = {leaf_id: j for j, leaf_id in enumerate(leaf_ids)}
        Z = np.zeros((len(X_model), len(leaf_ids)), dtype=float)

        leaf_dict = {leaf.node_id: leaf for leaf in self.leaf_nodes_}

        for i in range(len(X_model)):
            x_model = X_model.iloc[[i]]
            x_seg = X_seg.iloc[[i]]
            leaf_id = self._route_to_leaf_id(x_seg)
            leaf_j = leaf_pos[leaf_id]
            leaf_node = leaf_dict[leaf_id]
            Z[i, leaf_j] = leaf_node.node_model.predict_proba(x_model)[:, 1][0]

        return Z

    def _fit_global_leaf_stacking(self, X_model, X_seg, y, sample_weight=None):
        Z = self._build_global_leaf_matrix(X_model, X_seg)
        leaf_ids = [leaf.node_id for leaf in self.leaf_nodes_]
        self._log_leaf_stacking_info(Z, leaf_ids)

        model = clone(self.global_combiner_estimator_)

        model, had_warning = self._fit_estimator_with_warning_control(
            estimator=model,
            X=Z,
            y=y,
            sample_weight=sample_weight,
            context="global_leaf_stacking"
        )

        if had_warning:
            self._log(2, "[GLOBAL STACKING] ajuste com ConvergenceWarning", indent=1)

        self._log(1, "[GLOBAL STACKING] ajuste concluído", indent=1)
        return model

    # --------------------------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------------------------

    def _predict_one_leaf(self, x_model, x_seg):
        node = self.root_
        while not node.is_leaf:
            val = str(x_seg.iloc[0][node.split_variable])
            node = node.left_child if val in node.split_group else node.right_child
        return node.node_model.predict_proba(x_model)[0]

    def _predict_one_node_combiner(self, x_model, x_seg, node):
        val = str(x_seg.iloc[0][node.split_variable])
        in_left = val in node.split_group
        membership = np.array([0 if in_left else 1], dtype=int)

        Z = np.zeros((1, 2), dtype=float)
        if in_left:
            Z[0, 0] = node.left_model.predict_proba(x_model)[:, 1][0]
        else:
            Z[0, 1] = node.right_model.predict_proba(x_model)[:, 1][0]

        return self._combiner_predict_proba(node.combiner, Z, membership=membership)[0]

    def _predict_one_local_combiner(self, x_model, x_seg, node):
        if node.is_leaf:
            return node.node_model.predict_proba(x_model)[0]

        val = str(x_seg.iloc[0][node.split_variable])
        child = node.left_child if val in node.split_group else node.right_child

        if child is not None and not child.is_leaf:
            return self._predict_one_local_combiner(x_model, x_seg, child)

        return self._predict_one_node_combiner(x_model, x_seg, node)

    def _predict_one_cascade(self, x_model, x_seg, node):
        if node.is_leaf:
            return node.node_model.predict_proba(x_model)[0]

        val = str(x_seg.iloc[0][node.split_variable])
        in_left = val in node.split_group
        membership = np.array([0 if in_left else 1], dtype=int)

        p_left = node.left_model.predict_proba(x_model)[:, 1]
        p_right = node.right_model.predict_proba(x_model)[:, 1]

        Z = np.zeros((1, 2), dtype=float)
        if in_left:
            Z[0, 0] = p_left[0]
        else:
            Z[0, 1] = p_right[0]

        p_node = self._combiner_predict_proba(node.combiner, Z, membership=membership)[:, 1][0]
        child = node.left_child if in_left else node.right_child
        p_child = self._predict_one_cascade(x_model, x_seg, child)[1]

        p_final = 0.5 * p_node + 0.5 * p_child
        return np.array([1 - p_final, p_final], dtype=float)

    # --------------------------------------------------------------------------
    # PREPARAÇÃO DE DADOS
    # --------------------------------------------------------------------------

    def _prepare_views_fit(self, X):
        X = X.copy()

        self._categorical_features_ = self._resolve_categorical_features(X)
        self._model_numeric_scalers_ = {}
        self._seg_numeric_edges_ = {}
        self._seg_numeric_label_order_ = {}
        self._seg_columns_ = list(X.columns)

        seg_data = {}

        for col in X.columns:
            s = X[col]

            is_explicit_categorical = col in self._categorical_features_
            if pd.api.types.is_numeric_dtype(s) and self.auto_bin_numeric and not is_explicit_categorical:
                binned, edges = self._bin_numeric_series(s)
                seg_data[col] = binned.astype(str)
                self._seg_numeric_edges_[col] = edges
                self._seg_numeric_label_order_[col] = [str(category) for category in binned.cat.categories]
            else:
                seg_data[col] = s.astype(str)
    
        seg_df = pd.DataFrame(seg_data, index=X.index)

        model_df = X.copy()
        self._apply_model_numeric_scaling_fit(model_df)
        cat_cols = [
            c for c in model_df.columns
            if c in self._categorical_features_ or not pd.api.types.is_numeric_dtype(model_df[c])
        ]
        if len(cat_cols) > 0:
            model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=False)

        model_df = model_df.astype(float)
        self._model_columns_ = list(model_df.columns)

        return model_df.reset_index(drop=True), seg_df.reset_index(drop=True)

    def _prepare_views_predict(self, X):
        X = X.copy()

        seg_data = {}

        for col in self.feature_names_in_:
            s = X[col]

            if col in self._seg_numeric_edges_:
                edges = self._seg_numeric_edges_[col]
                seg_data[col] = pd.cut(
                    s.astype(float),
                    bins=[-np.inf] + list(edges) + [np.inf],
                    include_lowest=True
                ).astype(str)
            else:
                seg_data[col] = s.astype(str)
    
        seg_df = pd.DataFrame(seg_data, index=X.index)

        model_df = X.copy()
        self._apply_model_numeric_scaling_predict(model_df)
        cat_cols = [
            c for c in model_df.columns
            if c in getattr(self, "_categorical_features_", set()) or not pd.api.types.is_numeric_dtype(model_df[c])
        ]
        if len(cat_cols) > 0:
            model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=False)

        for col in self._model_columns_:
            if col not in model_df.columns:
                model_df[col] = 0.0

        model_df = model_df[self._model_columns_].astype(float)
        return model_df.reset_index(drop=True), seg_df.reset_index(drop=True)

    def _resolve_categorical_features(self, X):
        if self.categorical_features is None:
            return set()

        resolved = set()
        columns = list(X.columns)
        for feature in self.categorical_features:
            if isinstance(feature, (int, np.integer)):
                if feature < 0 or feature >= len(columns):
                    raise ValueError(
                        f"categorical_features index {feature} is out of bounds for X with "
                        f"{len(columns)} features."
                    )
                resolved.add(columns[int(feature)])
                continue

            feature_name = str(feature)
            if feature_name not in X.columns:
                raise ValueError(
                    f"categorical_features contains unknown feature '{feature_name}'."
                )
            resolved.add(feature_name)

        return resolved

    def _apply_model_numeric_scaling_fit(self, model_df):
        if not self.scale_model_numeric:
            return

        if self.model_numeric_scaling != "minmax":
            raise ValueError("model_numeric_scaling deve ser 'minmax'.")

        numeric_columns = [
            column for column in model_df.columns
            if column not in self._categorical_features_
            and pd.api.types.is_numeric_dtype(model_df[column])
        ]

        for column in numeric_columns:
            values = pd.to_numeric(model_df[column], errors="coerce").astype(float)
            min_value = float(values.min())
            max_value = float(values.max())
            denom = max_value - min_value
            self._model_numeric_scalers_[column] = (min_value, max_value)
            if not np.isfinite(denom) or denom <= 0.0:
                model_df[column] = 0.0
            else:
                model_df[column] = (values - min_value) / denom

    def _apply_model_numeric_scaling_predict(self, model_df):
        if not self.scale_model_numeric:
            return

        for column, (min_value, max_value) in self._model_numeric_scalers_.items():
            if column not in model_df.columns:
                continue
            values = pd.to_numeric(model_df[column], errors="coerce").astype(float)
            denom = max_value - min_value
            if not np.isfinite(denom) or denom <= 0.0:
                model_df[column] = 0.0
            else:
                scaled = (values - min_value) / denom
                model_df[column] = np.clip(scaled, 0.0, 1.0)

    def _bin_numeric_series(self, s):
        x = s.astype(float).to_numpy()

        if self.numeric_binning == "quantile":
            qs = np.linspace(0, 1, self.n_numeric_bins + 1)[1:-1]
            edges = np.unique(np.quantile(x, qs))
        elif self.numeric_binning == "uniform":
            mn, mx = np.min(x), np.max(x)
            edges = np.unique(np.linspace(mn, mx, self.n_numeric_bins + 1)[1:-1])
        else:
            raise ValueError("numeric_binning deve ser 'quantile' ou 'uniform'.")

        binned = pd.Series(
            pd.cut(x, bins=[-np.inf] + list(edges) + [np.inf], include_lowest=True),
            index=s.index,
        )
        return binned, edges

    # --------------------------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------------------------

    def _default_logistic_estimator(self):
        return LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=self.random_state
        )

    def _build_screening_estimator(self):
        if self.screening_estimator is not None:
            return clone(self.screening_estimator)
        return self._default_logistic_estimator()

    def _build_node_estimator(self):
        if self.node_estimator is not None:
            return clone(self.node_estimator)
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return self._default_logistic_estimator()

    def _build_segment_estimator(self):
        if self.segment_estimator is not None:
            return clone(self.segment_estimator)
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return self._default_logistic_estimator()

    def _build_local_combiner_estimator(self):
        if self.local_combiner_estimator is not None:
            return clone(self.local_combiner_estimator)
        return LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=self.random_state
        )

    def _build_global_combiner_estimator(self):
        if self.global_combiner_estimator is not None:
            return clone(self.global_combiner_estimator)
        return LogisticRegression(
            solver="lbfgs",
            C=self.global_stacking_C,
            max_iter=5000,
            random_state=self.random_state
        )
    '''
    def _build_screening_estimator(self):
        if self.screening_estimator is None:
            return LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
                random_state=self.random_state
            )
        return clone(self.screening_estimator)
    '''
    def _fit_estimator_safe(self, estimator_template, X, y, sample_weight=None, context="model"):
        y = np.asarray(y).astype(int)
    
        if len(y) == 0:
            return ConstantModel(self.global_prior_)
    
        if len(np.unique(y)) < 2:
            return ConstantModel(float(np.mean(y)))
    
        try:
            model = clone(estimator_template)
            model, had_warning = self._fit_estimator_with_warning_control(
                estimator=model,
                X=X,
                y=y,
                sample_weight=sample_weight,
                context=context
            )
    
            if had_warning:
                self._log(3, f"[MODEL] {context} ajustado com ConvergenceWarning", indent=2)
    
            return model
    
        except Exception as e:
            if self.verbose >= 3:
                print(f"Erro em _fit_estimator_safe ({context}):", e)
            return ConstantModel(float(np.mean(y)))

    def _fit_node_model_safe(self, X, y, sample_weight=None):
        return self._fit_estimator_safe(
            estimator_template=self.node_estimator_,
            X=X,
            y=y,
            sample_weight=sample_weight,
            context="node_estimator"
        )
    
    def _fit_segment_model_safe(self, X, y, sample_weight=None):
        return self._fit_estimator_safe(
            estimator_template=self.segment_estimator_,
            X=X,
            y=y,
            sample_weight=sample_weight,
            context="segment_estimator"
        )

    # --------------------------------------------------------------------------
    # SPLITS / REGRAS
    # --------------------------------------------------------------------------

    def _generate_split_groups(self, series):
        values = sorted(series.astype(str).unique().tolist())
        if len(values) <= 1:
            return []

        if self.group_binned_numeric and series.name in getattr(self, "_seg_numeric_label_order_", {}):
            present = set(values)
            ordered_values = [
                value
                for value in self._seg_numeric_label_order_[series.name]
                if value in present
            ]
            groups = []
            for start in range(len(ordered_values)):
                for end in range(start + 1, len(ordered_values) + 1):
                    group = frozenset(ordered_values[start:end])
                    if 0 < len(group) < len(ordered_values):
                        groups.append(group)

            groups = _dedupe_complementary_groups(groups, ordered_values)
            order_index = {value: index for index, value in enumerate(ordered_values)}
            groups = sorted(
                groups,
                key=lambda group: (
                    len(group),
                    min(order_index[value] for value in group),
                    [order_index[value] for value in group],
                ),
            )
            return groups

        groups = [frozenset([v]) for v in values]
        if self.use_grouping:
            groups = list(set(groups + _powerset(values, max_size=self.max_group_size)))

        groups = [g for g in groups if 0 < len(g) < len(values)]
        groups = _dedupe_complementary_groups(groups, values)
        groups = sorted(groups, key=lambda g: (len(g), sorted(list(g))))
        return groups

    def _resolve_min_leaf(self, n):
        if isinstance(self.min_samples_leaf, float) and self.min_samples_leaf < 1:
            return max(1, int(math.ceil(n * self.min_samples_leaf)))
        return int(self.min_samples_leaf)

    def _can_split_node(self, y):
        min_leaf = self._resolve_min_leaf(len(y))
        return len(y) >= 2 * min_leaf and len(np.unique(y)) >= 2

    def _valid_masks(self, y, left_mask, right_mask):
        min_leaf = self._resolve_min_leaf(len(y))
        left_n = int(np.sum(left_mask))
        right_n = int(np.sum(right_mask))

        if left_n < min_leaf or right_n < min_leaf:
            return False

        y_left = y.iloc[left_mask] if isinstance(y, pd.Series) else np.asarray(y)[left_mask]
        y_right = y.iloc[right_mask] if isinstance(y, pd.Series) else np.asarray(y)[right_mask]

        if np.sum(y_left == 1) == 0 or np.sum(y_left == 0) == 0:
            return False
        if np.sum(y_right == 1) == 0 or np.sum(y_right == 0) == 0:
            return False
        return True

    # --------------------------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------------------------

    def _calculate_metrics(self, y_true, y_prob):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)

        if len(y_true) == 0:
            return {
                "lift": 0.0, "ks": 0.0, "precision": 0.0,
                "auc": 0.0, "error": 1.0, "rocmin": 1.0
            }

        order = np.argsort(-y_prob)
        y_sorted = y_true[order]

        n = len(y_sorted)
        n_pos = np.sum(y_sorted == 1)
        n_neg = np.sum(y_sorted == 0)

        top_n = max(1, int(np.ceil(n * self.top_rate)))
        top_y = y_sorted[:top_n]
        base_rate = np.mean(y_sorted)
        top_rate_pos = np.mean(top_y) if top_n > 0 else 0.0

        lift = top_rate_pos / base_rate if base_rate > 0 else 0.0
        precision = top_rate_pos

        if n_pos == 0 or n_neg == 0:
            ks = 0.0
            auc = 0.0
            rocmin = 1.0
        else:
            cum_pos = np.cumsum(y_sorted == 1) / n_pos
            cum_neg = np.cumsum(y_sorted == 0) / n_neg
            ks = float(np.max(np.abs(cum_pos - cum_neg)))
            auc = float(roc_auc_score(y_true, y_prob))
            rocmin = float(1.0 - auc)

        pred = (y_prob >= self.classification_threshold).astype(int)
        error = float(np.mean(pred != y_true))

        return {
            "lift": float(lift),
            "ks": float(ks),
            "precision": float(precision),
            "auc": float(auc),
            "error": float(error),
            "rocmin": float(rocmin)
        }

    def _objective_from_metrics(self, metrics):
        if self.metric == "lift":
            return metrics["lift"]
        if self.metric == "ks":
            return metrics["ks"]
        if self.metric == "precision":
            return metrics["precision"]
        if self.metric == "auc":
            return metrics["auc"]
        if self.metric == "error":
            return -metrics["error"]
        if self.metric == "rocmin":
            return -metrics["rocmin"]
        if self.metric == "combined":
            w_lift, w_ks = self.metric_weights
            return (w_lift * metrics["lift"]) + (w_ks * metrics["ks"])
        raise ValueError("metric inválida.")

    # --------------------------------------------------------------------------
    # TABELAS / RESUMO / PLOT
    # --------------------------------------------------------------------------

    def _collect_node_table(self, root):
        rows = []

        def walk(node):
            rows.append({
                "node_id": node.node_id,
                "depth": node.depth,
                "n": len(node.train_index),
                "is_leaf": node.is_leaf,
                "split_variable": node.split_variable,
                "split_group": None if node.split_group is None else _group_text(node.split_group),
                "split_gain": node.split_gain,
                "global_train_objective": self._objective_from_metrics(node.global_metrics_train) if node.global_metrics_train else None,
                "global_val_objective": self._objective_from_metrics(node.global_metrics_val) if node.global_metrics_val else None,
                "split_val_objective": node.split_metrics_val.get("objective"),
                "split_val_auc": node.split_metrics_val.get("auc"),
                "split_val_ks": node.split_metrics_val.get("ks"),
                "split_val_lift": node.split_metrics_val.get("lift"),
                "split_val_error": node.split_metrics_val.get("error"),
                "split_val_rocmin": node.split_metrics_val.get("rocmin")
            })
            if node.left_child is not None:
                walk(node.left_child)
            if node.right_child is not None:
                walk(node.right_child)

        walk(root)
        return pd.DataFrame(rows).sort_values(["depth", "node_id"]).reset_index(drop=True)

    def _collect_split_table(self, root):
        frames = []

        def walk(node):
            if node.split_candidates_table is not None and not node.split_candidates_table.empty:
                df = node.split_candidates_table.copy()
                df["node_id"] = node.node_id
                df["depth"] = node.depth
                frames.append(df)
            if node.left_child is not None:
                walk(node.left_child)
            if node.right_child is not None:
                walk(node.right_child)

        walk(root)
        if len(frames) == 0:
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True)

    def _build_summary_dict(self):
        if not hasattr(self, "estimator_metadata_"):
            self.estimator_metadata_ = {
                "screening_estimator_name": None,
                "node_estimator_name": None,
                "segment_estimator_name": None,
                "local_combiner_estimator_name": None,
                "global_combiner_estimator_name": None,
                "screening_estimator_repr": None,
                "node_estimator_repr": None,
                "segment_estimator_repr": None,
                "local_combiner_estimator_repr": None,
                "global_combiner_estimator_repr": None,
            }
        node_table = self.node_table_
        split_nodes = node_table[~node_table["is_leaf"]]

        used_variables = []
        if len(split_nodes) > 0:
            used_variables = split_nodes["split_variable"].dropna().tolist()

        summary = {
            "screening_variables_requested": self.screening_variables if self.screening_variables is not None else "ALL",
            "screening_variables_effective": self._get_allowed_screening_variables(self.seg_view_.columns),
            "n_observations_train": int(len(self.model_view_)),
            "n_original_variables": int(self.n_features_in_),
            "n_model_features_after_encoding": int(self.model_view_.shape[1]),
            "max_depth_reached": int(node_table["depth"].max()) if len(node_table) > 0 else 0,
            "n_nodes": int(len(node_table)),
            "n_internal_nodes": int((~node_table["is_leaf"]).sum()),
            "n_leaves": int((node_table["is_leaf"]).sum()),
            "used_variables_sequence": used_variables,
            "used_variables_unique": sorted(set(used_variables)),
            "screening_estimator_name": self.estimator_metadata_["screening_estimator_name"],
            "node_estimator_name": self.estimator_metadata_["node_estimator_name"],
            "segment_estimator_name": self.estimator_metadata_["segment_estimator_name"],
            "local_combiner_estimator_name": self.estimator_metadata_["local_combiner_estimator_name"],
            "global_combiner_estimator_name": self.estimator_metadata_["global_combiner_estimator_name"],
            "screening_estimator_repr": self.estimator_metadata_["screening_estimator_repr"],
            "node_estimator_repr": self.estimator_metadata_["node_estimator_repr"],
            "segment_estimator_repr": self.estimator_metadata_["segment_estimator_repr"],
            "local_combiner_estimator_repr": self.estimator_metadata_["local_combiner_estimator_repr"],
            "global_combiner_estimator_repr": self.estimator_metadata_["global_combiner_estimator_repr"],
            "screening_mode": self.screening_mode,
            "factorial_max_interaction_features": self.factorial_max_interaction_features,
            "factorial_feature_selector": self.factorial_feature_selector,
            "factorial_include_main_effects": self.factorial_include_main_effects,
            "factorial_drop_first": self.factorial_drop_first,
            "global_simple_auc": float(self.global_simple_metrics_["auc"]),
            "global_simple_ks": float(self.global_simple_metrics_["ks"]),
            "global_simple_lift": float(self.global_simple_metrics_["lift"]),
            "global_simple_precision": float(self.global_simple_metrics_["precision"]),
            "global_simple_error": float(self.global_simple_metrics_["error"]),
            "global_simple_rocmin": float(self.global_simple_metrics_["rocmin"]),
            "riskseg_global_stacking_auc": float(self.global_stacking_metrics_["auc"]),
            "riskseg_global_stacking_ks": float(self.global_stacking_metrics_["ks"]),
            "riskseg_global_stacking_lift": float(self.global_stacking_metrics_["lift"]),
            "riskseg_global_stacking_precision": float(self.global_stacking_metrics_["precision"]),
            "riskseg_global_stacking_error": float(self.global_stacking_metrics_["error"]),
            "riskseg_global_stacking_rocmin": float(self.global_stacking_metrics_["rocmin"]),
            "train_final_prediction_auc": float(self.train_prediction_summary_["auc"]),
            "train_final_prediction_ks": float(self.train_prediction_summary_["ks"]),
            "train_final_prediction_lift": float(self.train_prediction_summary_["lift"]),
            "train_final_prediction_precision": float(self.train_prediction_summary_["precision"]),
            "train_final_prediction_error": float(self.train_prediction_summary_["error"]),
            "train_final_prediction_rocmin": float(self.train_prediction_summary_["rocmin"]),
            "delta_auc_vs_global_simple": float(self.global_stacking_metrics_["auc"] - self.global_simple_metrics_["auc"]),
            "delta_ks_vs_global_simple": float(self.global_stacking_metrics_["ks"] - self.global_simple_metrics_["ks"]),
            "delta_lift_vs_global_simple": float(self.global_stacking_metrics_["lift"] - self.global_simple_metrics_["lift"]),
            "delta_precision_vs_global_simple": float(self.global_stacking_metrics_["precision"] - self.global_simple_metrics_["precision"]),
            "delta_error_vs_global_simple": float(self.global_stacking_metrics_["error"] - self.global_simple_metrics_["error"]),
            "delta_rocmin_vs_global_simple": float(self.global_stacking_metrics_["rocmin"] - self.global_simple_metrics_["rocmin"])
        }
        return summary

    def plot_model_tree(
        self,
        figsize=(24, 12),
        node_fontsize=10,
        edge_fontsize=9,
        title_fontsize=13,
        show_support=True,
        show_metrics=True,
        metric_decimals=3,
        left_color="#2E8B57",
        right_color="#C0392B",
        internal_facecolor="#D9EDF7",
        leaf_facecolor="#DFF0D8",
        edge_linewidth=2.0,
        box_pad=0.5
    ):
        check_is_fitted(self, "is_fitted_")

        positions = {}
        labels = {}
        edges = []
        leaf_counter = [0]

        def assign_positions(node, depth=0):
            if node.is_leaf:
                x = leaf_counter[0]
                leaf_counter[0] += 1
                positions[node.node_id] = (x, -depth)
            else:
                assign_positions(node.left_child, depth + 1)
                assign_positions(node.right_child, depth + 1)

                xl, _ = positions[node.left_child.node_id]
                xr, _ = positions[node.right_child.node_id]
                positions[node.node_id] = ((xl + xr) / 2.0, -depth)

                edges.append({
                    "parent": node.node_id,
                    "child": node.left_child.node_id,
                    "side": "left",
                    "rule": f"{node.split_variable} in {_group_text(node.split_group)}",
                    "support": node.left_support
                })
                edges.append({
                    "parent": node.node_id,
                    "child": node.right_child.node_id,
                    "side": "right",
                    "rule": f"{node.split_variable} not in {_group_text(node.split_group)}",
                    "support": node.right_support
                })

        assign_positions(self.root_)

        def build_labels(node):
            if node.is_leaf:
                lines = [
                    f"NODE {node.node_id} | LEAF",
                    f"depth={node.depth}",
                    f"n={len(node.train_index)}"
                ]
                if show_metrics:
                    lines.extend([
                        f"AUC={node.global_metrics_train.get('auc', np.nan):.{metric_decimals}f}",
                        f"KS={node.global_metrics_train.get('ks', np.nan):.{metric_decimals}f}",
                        f"Lift={node.global_metrics_train.get('lift', np.nan):.{metric_decimals}f}",
                        f"Error={node.global_metrics_train.get('error', np.nan):.{metric_decimals}f}"
                    ])
                txt = "\n".join(lines)
            else:
                lines = [
                    f"NODE {node.node_id}",
                    f"Split: {node.split_variable}",
                    f"Group: {_group_text(node.split_group)}",
                    f"gain={node.split_gain:.{metric_decimals}f}" if node.split_gain is not None else "gain=NA"
                ]
                if show_metrics:
                    lines.extend([
                        f"val_auc={node.split_metrics_val.get('auc', np.nan):.{metric_decimals}f}",
                        f"val_ks={node.split_metrics_val.get('ks', np.nan):.{metric_decimals}f}",
                        f"val_lift={node.split_metrics_val.get('lift', np.nan):.{metric_decimals}f}",
                        f"val_error={node.split_metrics_val.get('error', np.nan):.{metric_decimals}f}"
                    ])
                txt = "\n".join(lines)

            labels[node.node_id] = txt

            if node.left_child is not None:
                build_labels(node.left_child)
            if node.right_child is not None:
                build_labels(node.right_child)

        build_labels(self.root_)

        fig, ax = plt.subplots(figsize=figsize)

        title = (
            "RISKSEG - Árvore de Modelos\n"
            f"Global simples | AUC={self.global_simple_metrics_['auc']:.{metric_decimals}f}, KS={self.global_simple_metrics_['ks']:.{metric_decimals}f}, Lift={self.global_simple_metrics_['lift']:.{metric_decimals}f}, Error={self.global_simple_metrics_['error']:.{metric_decimals}f}"
            "\n"
            f"RISKSEG global stacking | AUC={self.global_stacking_metrics_['auc']:.{metric_decimals}f}, KS={self.global_stacking_metrics_['ks']:.{metric_decimals}f}, Lift={self.global_stacking_metrics_['lift']:.{metric_decimals}f}, Error={self.global_stacking_metrics_['error']:.{metric_decimals}f}"
        )
        ax.set_title(title, fontsize=title_fontsize, pad=20)

        for edge in edges:
            parent = edge["parent"]
            child = edge["child"]
            side = edge["side"]
            rule = edge["rule"]
            support = edge["support"]

            x1, y1 = positions[parent]
            x2, y2 = positions[child]

            color = left_color if side == "left" else right_color
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=edge_linewidth, zorder=1)

            xm = (x1 + x2) / 2.0
            ym = (y1 + y2) / 2.0

            if show_support:
                edge_text = (
                    f"{'LEFT' if side == 'left' else 'RIGHT'}\n"
                    f"{rule}\n"
                    f"n={support.get('n', 'NA')} | pos={support.get('positives', 'NA')} | neg={support.get('negatives', 'NA')}"
                )
            else:
                edge_text = f"{'LEFT' if side == 'left' else 'RIGHT'}\n{rule}"

            dx = -0.15 if side == "left" else 0.15
            dy = 0.12

            ax.text(
                xm + dx,
                ym + dy,
                edge_text,
                ha="center",
                va="center",
                fontsize=edge_fontsize,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.95),
                zorder=3
            )

        for node_id, (x, y) in positions.items():
            node_row = self.node_table_[self.node_table_["node_id"] == node_id].iloc[0]
            is_leaf = bool(node_row["is_leaf"])
            facecolor = leaf_facecolor if is_leaf else internal_facecolor

            ax.text(
                x,
                y,
                labels[node_id],
                ha="center",
                va="center",
                fontsize=node_fontsize,
                bbox=dict(boxstyle=f"round,pad={box_pad}", facecolor=facecolor, edgecolor="black", linewidth=1.5),
                zorder=4
            )

        legend_y = max(y for _, y in positions.values()) + 0.6
        ax.text(min(x for x, _ in positions.values()), legend_y, "LEFT = condição verdadeira (segmento selecionado)", color=left_color, fontsize=edge_fontsize + 1, ha="left", va="center")
        ax.text(min(x for x, _ in positions.values()), legend_y - 0.25, "RIGHT = complemento da condição", color=right_color, fontsize=edge_fontsize + 1, ha="left", va="center")

        ax.axis("off")
        plt.tight_layout()
        return fig, ax
