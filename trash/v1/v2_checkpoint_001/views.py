"""Visões de dados do RiskSeg V2.

* `SegView`: codifica todas as colunas em inteiros (categorias / faixas
  numéricas) para roteamento vetorizado da árvore. Trabalha em `numpy`
  puro depois do `fit`.

* `ModelView`: prepara matriz numérica para os modelos preditivos —
  numéricas escaladas em [0,1] (alinhado ao protocolo dos artigos) e
  categóricas em dummies one-hot. Mantém rastreio de quais colunas
  derivaram de cada variável original, para permitir o drop da variável
  de split nos modelos descendentes (eliminação do efeito conforme a tese).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


_RESERVED_INT_DTYPE = np.int32
_RESERVED_NA_CODE = np.int32(-1)


@dataclass
class SegView:
    n_numeric_bins: int = 4
    categorical_features: tuple = ()

    columns_: list = field(default_factory=list)
    is_numeric_: dict = field(default_factory=dict)
    category_index_: dict = field(default_factory=dict)
    numeric_edges_: dict = field(default_factory=dict)
    n_categories_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "SegView":
        self.columns_ = list(X.columns)
        for col in self.columns_:
            s = X[col]
            forced_cat = col in self.categorical_features
            if pd.api.types.is_numeric_dtype(s) and not forced_cat:
                self.is_numeric_[col] = True
                edges = self._quantile_edges(s.to_numpy(dtype=float))
                self.numeric_edges_[col] = edges
                self.n_categories_[col] = max(1, len(edges) + 1)
            else:
                self.is_numeric_[col] = False
                values = pd.Index(s.astype(str).unique())
                self.category_index_[col] = {v: i for i, v in enumerate(values)}
                self.n_categories_[col] = len(values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        out = np.empty((n, len(self.columns_)), dtype=_RESERVED_INT_DTYPE)
        for j, col in enumerate(self.columns_):
            s = X[col]
            if self.is_numeric_[col]:
                edges = self.numeric_edges_[col]
                if edges.size == 0:
                    out[:, j] = 0
                else:
                    out[:, j] = np.searchsorted(edges, s.to_numpy(dtype=float), side="right")
            else:
                idx = self.category_index_[col]
                arr = s.astype(str).map(idx)
                if arr.isna().any():
                    arr = arr.fillna(_RESERVED_NA_CODE)
                out[:, j] = arr.to_numpy(dtype=_RESERVED_INT_DTYPE)
        return out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def column_index(self, name: str) -> int:
        return self.columns_.index(name)

    def categories_present(self, X_codes: np.ndarray, col_idx: int) -> np.ndarray:
        col = X_codes[:, col_idx]
        return np.unique(col[col >= 0])

    def is_numeric_column(self, name: str) -> bool:
        return self.is_numeric_[name]

    def _quantile_edges(self, values: np.ndarray) -> np.ndarray:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.array([], dtype=float)
        n_bins = self.n_numeric_bins
        if n_bins <= 1:
            return np.array([], dtype=float)
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(values, qs))
        return edges


@dataclass
class ModelView:
    scale_numeric: bool = True
    categorical_features: tuple = ()

    columns_: list = field(default_factory=list)
    numeric_cols_: list = field(default_factory=list)
    categorical_cols_: list = field(default_factory=list)
    numeric_min_: dict = field(default_factory=dict)
    numeric_range_: dict = field(default_factory=dict)
    category_values_: dict = field(default_factory=dict)
    out_columns_: list = field(default_factory=list)
    source_feature_: dict = field(default_factory=dict)
    feature_to_columns_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "ModelView":
        self.columns_ = list(X.columns)
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        for col in self.columns_:
            s = X[col]
            forced_cat = col in self.categorical_features
            if pd.api.types.is_numeric_dtype(s) and not forced_cat:
                self.numeric_cols_.append(col)
                arr = s.to_numpy(dtype=float)
                arr = arr[~np.isnan(arr)]
                lo = float(arr.min()) if arr.size else 0.0
                hi = float(arr.max()) if arr.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                self.numeric_min_[col] = lo
                self.numeric_range_[col] = rng
            else:
                self.categorical_cols_.append(col)
                vals = pd.Index(s.astype(str).unique())
                self.category_values_[col] = list(vals)

        out_columns = []
        source_feature = {}
        feature_to_columns: dict = {col: [] for col in self.columns_}

        for col in self.numeric_cols_:
            out_columns.append(col)
            source_feature[col] = col
            feature_to_columns[col].append(len(out_columns) - 1)

        for col in self.categorical_cols_:
            for v in self.category_values_[col]:
                name = f"{col}__{v}"
                out_columns.append(name)
                source_feature[name] = col
                feature_to_columns[col].append(len(out_columns) - 1)

        self.out_columns_ = out_columns
        self.source_feature_ = source_feature
        self.feature_to_columns_ = feature_to_columns
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, len(self.out_columns_)), dtype=np.float64)
        j = 0
        for col in self.numeric_cols_:
            arr = X[col].to_numpy(dtype=float)
            arr = np.where(np.isnan(arr), self.numeric_min_[col], arr)
            if self.scale_numeric:
                arr = (arr - self.numeric_min_[col]) / self.numeric_range_[col]
                arr = np.clip(arr, 0.0, 1.0)
            out[:, j] = arr
            j += 1
        for col in self.categorical_cols_:
            s = X[col].astype(str).to_numpy()
            for v in self.category_values_[col]:
                out[:, j] = (s == v).astype(np.float64)
                j += 1
        return out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def columns_excluding(self, features: Iterable[str]) -> np.ndarray:
        """Índices das colunas a manter quando se descartam variáveis originais."""
        drop = set(features)
        keep = [i for name, i in zip(self.out_columns_, range(len(self.out_columns_)))
                if self.source_feature_[name] not in drop]
        return np.asarray(keep, dtype=int)

    def all_columns(self) -> np.ndarray:
        return np.arange(len(self.out_columns_), dtype=int)
