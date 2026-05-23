"""Visões de dados do fbtseg.

A tese (Cap. 4, Seção 4.2.1.5) descreve um aspecto sutil mas crucial
do FBTSeg: variáveis numéricas precisam ser **categorizadas** (em
quartis, por exemplo) **só para a etapa de segmentação**, enquanto os
**modelos preditivos** continuam recebendo as variáveis numéricas
**na sua forma original** (normalizadas).

Isso evita duas patologias comuns:
- categorizar e re-codar como dummies destrói a relação contínua que
  uma regressão logística aproveita;
- usar a variável contínua direto no critério de quebra força a árvore
  a achar pontos de corte em vez de **conjuntos de categorias**, que é
  o ponto forte do FBTSeg.

Para suportar isso o pacote mantém **duas visões** do mesmo dataset:

* `SegView` — visão para **segmentação**. Codifica todas as colunas em
  inteiros (categorias / faixas numéricas via quartis) para roteamento
  vetorizado da árvore. Trabalha em `numpy` puro depois do `fit`, sem
  overhead de pandas no caminho de predição.

* `ModelView` — visão para os **modelos preditivos** dos nós/segmentos.
  Numéricas escaladas em [0, 1] (alinhado ao protocolo dos artigos
  ICAI/ICTAI 2012, Seção 3 de cada um) e categóricas em dummies
  one-hot. Mantém rastreio de quais colunas dummies derivaram de cada
  variável original — isso permite **descartar** todas as colunas
  derivadas da variável de split nos modelos descendentes
  (`drop_split_feature_in_children=True`), implementando a recomendação
  da tese de "eliminar o efeito da variável segmentadora nos modelos
  seguintes" (Cap. 4, parágrafo final da Seção 4.3).

Referências (`docs/references.md`):
- SANTOS, 2010 — Cap. 4, Seção 4.2.1.5 (categorização das numéricas).
- SANTOS & BARROS, 2012 (ICAI) — Seção 3 (protocolo de preprocessamento).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# Tipo inteiro usado em todas as colunas codificadas. `int32` é
# suficiente — ninguém terá 2^31 categorias num modelo de risco.
_RESERVED_INT_DTYPE = np.int32

# Código reservado para "categoria não vista no fit". O roteamento da
# árvore trata isso como "sem match" (vai para o segmento direito por
# padrão, já que `np.isin` retorna False).
_RESERVED_NA_CODE = np.int32(-1)


@dataclass
class SegView:
    """Codifica cada coluna em códigos inteiros para roteamento vetorizado.

    Para colunas **numéricas** (não explicitamente categóricas):
        - Discretiza em `n_numeric_bins` faixas usando quartis (q1, q2, q3
          para n_numeric_bins=4 — o default do paper).
        - `transform` aplica `np.searchsorted` sobre essas bordas,
          mapeando cada valor para o índice da faixa.

    Para colunas **categóricas** (ou marcadas em `categorical_features`):
        - Constrói `{valor_str: índice_inteiro}` no fit.
        - `transform` aplica esse map e atribui `-1` a valores não vistos.

    O resultado de `transform` é uma matriz `np.int32` de forma
    `(n_obs, n_cols)` que serve de entrada para:
        - `tree.route_observations` (roteamento por máscaras booleanas);
        - screening fatorial (uma vez por nó, sem reprocessar pandas).
    """

    n_numeric_bins: int = 4
    categorical_features: tuple = ()

    # Atributos preenchidos no fit:
    columns_: list = field(default_factory=list)
    is_numeric_: dict = field(default_factory=dict)
    category_index_: dict = field(default_factory=dict)
    numeric_edges_: dict = field(default_factory=dict)
    n_categories_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "SegView":
        """Aprende, para cada coluna, o mapa de codificação inteira."""
        self.columns_ = list(X.columns)
        for col in self.columns_:
            s = X[col]
            forced_cat = col in self.categorical_features
            if pd.api.types.is_numeric_dtype(s) and not forced_cat:
                # Numérica: discretiza em quartis (ou n_numeric_bins faixas).
                self.is_numeric_[col] = True
                edges = self._quantile_edges(s.to_numpy(dtype=float))
                self.numeric_edges_[col] = edges
                # nº de faixas = nº de bordas internas + 1.
                self.n_categories_[col] = max(1, len(edges) + 1)
            else:
                # Categórica: enumera valores únicos como strings.
                self.is_numeric_[col] = False
                values = pd.Index(s.astype(str).unique())
                self.category_index_[col] = {v: i for i, v in enumerate(values)}
                self.n_categories_[col] = len(values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica a codificação aprendida, devolvendo `np.ndarray(int32)`."""
        n = len(X)
        out = np.empty((n, len(self.columns_)), dtype=_RESERVED_INT_DTYPE)
        for j, col in enumerate(self.columns_):
            s = X[col]
            if self.is_numeric_[col]:
                edges = self.numeric_edges_[col]
                if edges.size == 0:
                    # Numérica degenerada (constante): cai tudo na faixa 0.
                    out[:, j] = 0
                else:
                    # `searchsorted(side="right")` devolve o índice da faixa.
                    # Faixa 0 = abaixo da 1ª borda, faixa k = acima da última.
                    out[:, j] = np.searchsorted(edges, s.to_numpy(dtype=float), side="right")
            else:
                idx = self.category_index_[col]
                arr = s.astype(str).map(idx)
                if arr.isna().any():
                    # Categoria nova no predict (não vista no fit) → -1.
                    arr = arr.fillna(_RESERVED_NA_CODE)
                out[:, j] = arr.to_numpy(dtype=_RESERVED_INT_DTYPE)
        return out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def column_index(self, name: str) -> int:
        """Índice (na ordem de `columns_`) da coluna com esse nome."""
        return self.columns_.index(name)

    def categories_present(self, X_codes: np.ndarray, col_idx: int) -> np.ndarray:
        """Códigos únicos presentes na coluna, ignorando o NA-code (-1)."""
        col = X_codes[:, col_idx]
        return np.unique(col[col >= 0])

    def is_numeric_column(self, name: str) -> bool:
        return self.is_numeric_[name]

    def _quantile_edges(self, values: np.ndarray) -> np.ndarray:
        """Bordas internas dos `n_numeric_bins` quartis (sem -inf/+inf)."""
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.array([], dtype=float)
        n_bins = self.n_numeric_bins
        if n_bins <= 1:
            return np.array([], dtype=float)
        # qs = (1/n, 2/n, ..., (n-1)/n). Para n=4: 0.25, 0.5, 0.75.
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        # `unique` colapsa quartis degenerados (ex.: distribuições muito
        # concentradas onde dois quartis coincidem).
        edges = np.unique(np.quantile(values, qs))
        return edges


@dataclass
class ModelView:
    """Prepara matriz numérica densa para os modelos preditivos.

    O `ModelView` tem dois jobs:

    1. Numericas: escala em [0, 1] (min-max) — exatamente o
       preprocessamento adotado nos papers ICAI/ICTAI 2012 ("all numeric
       variables were normalized with continuous values in the [0, 1]
       interval", ICAI Seção 3).

    2. Categóricas: one-hot encoding (uma coluna por valor único). O
       paper usa "binary codifications" (ICAI Seção 3) — o
       equivalente moderno é `OneHotEncoder(handle_unknown='ignore')`.

    Além disso, mantém o mapa **coluna_dummy -> variável_original** via
    `source_feature_`. Isso é o que viabiliza
    `drop_split_feature_in_children=True`: ao crescer um filho da
    árvore, o estimador pede `columns_excluding([var_de_split])` e
    recebe só as colunas que NÃO derivam da variável usada — então o
    modelo filho não tem como reaprender a regra de divisão.
    """

    scale_numeric: bool = True
    categorical_features: tuple = ()

    # Atributos preenchidos no fit:
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
        """Aprende min/max das numéricas e o conjunto de categorias."""
        self.columns_ = list(X.columns)
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        # Passo 1: detecta tipo coluna a coluna e guarda estatísticas.
        for col in self.columns_:
            s = X[col]
            forced_cat = col in self.categorical_features
            if pd.api.types.is_numeric_dtype(s) and not forced_cat:
                self.numeric_cols_.append(col)
                arr = s.to_numpy(dtype=float)
                arr = arr[~np.isnan(arr)]
                lo = float(arr.min()) if arr.size else 0.0
                hi = float(arr.max()) if arr.size else 1.0
                # `rng=1.0` se a coluna for constante — evita /0 no scale.
                rng = hi - lo if hi > lo else 1.0
                self.numeric_min_[col] = lo
                self.numeric_range_[col] = rng
            else:
                self.categorical_cols_.append(col)
                vals = pd.Index(s.astype(str).unique())
                self.category_values_[col] = list(vals)

        # Passo 2: constrói a lista final de colunas de saída na ordem
        # [numéricas..., dummies das categóricas...] e o mapa inverso
        # dummy -> feature original.
        out_columns = []
        source_feature = {}
        feature_to_columns: dict = {col: [] for col in self.columns_}

        for col in self.numeric_cols_:
            out_columns.append(col)
            source_feature[col] = col  # numérica = uma coluna só
            feature_to_columns[col].append(len(out_columns) - 1)

        for col in self.categorical_cols_:
            for v in self.category_values_[col]:
                name = f"{col}__{v}"  # convenção: "<feature>__<valor>"
                out_columns.append(name)
                source_feature[name] = col
                feature_to_columns[col].append(len(out_columns) - 1)

        self.out_columns_ = out_columns
        self.source_feature_ = source_feature
        self.feature_to_columns_ = feature_to_columns
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica scaling + one-hot, devolvendo `np.ndarray(float64)`."""
        n = len(X)
        out = np.zeros((n, len(self.out_columns_)), dtype=np.float64)
        j = 0
        # Numéricas primeiro, na ordem aprendida.
        for col in self.numeric_cols_:
            arr = X[col].to_numpy(dtype=float)
            # NaN -> min observado (imputação simples, suficiente para LR).
            arr = np.where(np.isnan(arr), self.numeric_min_[col], arr)
            if self.scale_numeric:
                arr = (arr - self.numeric_min_[col]) / self.numeric_range_[col]
                arr = np.clip(arr, 0.0, 1.0)  # safety se algo extrapola
            out[:, j] = arr
            j += 1
        # Categóricas: uma coluna 0/1 por valor.
        for col in self.categorical_cols_:
            s = X[col].astype(str).to_numpy()
            for v in self.category_values_[col]:
                out[:, j] = (s == v).astype(np.float64)
                j += 1
        return out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def columns_excluding(self, features: Iterable[str]) -> np.ndarray:
        """Índices das colunas que NÃO derivam de `features`.

        Usado pelo estimador para implementar
        `drop_split_feature_in_children=True`: quando um nó cresce
        seus filhos, ele passa `columns_excluding([var_de_split])` para
        o `_fit_clone`, e o modelo filho só vê o subset que não inclui
        as dummies da variável já usada. Implementa a recomendação da
        tese de "eliminar o efeito da variável segmentadora nos
        modelos seguintes" (Cap. 4, Seção 4.3).
        """
        drop = set(features)
        keep = [i for name, i in zip(self.out_columns_, range(len(self.out_columns_)))
                if self.source_feature_[name] not in drop]
        return np.asarray(keep, dtype=int)

    def all_columns(self) -> np.ndarray:
        """Todos os índices de coluna — atalho para nós raiz / sem drop."""
        return np.arange(len(self.out_columns_), dtype=int)
