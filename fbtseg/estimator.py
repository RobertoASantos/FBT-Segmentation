"""Estimador principal do fbtseg.

Implementação fiel ao Capítulo 4 da tese de Roberto Angelo Fernandes
Santos (UFPE, 2010) e aos artigos ICAI 2012 e ICTAI 2012 (Santos &
Barros). Resolve os desvios identificados na revisão da V1 (`riskseg`):

1. **Modelos descendentes treinam sem a variável de split**
   (`drop_split_feature_in_children=True`). A tese é explícita:
   "A segmentação elimina o efeito da variável (ou da categoria) nos
   modelos seguintes" (Cap. 4, Seção 4.3).
2. **Marginal Odds usa segmento de referência real** (Thomas, Edelman
   & Crook, 2002) — implementado em `combiners.py`.
3. **`grouping_features` é lista por variável** (= `rUsaBlocos` da
   tese, Seção 4.2.1.6); `max_group_size` (= `rQtdeBlocos`).
4. **`min_gain_pct` / `max_loss_pct` em percentual do baseline** —
   reproduz o critério "ganho mínimo / perda máxima aceitável" da
   tese (parâmetro `B` em Seção 4.2.1.6).
5. **`prediction_mode="leaf"` é o default** (sem `global_stacking`
   forçado, que era um desvio da V1).
6. **Baseline `LogisticRegression(penalty=None)` por default** —
   aproxima `PROC LOGISTIC` do SAS Enterprise Miner usado no paper.
7. **Predição totalmente vetorizada por folha** (vide `tree.py`).
8. **Screening fatorial vetorizado em numpy** (vide `_screen` abaixo).
9. **Métrica `odds_ratio` disponível** (Cap. 4, Seção 4.2.1.6).

Referências (`docs/references.md`):
- SANTOS, 2010 — Tese, Cap. 4 inteiro.
- SANTOS & BARROS, 2012a (ICAI) — Algoritmo 1 e protocolo experimental.
- SANTOS & BARROS, 2012b (ICTAI) — Datasets sintéticos.
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


# Iter máximas para os solvers lbfgs — alto o suficiente para convergir
# em segmentos típicos, baixo o suficiente para não pendurar fits em
# segmentos degenerados.
_DEFAULT_MAX_ITER = 2000


def _default_base_estimator(random_state: int = 42) -> LogisticRegression:
    """Logística sem regularização — aproxima `PROC LOGISTIC` do SAS.

    O paper ICAI 2012 usou `PROC LOGISTIC` do SAS Enterprise Miner,
    que estima por máxima verossimilhança sem regularização. A V1
    deste pacote usava `C=1.0` (L2 forte) por default, o que sozinho
    explicava a maior parte do desvio numérico vs paper (vide
    `docs/benchmark_final.md`).
    """
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=_DEFAULT_MAX_ITER,
        random_state=random_state,
    )


def _fit_clone(estimator, X, y, sample_weight=None):
    """Treina uma cópia do estimador (via `sklearn.clone`) com defesas.

    Defesas:
    - Se `y` tem só uma classe, devolve um `_ConstantClassifier` em vez
      de tentar ajustar (LR/MLP do sklearn lançariam exceção).
    - Se o estimador não aceita `sample_weight` (ex.: alguns MLPs),
      cai no `fit(X, y)` sem peso em vez de propagar `TypeError`.
    - Silencia `ConvergenceWarning` aqui — o estimador chamador já
      controla isso uma vez no `fit`.
    """
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if classes.size < 2:
        return _ConstantClassifier(float(y.mean()) if y.size else 0.5)
    m = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            if sample_weight is not None:
                m.fit(X, y, sample_weight=sample_weight)
            else:
                m.fit(X, y)
        except TypeError:
            # Estimador não aceita sample_weight → tenta sem.
            m.fit(X, y)
    return m


@dataclass
class _ConstantClassifier:
    """Substituto trivial quando o segmento só tem uma classe.

    Devolve sempre `p_` para classe positiva. Mantém a API mínima
    (`predict_proba`, `predict`) que o estimador principal precisa.
    """
    p_: float

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full((n, 2), [1 - self.p_, self.p_], dtype=float)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RiskSegV2(ClassifierMixin, BaseEstimator):
    """fbtseg — segmentação binária recursiva por modelo fatorial (RiskSeg/FBTSeg).

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
        - `'leaf'`: predição da folha alcançada (default, mais fiel).
        - `'pair_combiner'`: aplica o combiner do nó pai à folha.
        - `'cascade'`: combina o combiner de cada nível ao longo do
          caminho até a folha (média uniforme).
        - `'global_stacking'`: meta-modelo global sobre as folhas,
          treinado com predições **out-of-fold** (sem leakage interno).
    - `screening_variables` (`P`): lista de variáveis candidatas a
      split. Se `None`, usa todas as colunas.
    - `classification_threshold`: corte para `predict` (default 0.5).
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
        screening_variables: tuple | list | None = None,
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
        classification_threshold: float = 0.5,
        global_stacking_n_splits: int = 5,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.base_estimator = base_estimator
        self.screening_estimator = screening_estimator
        self.categorical_features = categorical_features
        self.screening_variables = screening_variables
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
        self.classification_threshold = classification_threshold
        self.global_stacking_n_splits = global_stacking_n_splits
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

        if sample_weight is None:
            sw_arr = None
        else:
            sw_arr = np.asarray(sample_weight, dtype=float)
            if sw_arr.size != y_arr.size:
                raise ValueError("sample_weight com tamanho diferente de y.")

        self._base = self.base_estimator or _default_base_estimator(self.random_state)
        self._screening = self.screening_estimator or self._base

        if self.screening_variables is not None:
            invalid = set(self.screening_variables) - set(self.seg_view_.columns_)
            if invalid:
                raise ValueError(
                    f"screening_variables com nomes desconhecidos: {sorted(invalid)}"
                )

        self._node_counter = 0
        self._log(1, f"[FIT] n={len(y_arr)} features={X.shape[1]} model_cols={model_mat.shape[1]}")

        self.root_ = self._grow(
            idx=np.arange(len(y_arr), dtype=np.int64),
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y_arr,
            sample_weight=sw_arr,
            depth=0,
            used_features=tuple(),
        )

        self.leaves_ = collect_leaves(self.root_)
        self.nodes_ = collect_nodes(self.root_)
        self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]

        if self.prediction_mode == "global_stacking":
            self.global_stacking_model_ = self._fit_global_stacking_oof(
                seg_codes=seg_codes,
                model_mat=model_mat,
                y=y_arr,
                sample_weight=sw_arr,
            )
        else:
            self.global_stacking_model_ = None

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
        if self.prediction_mode == "cascade":
            return self._predict_proba_cascade(seg_codes, model_mat)
        if self.prediction_mode == "global_stacking":
            return self._predict_proba_global_stacking(seg_codes, model_mat)
        raise ValueError(f"prediction_mode '{self.prediction_mode}' não suportado.")

    def predict(self, X):
        proba = self.predict_proba(X)
        encoded = (proba[:, 1] >= self.classification_threshold).astype(int)
        return self.classes_[encoded]

    def score(self, X, y):
        y_enc = self._encode_y(y)
        return -np.mean((self.predict(X) != self.classes_[y_enc]).astype(float)) + 1.0

    # =========================================================================
    # Crescimento da árvore
    # =========================================================================

    def _grow(self, idx, seg_codes, model_mat, y, sample_weight, depth, used_features):
        """Cresce recursivamente um nó da árvore.

        Implementa o **Algoritmo 1** do paper ICAI 2012 (Santos & Barros,
        2012a) e o Cap. 4 da tese:

        1. Treina o modelo do nó com todas as linhas que chegam aqui
           (`node_model`).
        2. Verifica critérios de parada `B` (profundidade, mínimo por
           segmento, classe única).
        3. Divide treino interno em treino/validação (`rTamValidação`).
        4. Treina baseline na validação (=`rMod_0` do Algoritmo 1).
        5. **Screening fatorial** (`_screen`): para cada variável
           candidata, ajusta `LR(dummies + main + interações) → y` e
           ordena por métrica `D` (= "primeiro laço" do Algoritmo 1).
        6. **Busca de categoria/grupo** (`_best_split_for_variable`):
           para o top-`k` variáveis (`rQtdeVarTeste`), testa todas as
           categorias e grupos como regra de divisão (= "segundo laço"
           do Algoritmo 1).
        7. **Aceitação**: ganho >= `min_gain_pct` ou perda <=
           `max_loss_pct` (parâmetro `B` da tese).
        8. **Recursão** nos dois filhos. Se
           `drop_split_feature_in_children=True`, a variável usada é
           propagada em `used_features` e os filhos não a verão.
        """
        node_id = self._next_node_id()
        used_features = tuple(used_features)

        # === Passo 1: modelo do nó ===
        # Treinado com TODAS as linhas que chegam aqui (não usa o split
        # interno treino/val — esse split é só para escolher o split).
        # `cols_to_use` exclui as variáveis já usadas em ancestrais
        # se `drop_split_feature_in_children=True`.
        cols_to_use = self._cols_for_features(used_features)
        node_model = _fit_clone(
            self._base,
            model_mat[idx][:, cols_to_use],
            y[idx],
            sample_weight[idx] if sample_weight is not None else None,
        )

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

        # === Passo 2: critérios de parada (parâmetro B da tese) ===
        min_leaf = self._resolve_min_leaf(idx.size)
        if (
            depth >= self.max_depth
            or idx.size < 2 * min_leaf      # nem cabe split mínimo
            or np.unique(y[idx]).size < 2   # alvo constante
        ):
            self._log(2, f"[NODE {node_id}] folha por parada (depth={depth}, n={idx.size})")
            return node

        # === Passo 3: split interno treino/validação (rTamValidação) ===
        # Estratificado por padrão para preservar prior das classes.
        tr_rel, va_rel = self._internal_train_val_split(y[idx])
        tr_abs = idx[tr_rel]
        va_abs = idx[va_rel]
        sw_tr = sample_weight[tr_abs] if sample_weight is not None else None

        # === Passo 4: baseline na validação (rMod_0 do Algoritmo 1) ===
        # Modelo treinado SÓ no treino interno e avaliado na validação.
        # Esta é a régua que vai ser usada para comparar com os splits.
        baseline_model = _fit_clone(
            self._base, model_mat[tr_abs][:, cols_to_use], y[tr_abs], sw_tr
        )
        baseline_proba = baseline_model.predict_proba(model_mat[va_abs][:, cols_to_use])[:, 1]
        baseline_obj = metric_score(self.metric, y[va_abs], baseline_proba)
        node.baseline_obj = baseline_obj

        # === Passo 5: screening fatorial (primeiro laço do Algoritmo 1) ===
        # Ranqueia variáveis candidatas pela performance de uma regressão
        # com interações dummy(var) × X_model.
        screening = self._screen(
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y,
            tr_abs=tr_abs,
            va_abs=va_abs,
            cols_to_use=cols_to_use,
            used_features=used_features,
            sample_weight_tr=sw_tr,
        )
        if not screening:
            self._log(2, f"[NODE {node_id}] screening vazio → folha")
            return node

        # Top-k variáveis (`rQtdeVarTeste`); default k=1 como no
        # Algoritmo 1 original.
        top_vars = [v for v, _ in screening[: max(1, self.top_k_variables)]]
        self._log(
            3,
            f"[NODE {node_id}] screening top-{len(top_vars)}: "
            + ", ".join(f"{v}={obj:.4f}" for v, obj in screening[:5]),
        )

        # === Passo 6: busca de categoria/grupo (segundo laço) ===
        # Para cada variável candidata, testa todas as quebras possíveis
        # e devolve a melhor. Mantém um único `best` global entre as
        # variáveis do top-k.
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
                sample_weight=sample_weight,
            )

        if best is None:
            self._log(2, f"[NODE {node_id}] sem split candidato válido → folha")
            return node

        # === Passo 7: aceitação (parâmetro B — ganho mín / perda máx) ===
        # `gain_pct` é o ganho percentual sobre o baseline. Pode ser
        # negativo (perda). A tese permite aceitar pequenas perdas para
        # escapar de máximos locais (Cap. 4, Seção 4.2.1.6).
        denom = abs(baseline_obj) if baseline_obj != 0 else 1.0
        gain_pct = (best["obj"] - baseline_obj) / denom
        node.gain_pct = gain_pct
        node.split_obj = best["obj"]

        if gain_pct >= 0:
            # Ganho positivo: exige pelo menos `min_gain_pct`.
            accept = gain_pct >= self.min_gain_pct
        else:
            # Perda: aceita se for menor que `max_loss_pct`.
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
            sample_weight=sample_weight,
            depth=depth + 1,
            used_features=new_used,
        )
        node.right = self._grow(
            idx=right_idx,
            seg_codes=seg_codes,
            model_mat=model_mat,
            y=y,
            sample_weight=sample_weight,
            depth=depth + 1,
            used_features=new_used,
        )
        return node

    # =========================================================================
    # Screening fatorial
    # =========================================================================

    def _screen(self, seg_codes, model_mat, y, tr_abs, va_abs, cols_to_use, used_features, sample_weight_tr=None):
        """Screening fatorial — primeiro laço do Algoritmo 1 do paper ICAI.

        Para cada variável candidata `Xi`, ajusta uma regressão:

            y ~ dummies(Xi) + main_effects(X_model) + interações(dummies(Xi), X_model)

        e avalia na validação interna pela métrica `D`. O ranking por
        `D` indica quais variáveis "interagem mais fortemente" com as
        demais — exatamente o critério que a tese (Cap. 4, equações
        4.1.1-4.1.6) demonstra ser equivalente à equivalência
        fatorial-segmentação.

        Implementação **totalmente vetorizada** em numpy:

        - `dummies` = `(codes[:, None] == present[None, :])` é
          one-hot inline sem precisar de `OneHotEncoder`.
        - Interações = `(dummies[:, :, None] * main[:, None, :]).reshape`
          produz `k * p` colunas num único array op sem laço Python.

        O parâmetro `factorial_max_interaction_features` limita as
        colunas usadas como segundo fator nas interações (top-N por
        variância). Isso evita explosão para `n_features > 100`.
        Defina `None` para o desenho fatorial pleno da tese.
        """
        results = []

        # Lista de candidatas: respeitando `screening_variables` (P) e
        # excluindo variáveis já usadas em ancestrais (se drop).
        if self.screening_variables is not None:
            allowed = set(self.screening_variables)
            candidate_vars = [c for c in self.seg_view_.columns_ if c in allowed and c not in used_features]
        else:
            candidate_vars = [c for c in self.seg_view_.columns_ if c not in used_features]

        # `main_train` / `main_val`: matriz X_model do nó (após drop).
        # Esses são os "main effects" — variáveis que entram no fatorial
        # tanto como termo principal quanto como segundo fator das
        # interações.
        main_train = model_mat[tr_abs][:, cols_to_use]
        main_val = model_mat[va_abs][:, cols_to_use]

        # Pré-seleção de colunas para INTERAÇÕES (não afeta main effects).
        # Default: top-N por variância. Variáveis quase-constantes contribuem
        # pouco para o fatorial e custam muito (n_cols^2 explode).
        max_inter = self.factorial_max_interaction_features
        if max_inter is None or max_inter >= main_train.shape[1]:
            inter_cols = np.arange(main_train.shape[1])
        else:
            variances = main_train.var(axis=0)
            inter_cols = np.argsort(-variances)[:max_inter]
        sub_train = main_train[:, inter_cols] if inter_cols.size else main_train[:, :0]
        sub_val = main_val[:, inter_cols] if inter_cols.size else main_val[:, :0]

        # === Loop por variável candidata (paralelizável, mas síncrono aqui) ===
        for var_name in candidate_vars:
            var_col = self.seg_view_.column_index(var_name)
            codes_tr = seg_codes[tr_abs, var_col]
            codes_va = seg_codes[va_abs, var_col]
            # Categorias presentes APENAS no treino interno (evita usar
            # categorias só vistas na validação).
            present = np.unique(codes_tr[codes_tr >= 0])
            if present.size < 2:
                # 1 categoria só => não tem o que segmentar.
                continue

            # Dummies one-hot: shape (n_treino, k_categorias).
            dummies_tr = (codes_tr[:, None] == present[None, :]).astype(np.float64)
            dummies_va = (codes_va[:, None] == present[None, :]).astype(np.float64)

            # Interações dummy × X_model: shape (n, k * p_inter).
            # `(n, k, 1) * (n, 1, p) -> (n, k, p)` por broadcasting,
            # depois `reshape` achatando para (n, k * p).
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

            # Matriz fatorial completa para este `var_name`:
            #   [dummies(var)  |  main_effects(X_model)  |  interações]
            Z_tr = np.concatenate([dummies_tr, main_train, inter_tr], axis=1)
            Z_va = np.concatenate([dummies_va, main_val, inter_va], axis=1)

            # Ajusta UM modelo (default: LR sem regularização) e mede `D`.
            model = _fit_clone(self._screening, Z_tr, y[tr_abs], sample_weight_tr)
            p_va = model.predict_proba(Z_va)[:, 1]
            obj = metric_score(self.metric, y[va_abs], p_va)
            results.append((var_name, obj))

        # Ranking decrescente por `D` — `top_k_variables` primeiras
        # serão tentadas na busca de categoria/grupo.
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
        sample_weight=None,
    ):
        """Segundo laço do Algoritmo 1: melhor categoria/grupo para `var_name`.

        Para cada grupo candidato (categorias individuais e/ou grupos
        contíguos / blocos, dependendo de `grouping_features` e
        `ordered_groups_for_numeric`):

        1. Separa treino/validação interna em "esquerda" (cai no grupo)
           e "direita" (complemento).
        2. Valida que cada lado tem `min_samples_leaf` linhas e ambas
           as classes no treino interno.
        3. Treina `left_model` e `right_model` (especialistas) **sem
           a variável de split** (drop é o default).
        4. Combina os escores via `Stacking` ou `MarginalOdds` —
           combiner é treinado no treino interno e aplicado na validação.
        5. Mede `D` na validação. O melhor grupo (entre todos
           candidatos para esta variável e o `current_best` global)
           sobrevive.

        Mantém `current_best` entre chamadas para que múltiplas
        variáveis do top-k possam disputar pelo melhor split global.
        """
        var_col = self.seg_view_.column_index(var_name)
        node_codes = seg_codes[idx, var_col]
        codes_present = np.unique(node_codes[node_codes >= 0])
        if codes_present.size < 2:
            # 1 categoria só => sem split possível.
            return current_best

        # Enumera grupos candidatos. Para numéricas com
        # `ordered_groups_for_numeric=True`, só blocos contíguos são
        # considerados (preserva ordem natural).
        is_numeric = self.seg_view_.is_numeric_column(var_name)
        groups = self._enumerate_groups(var_name, codes_present, is_numeric)
        if not groups:
            return current_best

        # Colunas do `ModelView` que os modelos especialistas verão.
        # **Crucial:** descarta as colunas derivadas da própria variável
        # de split (`var_name`). Implementa a recomendação da tese.
        seg_cols = self.model_view_.columns_excluding(
            list(used_features) + ([var_name] if self.drop_split_feature_in_children else [])
        )
        if seg_cols.size == 0:
            # Caso degenerado: sem colunas. Mantém o conjunto completo
            # para não quebrar o fit (modelo vai virar constante).
            seg_cols = self.model_view_.all_columns()

        codes_tr = seg_codes[tr_abs, var_col]
        codes_va = seg_codes[va_abs, var_col]
        min_leaf = self._resolve_min_leaf(idx.size)

        best = current_best

        # === Loop por grupo candidato ===
        for group_codes in groups:
            # Máscaras booleanas: lado "esquerdo" = pertence ao grupo.
            left_mask_tr = np.isin(codes_tr, group_codes)
            right_mask_tr = ~left_mask_tr
            left_mask_va = np.isin(codes_va, group_codes)
            right_mask_va = ~left_mask_va

            # Filtro 1: cada lado precisa ter mass suficiente.
            if (
                left_mask_tr.sum() < min_leaf
                or right_mask_tr.sum() < min_leaf
                or left_mask_va.sum() < 1
                or right_mask_va.sum() < 1
            ):
                continue

            # Filtro 2: cada lado precisa ter ambas as classes no treino
            # (caso contrário a LR não tem o que aprender).
            y_tr_left = y[tr_abs][left_mask_tr]
            y_tr_right = y[tr_abs][right_mask_tr]
            if np.unique(y_tr_left).size < 2 or np.unique(y_tr_right).size < 2:
                continue

            # Slicing dos `ModelView` para cada lado, em treino e validação.
            X_tr_left = model_mat[tr_abs[left_mask_tr]][:, seg_cols]
            X_tr_right = model_mat[tr_abs[right_mask_tr]][:, seg_cols]
            X_va_left = model_mat[va_abs[left_mask_va]][:, seg_cols]
            X_va_right = model_mat[va_abs[right_mask_va]][:, seg_cols]

            # Pesos por amostra, se fornecidos.
            if sample_weight is not None:
                sw_tr_left = sample_weight[tr_abs[left_mask_tr]]
                sw_tr_right = sample_weight[tr_abs[right_mask_tr]]
            else:
                sw_tr_left = None
                sw_tr_right = None

            # Treina os dois especialistas (= modelos das equações 4.2-4.5
            # da tese, Cap. 4).
            left_model = _fit_clone(self._base, X_tr_left, y_tr_left, sw_tr_left)
            right_model = _fit_clone(self._base, X_tr_right, y_tr_right, sw_tr_right)

            # Coletа escores no TREINO interno para ajustar o combiner.
            # Cada lado preenche só sua máscara — o resto fica 0.
            score_tr_left = np.zeros(tr_abs.size, dtype=float)
            score_tr_right = np.zeros(tr_abs.size, dtype=float)
            score_tr_left[left_mask_tr] = left_model.predict_proba(X_tr_left)[:, 1]
            score_tr_right[right_mask_tr] = right_model.predict_proba(X_tr_right)[:, 1]
            membership_tr = right_mask_tr.astype(int)  # 0=left, 1=right

            # Treina o combiner (Stacking ou Marginal Odds) no TREINO interno.
            combiner = build_combiner(self.combiner_method, random_state=self.random_state)
            combiner.fit(
                score_left=score_tr_left,
                score_right=score_tr_right,
                membership=membership_tr,
                y=y[tr_abs],
            )

            # Avalia o split + combiner na VALIDAÇÃO interna.
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
        """Modo `'leaf'`: predição vetorizada por folha (default).

        Implementação 50-3000× mais rápida que a V1:

        1. `route_observations` resolve em uma travessia o `leaf_id`
           de cada uma das `n` linhas (`O(n * depth)` máscaras numpy).
        2. Agrupamos as linhas por `leaf_id` e chamamos
           `leaf.node_model.predict_proba(X[mask])` **uma vez por folha**
           — em vez de `n` vezes uma a uma como na V1.

        Cada folha usa seu próprio subset de colunas (`leaf.node_columns`),
        que pode ter excluído variáveis usadas em ancestrais.
        """
        n = seg_codes.shape[0]
        leaf_ids = route_observations(self.root_, seg_codes)
        # Inicializa com 0.5/0.5 — só substituído onde tem folha.
        out = np.full((n, 2), [0.5, 0.5], dtype=float)
        leaf_by_id = {leaf.node_id: leaf for leaf in self.leaves_}
        for leaf_id in np.unique(leaf_ids):
            if leaf_id < 0:
                # Linha não rateada (caso degenerado de árvore vazia).
                continue
            mask = leaf_ids == leaf_id
            leaf = leaf_by_id[int(leaf_id)]
            X = model_mat[mask][:, leaf.node_columns]
            # UMA chamada predict_proba para todas as linhas da folha.
            out[mask] = leaf.node_model.predict_proba(X)
        return out

    def _predict_proba_pair(self, seg_codes, model_mat):
        """Modo `'pair_combiner'`: aplica o combiner do nó-pai da folha.

        Em vez de usar o `node_model` da folha, aplicamos o `combiner`
        treinado no pai imediato — que faz alinhamento Stacking ou
        Marginal Odds entre os dois segmentos. Pode ser mais robusto
        para escala de escores em árvores profundas.

        Para cada linha:
        - encontra o pai imediato da folha (`route_pair_parent`);
        - obtém o `score` do segmento correspondente (`left_model` se
          a linha caiu na esquerda, `right_model` na direita);
        - passa para `combiner.predict_proba` que devolve `p` alinhado.

        Linhas onde a raiz é folha (sem split aceito) caem no caminho
        do `node_model` da raiz.
        """
        n = seg_codes.shape[0]
        parent_ids, sides = route_pair_parent(self.root_, seg_codes)
        out = np.full((n, 2), [0.5, 0.5], dtype=float)
        node_by_id = {node.node_id: node for node in self.nodes_}
        for parent_id in np.unique(parent_ids):
            if parent_id < 0:
                # Linhas que não têm pai — raiz já é folha.
                mask = parent_ids < 0
                X = model_mat[mask][:, self.root_.node_columns]
                out[mask] = self.root_.node_model.predict_proba(X)
                continue
            parent = node_by_id[int(parent_id)]
            seg_cols = parent.segment_columns
            mask = parent_ids == parent_id
            local_sides = sides[mask]
            X = model_mat[mask][:, seg_cols]
            # Cada lado preenche só sua máscara — coerente com o que o
            # combiner espera (Z = [score_seg_pertencente, 0]).
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

    def _predict_proba_cascade(self, seg_codes, model_mat):
        """Combinação dos combiners encontrados ao longo do caminho até a folha.

        Para cada observação, soma a probabilidade de cada nó interno
        no caminho (via combiner do nó) e da folha, dividindo pelo nº
        de etapas — média uniforme das previsões alinhadas.
        """
        n = seg_codes.shape[0]
        sum_p = np.zeros(n, dtype=float)
        count = np.zeros(n, dtype=int)

        def walk(node, mask):
            if not mask.any():
                return
            if node.is_leaf:
                X = model_mat[mask][:, node.node_columns]
                p = node.node_model.predict_proba(X)[:, 1]
                sum_p[mask] += p
                count[mask] += 1
                return

            left_match = np.isin(seg_codes[:, node.split_col], node.split_codes)
            left_mask = mask & left_match
            right_mask = mask & ~left_match

            seg_cols = node.segment_columns
            score_left = np.zeros(int(mask.sum()), dtype=float)
            score_right = np.zeros(int(mask.sum()), dtype=float)
            local_idx_left = np.where(left_mask[mask])[0]
            local_idx_right = np.where(right_mask[mask])[0]

            if local_idx_left.size:
                score_left[local_idx_left] = node.left_model.predict_proba(
                    model_mat[left_mask][:, seg_cols]
                )[:, 1]
            if local_idx_right.size:
                score_right[local_idx_right] = node.right_model.predict_proba(
                    model_mat[right_mask][:, seg_cols]
                )[:, 1]
            membership = np.zeros(int(mask.sum()), dtype=int)
            membership[local_idx_right] = 1
            p = node.combiner.predict_proba(
                score_left=score_left,
                score_right=score_right,
                membership=membership,
            )[:, 1]
            sum_p[mask] += p
            count[mask] += 1

            walk(node.left, left_mask)
            walk(node.right, right_mask)

        walk(self.root_, np.ones(n, dtype=bool))
        count = np.where(count == 0, 1, count)
        p_final = sum_p / count
        return np.column_stack([1 - p_final, p_final])

    def _predict_proba_global_stacking(self, seg_codes, model_mat):
        if self.global_stacking_model_ is None:
            raise RuntimeError("global_stacking_model_ não foi treinado.")
        Z = self._build_global_leaf_matrix(seg_codes, model_mat)
        return self.global_stacking_model_.predict_proba(Z)

    def _build_global_leaf_matrix(self, seg_codes, model_mat) -> np.ndarray:
        """Z[i, j] = score da folha j para a observação i (zero se i não cai em j)."""
        n = seg_codes.shape[0]
        leaves = self.leaves_
        Z = np.zeros((n, len(leaves)), dtype=float)
        leaf_ids = route_observations(self.root_, seg_codes)
        for j, leaf in enumerate(leaves):
            mask = leaf_ids == leaf.node_id
            if mask.any():
                X = model_mat[mask][:, leaf.node_columns]
                Z[mask, j] = leaf.node_model.predict_proba(X)[:, 1]
        return Z

    def _fit_global_stacking_oof(self, seg_codes, model_mat, y, sample_weight):
        """Treina meta-modelo global sobre folhas com predições out-of-fold.

        Cada folha tem seu modelo refeito em folds; a coluna `j` do
        meta-input recebe a predição da folha `j` para as linhas de
        validação do fold. Assim, o treino do meta-modelo não vê
        predições in-sample das folhas (evita o leakage da V1).
        """
        from sklearn.model_selection import StratifiedKFold

        n = y.size
        leaves = self.leaves_
        n_leaves = len(leaves)
        Z = np.zeros((n, n_leaves), dtype=float)
        leaf_ids = route_observations(self.root_, seg_codes)
        leaf_pos = {leaf.node_id: j for j, leaf in enumerate(leaves)}

        n_splits = max(2, min(self.global_stacking_n_splits, np.bincount(y).min()))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for fold_idx, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            for leaf in leaves:
                leaf_tr = tr[leaf_ids[tr] == leaf.node_id]
                leaf_va = va[leaf_ids[va] == leaf.node_id]
                if leaf_tr.size < 2 or np.unique(y[leaf_tr]).size < 2:
                    Z[leaf_va, leaf_pos[leaf.node_id]] = float(y[leaf_tr].mean()) if leaf_tr.size else 0.5
                    continue
                sw = sample_weight[leaf_tr] if sample_weight is not None else None
                m = _fit_clone(self._base, model_mat[leaf_tr][:, leaf.node_columns], y[leaf_tr], sw)
                if leaf_va.size:
                    Z[leaf_va, leaf_pos[leaf.node_id]] = m.predict_proba(
                        model_mat[leaf_va][:, leaf.node_columns]
                    )[:, 1]

        meta = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=_DEFAULT_MAX_ITER,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if sample_weight is not None:
                meta.fit(Z, y, sample_weight=sample_weight)
            else:
                meta.fit(Z, y)
        return meta

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

    def get_summary(self) -> dict:
        """Resumo agregado da árvore (mapeado à terminologia da tese)."""
        check_is_fitted(self, "is_fitted_")
        used_seq = [n.split_variable for n in self.nodes_ if not n.is_leaf]
        return {
            "n_features": int(self.n_features_in_),
            "n_nodes": len(self.nodes_),
            "n_internal_nodes": int(sum(1 for n in self.nodes_ if not n.is_leaf)),
            "n_leaves": len(self.leaves_),
            "max_depth_reached": int(max(n.depth for n in self.nodes_)),
            "used_variables_unique": sorted(set(v for v in used_seq if v is not None)),
            "used_variables_sequence": [v for v in used_seq if v is not None],
            "prediction_mode": self.prediction_mode,
            "combiner_method": self.combiner_method,
            "metric": self.metric,
            "drop_split_feature_in_children": self.drop_split_feature_in_children,
            "classes": list(self.classes_),
        }

    def plot_model_tree(self, max_chars: int = 80) -> str:
        """Renderiza a árvore em texto ASCII (estilo `tree`).

        Caracteres puro ASCII para evitar problema de encoding em
        terminais Windows (cp1252).
        """
        check_is_fitted(self, "is_fitted_")
        lines: list[str] = []

        def fmt_node(node, prefix, is_tail):
            marker = "`-- " if is_tail else "|-- "
            if node.is_leaf:
                desc = (
                    f"[LEAF] id={node.node_id} d={node.depth} "
                    f"n={node.n_train} pos={node.n_pos}"
                )
            else:
                gain = node.gain_pct if node.gain_pct is not None else float("nan")
                desc = (
                    f"[SPLIT] id={node.node_id} d={node.depth} "
                    f"var={node.split_variable} group={node.split_group_text} "
                    f"gain_pct={gain:.3f}"
                )
            if len(desc) > max_chars:
                desc = desc[: max_chars - 3] + "..."
            lines.append(prefix + marker + desc)
            new_prefix = prefix + ("    " if is_tail else "|   ")
            if not node.is_leaf:
                fmt_node(node.left, new_prefix, is_tail=False)
                fmt_node(node.right, new_prefix, is_tail=True)

        root = self.root_
        if root.is_leaf:
            lines.append(f"[LEAF] id={root.node_id} d=0 n={root.n_train} pos={root.n_pos}")
        else:
            gain = root.gain_pct if root.gain_pct is not None else float("nan")
            lines.append(
                f"[ROOT] id={root.node_id} var={root.split_variable} "
                f"group={root.split_group_text} gain_pct={gain:.3f}"
            )
            fmt_node(root.left, "", is_tail=False)
            fmt_node(root.right, "", is_tail=True)
        return "\n".join(lines)


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
