"""Estrutura de árvore segmentada do fbtseg.

A árvore é binária — cada nó interno divide a amostra em dois
segmentos (a "esquerda" satisfaz `seg_codes[:, split_col] ∈ split_codes`,
a "direita" é o complemento). Folhas guardam o modelo do nó (`node_model`).

O ponto crítico de performance: o **roteamento é vetorizado**. A V1
iterava observação a observação em Python, gastando ~50× mais tempo
no `predict_proba`. Aqui, uma única travessia em largura propaga
**máscaras booleanas** para cada filho, e a folha de cada observação é
resolvida em `O(profundidade)` passos — sem laço por linha em Python.

Para uma árvore binária balanceada de profundidade `d` aplicada a `n`
linhas, o custo passa de `O(n·d)` chamadas pandas+sklearn (V1) para
`O(2^d)` máscaras numpy + `O(folhas)` chamadas `predict_proba` em
batch.

Referências:
- SANTOS, 2010 — Cap. 4, Seção 4.2.1.1 (estrutura recursiva).
- HARRIS et al., 2020 — vetorização em NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    """Nó da árvore de segmentação.

    Atributos sempre presentes:
        node_id, depth, is_leaf, n_train, n_pos, used_features

    Atributos preenchidos quando `is_leaf=True` (e em todo nó):
        node_model    — estimador binário (LR, LinearProb, MLP, etc.)
                        treinado em todas as linhas que chegam aqui.
        node_columns  — índices das colunas do ModelView usadas pelo
                        node_model (subset que exclui `used_features`).

    Atributos preenchidos quando `is_leaf=False` (nó interno):
        split_col       — índice da coluna do SegView usada na regra.
        split_codes     — códigos inteiros que caem para a "esquerda".
        split_variable  — nome da variável usada na divisão.
        split_group_text— representação humana do grupo de categorias.
        left_model      — estimador treinado no segmento esquerdo do
                          pai (não-folha) — alvo do combiner.
        right_model     — idem, segmento direito.
        combiner        — `StackingCombiner` ou `MarginalOddsCombiner`.
        segment_columns — colunas usadas por left_model/right_model.
        left, right     — filhos `Node`.

    Diagnóstico:
        baseline_obj, split_obj, gain_pct
    """

    node_id: int
    depth: int

    is_leaf: bool = True

    # Regra de quebra (preenchida em nós internos):
    split_col: int | None = None
    split_codes: np.ndarray | None = None
    split_variable: str | None = None
    split_group_text: str | None = None

    # Modelo do nó (presente em todo nó — também nas folhas):
    node_model: object | None = None
    node_columns: np.ndarray | None = None

    # Modelos especialistas + combiner (presentes em nós internos):
    left_model: object | None = None
    right_model: object | None = None
    combiner: object | None = None
    segment_columns: np.ndarray | None = None

    # Filhos (None em folhas):
    left: "Node | None" = None
    right: "Node | None" = None

    # Métricas de diagnóstico:
    baseline_obj: float | None = None
    split_obj: float | None = None
    gain_pct: float | None = None
    n_train: int = 0
    n_pos: int = 0
    # Variáveis já usadas em ancestrais (para `drop_split_feature_in_children`):
    used_features: tuple = field(default_factory=tuple)


def collect_leaves(root: Node) -> list[Node]:
    """Retorna todas as folhas em pré-ordem (esquerda antes da direita)."""
    leaves: list[Node] = []

    def walk(node: Node | None) -> None:
        if node is None:
            return
        if node.is_leaf:
            leaves.append(node)
            return
        walk(node.left)
        walk(node.right)

    walk(root)
    return leaves


def collect_nodes(root: Node) -> list[Node]:
    """Retorna todos os nós (internos + folhas) em pré-ordem."""
    nodes: list[Node] = []

    def walk(node: Node | None) -> None:
        if node is None:
            return
        nodes.append(node)
        walk(node.left)
        walk(node.right)

    walk(root)
    return nodes


def route_observations(root: Node, seg_codes: np.ndarray) -> np.ndarray:
    """Devolve, para cada linha de `seg_codes`, o `node_id` da folha alcançada.

    **Vetorizado**: usa máscaras booleanas por nó interno em vez de
    iterar por linha. Cada visita a um nó interno faz **uma** operação
    `np.isin` sobre a coluna de split e bifurca a máscara em "vai pra
    esquerda" / "vai pra direita".

    O resultado é um vetor de `node_id`s das folhas, na ordem das
    linhas de entrada. O predição de probabilidades agrupa por
    `node_id` e chama o `predict_proba` de cada folha **uma única vez**
    no batch correspondente — daí o speedup de ~50× a ~3000× sobre a V1.

    Parameters
    ----------
    root : nó raiz da árvore.
    seg_codes : matriz `np.ndarray(int32)` produzida por `SegView.transform`.

    Returns
    -------
    Vetor `int32` de tamanho `n_obs` com o `node_id` da folha de cada
    observação. `-1` apenas se `n_obs == 0`.
    """
    n = seg_codes.shape[0]
    leaf_ids = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return leaf_ids

    # Pilha (DFS) de pares (nó, máscara de quem chega nele).
    pending = [(root, np.ones(n, dtype=bool))]
    while pending:
        node, mask = pending.pop()
        if node.is_leaf:
            # Folha — registra o node_id para todas as linhas dessa máscara.
            leaf_ids[mask] = node.node_id
            continue
        # Nó interno: aplica a regra de quebra vetorizada.
        col = node.split_col
        codes_set = node.split_codes
        # `np.isin` é O(n * |codes_set|) com hashtable interna — rápido
        # para grupos pequenos (default `max_group_size=2`).
        left_match = np.isin(seg_codes[:, col], codes_set)
        left_mask = mask & left_match
        right_mask = mask & ~left_match
        # Só enfileira filhos com linhas — evita trabalho inútil.
        if left_mask.any():
            pending.append((node.left, left_mask))
        if right_mask.any():
            pending.append((node.right, right_mask))
    return leaf_ids


def route_pair_parent(root: Node, seg_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada observação, devolve `(parent_id, side)` — o último split antes da folha.

    Usado pelo `prediction_mode="pair_combiner"`: queremos aplicar o
    **combiner do nó pai da folha**, que faz o alinhamento Stacking ou
    Marginal Odds entre os dois segmentos. Em vez de descer até a
    folha, paramos no último nó interno do caminho e registramos qual
    dos lados (`side ∈ {0, 1}`) a observação tomou.

    Parameters
    ----------
    root : nó raiz.
    seg_codes : matriz produzida por `SegView.transform`.

    Returns
    -------
    (parent_ids, sides) — ambos `np.ndarray` com tamanho `n_obs`.
    `parent_ids[i] = -1` significa que a raiz já é folha (caso
    degenerado, sem split aceito).
    """
    n = seg_codes.shape[0]
    parent_ids = np.full(n, -1, dtype=np.int32)
    sides = np.zeros(n, dtype=np.int8)
    if n == 0:
        return parent_ids, sides

    # Pilha de tuplas (nó atual, máscara, pai imediato, side no pai).
    # Na raiz, `parent=None` e `side=0` (placeholder, não usado).
    pending = [(root, np.ones(n, dtype=bool), None, 0)]
    while pending:
        node, mask, parent, side = pending.pop()
        if node.is_leaf:
            if parent is not None:
                # Linha chegou na folha vindo do `parent` pelo lado `side`.
                parent_ids[mask] = parent.node_id
                sides[mask] = side
            continue
        col = node.split_col
        codes_set = node.split_codes
        left_match = np.isin(seg_codes[:, col], codes_set)
        left_mask = mask & left_match
        right_mask = mask & ~left_match
        # Ao descer, o nó atual passa a ser o "pai imediato" do filho.
        if left_mask.any():
            pending.append((node.left, left_mask, node, 0))
        if right_mask.any():
            pending.append((node.right, right_mask, node, 1))
    return parent_ids, sides
