"""Estrutura de árvore segmentada do RiskSeg V2.

Roteamento e coleta são vetorizados: dado o `SegView` codificado, uma
única travessia em largura propaga máscaras booleanas para cada filho,
e a folha de cada observação é resolvida em `O(profundidade)` passos
sem laço por linha em Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    node_id: int
    depth: int

    is_leaf: bool = True

    split_col: int | None = None
    split_codes: np.ndarray | None = None
    split_variable: str | None = None
    split_group_text: str | None = None

    node_model: object | None = None
    node_columns: np.ndarray | None = None

    left_model: object | None = None
    right_model: object | None = None
    combiner: object | None = None
    segment_columns: np.ndarray | None = None

    left: "Node | None" = None
    right: "Node | None" = None

    baseline_obj: float | None = None
    split_obj: float | None = None
    gain_pct: float | None = None
    n_train: int = 0
    n_pos: int = 0
    used_features: tuple = field(default_factory=tuple)


def collect_leaves(root: Node) -> list[Node]:
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

    Vetorizado: usa máscaras booleanas por nó interno em vez de iterar por linha.
    """
    n = seg_codes.shape[0]
    leaf_ids = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return leaf_ids

    pending = [(root, np.ones(n, dtype=bool))]
    while pending:
        node, mask = pending.pop()
        if node.is_leaf:
            leaf_ids[mask] = node.node_id
            continue
        col = node.split_col
        codes_set = node.split_codes
        left_match = np.isin(seg_codes[:, col], codes_set)
        left_mask = mask & left_match
        right_mask = mask & ~left_match
        if left_mask.any():
            pending.append((node.left, left_mask))
        if right_mask.any():
            pending.append((node.right, right_mask))
    return leaf_ids


def route_pair_parent(root: Node, seg_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada observação, encontra o nó interno mais profundo cujos filhos
    são todos folhas (ou um dos filhos é folha). Retorna `(parent_id, side)`
    onde `side==0` significa "esquerda" e `side==1` significa "direita".

    Permite usar o combiner do nó-pai para a predição da folha.
    """
    n = seg_codes.shape[0]
    parent_ids = np.full(n, -1, dtype=np.int32)
    sides = np.zeros(n, dtype=np.int8)
    if n == 0:
        return parent_ids, sides

    pending = [(root, np.ones(n, dtype=bool), None, 0)]
    while pending:
        node, mask, parent, side = pending.pop()
        if node.is_leaf:
            if parent is not None:
                parent_ids[mask] = parent.node_id
                sides[mask] = side
            continue
        col = node.split_col
        codes_set = node.split_codes
        left_match = np.isin(seg_codes[:, col], codes_set)
        left_mask = mask & left_match
        right_mask = mask & ~left_match
        if left_mask.any():
            pending.append((node.left, left_mask, node, 0))
        if right_mask.any():
            pending.append((node.right, right_mask, node, 1))
    return parent_ids, sides
