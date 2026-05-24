"""Visualização da árvore de segmentação do FBTSeg.

Exporta duas funções:

- ``plot_model_tree(model)``  — texto ASCII, sem dependências extras.
- ``plot_tree(model, ...)``   — gráfico matplotlib com caixas e arestas.

Uso::

    from fbtseg import FBTSeg
    from fbtseg.plot import plot_tree, plot_model_tree

    model = FBTSeg().fit(X, y)

    # Texto
    print(plot_model_tree(model))

    # Visual
    fig = plot_tree(model, figsize=(14, 7))
    fig.savefig("arvore.png", dpi=150, bbox_inches="tight")
    fig.show()

O matplotlib é uma dependência **opcional** — se não estiver instalado,
``plot_tree`` levanta ``ImportError`` com instrução de instalação.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# --------------------------------------------------------------------------- #
# Texto (sem dependências)                                                     #
# --------------------------------------------------------------------------- #

def plot_model_tree(model, max_chars: int = 80) -> str:
    """Renderiza a árvore em texto ASCII.

    Equivalente a ``model.plot_model_tree()`` — mantido aqui para uso
    funcional sem instanciar o estimador.
    """
    return model.plot_model_tree(max_chars=max_chars)


# --------------------------------------------------------------------------- #
# Visual (matplotlib)                                                          #
# --------------------------------------------------------------------------- #

def plot_tree(
    model,
    figsize: tuple[float, float] | None = None,
    ax=None,
    node_width: float = 3.2,
    node_height: float = 0.9,
    h_gap: float = 0.5,
    v_gap: float = 1.8,
    fontsize: int | None = None,
    color_split: str = "#4C72B0",
    color_leaf: str = "#55A868",
    color_edge: str = "#555555",
    title: str | None = None,
):
    """Renderiza a árvore de segmentação graficamente usando matplotlib.

    Parameters
    ----------
    model : FBTSeg treinado.
    figsize : tamanho da figura (largura, altura) em polegadas.
        Se ``None``, calculado automaticamente pelo número de folhas e
        profundidade da árvore.
    ax : eixo matplotlib existente (cria figura nova se None).
    node_width, node_height : dimensões das caixas dos nós.
    h_gap : espaço horizontal mínimo entre caixas irmãs.
    v_gap : espaço vertical entre níveis da árvore.
    fontsize : tamanho da fonte dentro das caixas. Se ``None``, calculado
        adaptativamente — árvores maiores recebem fonte menor.
    color_split : cor das caixas de nós internos (splits).
    color_leaf : cor das caixas de folhas.
    color_edge : cor das arestas.
    title : título do gráfico (usa métrica + preset se None).

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        raise ImportError(
            "matplotlib é necessário para plot_tree(). "
            "Instale com:  pip install matplotlib"
        ) from e

    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(model, "is_fitted_")

    root      = model.root_
    n_leaves  = len(model.leaves_)
    max_depth = max(n.depth for n in model.nodes_)

    # ------------------------------------------------------------------ #
    # Fontsize adaptativo: árvores maiores recebem fonte menor             #
    # ------------------------------------------------------------------ #
    if fontsize is None:
        # 8 pt para ≤4 folhas, reduz 1 pt a cada 2 folhas extras, mín. 5
        fontsize = max(5, 8 - max(0, (n_leaves - 4) // 2))

    # ------------------------------------------------------------------ #
    # 1. Layout: atribui posições (x, y) a cada nó                        #
    # ------------------------------------------------------------------ #

    positions: dict[int, tuple[float, float]] = {}
    leaf_counter = [0]

    def assign_x(node, depth: int) -> float:
        """Retorna o x central do nó (recursivo)."""
        if node.is_leaf:
            x = leaf_counter[0] * (node_width + h_gap)
            leaf_counter[0] += 1
            positions[node.node_id] = (x, -depth * v_gap)
            return x
        x_left = assign_x(node.left, depth + 1)
        x_right = assign_x(node.right, depth + 1)
        x = (x_left + x_right) / 2
        positions[node.node_id] = (x, -depth * v_gap)
        return x

    assign_x(root, 0)

    # ------------------------------------------------------------------ #
    # 2. Figura                                                            #
    # ------------------------------------------------------------------ #

    if ax is None:
        if figsize is None:
            # Largura proporcional às folhas, altura proporcional à profundidade
            fig_w = max(10.0, n_leaves * (node_width + h_gap) * 0.65)
            fig_h = max(5.0, (max_depth + 1) * v_gap * 0.55)
            figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_axis_off()
    ax.set_aspect("equal")

    # Margem automática
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - node_width, max(xs) + node_width)
    ax.set_ylim(min(ys) - node_height * 2, max(ys) + node_height * 2)

    # ------------------------------------------------------------------ #
    # 3. Arestas                                                           #
    # ------------------------------------------------------------------ #

    def draw_edges(node):
        if node.is_leaf:
            return
        x_p, y_p = positions[node.node_id]
        for child, label in [(node.left, "SIM"), (node.right, "NÃO")]:
            x_c, y_c = positions[child.node_id]
            ax.plot(
                [x_p, x_c],
                [y_p - node_height / 2, y_c + node_height / 2],
                color=color_edge, lw=1.2, zorder=1,
            )
            # Rótulo SIM/NÃO no meio da aresta
            mx, my = (x_p + x_c) / 2, (y_p - node_height / 2 + y_c + node_height / 2) / 2
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=fontsize - 1, color=color_edge,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
            draw_edges(child)

    draw_edges(root)

    # ------------------------------------------------------------------ #
    # 4. Caixas dos nós                                                    #
    # ------------------------------------------------------------------ #

    def node_label(node) -> list[str]:
        if not node.is_leaf:
            gain = f"{node.gain_pct:.3f}" if node.gain_pct is not None else "?"
            grp = node.split_group_text or "?"
            # Trunca grupo longo
            if len(grp) > 22:
                grp = grp[:20] + "…"
            return [
                f"SPLIT  |  var: {node.split_variable}",
                f"grupo: {grp}",
                f"ganho: {gain}  |  n={node.n_train}",
            ]
        else:
            base = node.n_pos / node.n_train if node.n_train > 0 else 0
            return [
                f"FOLHA  id={node.node_id}",
                f"n={node.n_train}  pos={node.n_pos}",
                f"base rate: {base:.1%}",
            ]

    def draw_nodes(node):
        x, y = positions[node.node_id]
        color = color_split if not node.is_leaf else color_leaf
        alpha = 0.85 if not node.is_leaf else 0.75

        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width, node_height,
            boxstyle="round,pad=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(box)

        lines = node_label(node)
        n_lines = len(lines)
        for i, line in enumerate(lines):
            offset = (i - (n_lines - 1) / 2) * (node_height / (n_lines + 0.5))
            weight = "bold" if i == 0 else "normal"
            ax.text(
                x, y + offset, line,
                ha="center", va="center",
                fontsize=fontsize,
                fontweight=weight,
                color="white",
                zorder=3,
                clip_on=True,
            )
        if not node.is_leaf:
            draw_nodes(node.left)
            draw_nodes(node.right)

    draw_nodes(root)

    # ------------------------------------------------------------------ #
    # 5. Título e legenda                                                  #
    # ------------------------------------------------------------------ #

    if title is None:
        title = (
            f"FBTSeg — árvore de segmentação  "
            f"(prof. máx={max_depth}, folhas={n_leaves}, "
            f"métrica={model.metric})"
        )
    ax.set_title(title, fontsize=fontsize + 2, pad=10)

    # Legenda
    patch_split = mpatches.Patch(color=color_split, alpha=0.85, label="Nó de split")
    patch_leaf = mpatches.Patch(color=color_leaf, alpha=0.75, label="Folha")
    ax.legend(handles=[patch_split, patch_leaf], loc="lower right",
              fontsize=fontsize, framealpha=0.8)

    fig.tight_layout()
    return fig
