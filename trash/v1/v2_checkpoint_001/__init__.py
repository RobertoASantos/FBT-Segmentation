"""RiskSeg V2 — implementacao revisada, fiel a tese e vetorizada.

Diferencas frente a V1 (`riskseg.optimizer`):

- modelos descendentes treinam sem a variavel de split;
- `Marginal Odds` com segmento de referencia real;
- `grouping_features` (`rUsaBlocos`) por variavel;
- `min_gain_pct` / `max_loss_pct` em percentual do baseline;
- predicao vetorizada por folha;
- baseline `LogisticRegression(penalty=None)` por padrao;
- screening fatorial vetorizado em numpy;
- metrica `odds_ratio` disponivel.
"""

from .combiners import MarginalOddsCombiner, StackingCombiner, build_combiner
from .estimator import (
    RiskSegV2,
    article_synthetic_preset,
    article_uci_preset,
    thesis_preset,
)
from .metrics import all_metrics, metric_score
from .tree import Node, collect_leaves, collect_nodes, route_observations, route_pair_parent
from .views import ModelView, SegView

__all__ = [
    "RiskSegV2",
    "thesis_preset",
    "article_uci_preset",
    "article_synthetic_preset",
    "SegView",
    "ModelView",
    "StackingCombiner",
    "MarginalOddsCombiner",
    "build_combiner",
    "metric_score",
    "all_metrics",
    "Node",
    "collect_leaves",
    "collect_nodes",
    "route_observations",
    "route_pair_parent",
]
