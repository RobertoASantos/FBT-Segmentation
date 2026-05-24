"""fbtseg — Find Best Tree Segmentation.

Implementacao do metodo de segmentacao binaria recursiva descrito por
Roberto Santos & Roberto Barros (UFPE, ICAI 2012 e ICTAI 2012). A
estrategia em cada no terminal e:

1. Screening fatorial das variaveis candidatas (regressao com interacoes).
2. Teste de cada categoria/grupo da variavel escolhida.
3. Treino de classificadores especialistas para cada segmento.
4. Combinacao por Stacking ou Marginal Odds.
5. Aceitacao do split conforme ganho minimo / perda maxima.

Estimador binario, sklearn-compativel, com predicao vetorizada.

Importacoes principais:

- `FBTSeg` (alias `RiskSegV2` por compatibilidade): estimador principal.
- `thesis_preset`, `article_uci_preset`, `article_synthetic_preset`:
    presets fieis a tese, ICAI 2012 e ICTAI 2012 respectivamente.
- `LinearProbabilityClassifier`: base learner usado para replicar a
    config "Linear" do paper.
- `load_article_dataset`, `article_specs`, `get_spec`: loaders das 5
    bases UCI usadas pelo paper.
"""

from .base_learners import LinearProbabilityClassifier
from .plot import plot_tree, plot_model_tree
from .combiners import MarginalOddsCombiner, StackingCombiner, build_combiner
from .datasets import DatasetSpec, article_specs, get_spec, load_article_dataset
from .estimator import (
    RiskSegV2,
    article_synthetic_preset,
    article_uci_preset,
    thesis_preset,
)
from .metrics import all_metrics, metric_score
from .tree import Node, collect_leaves, collect_nodes, route_observations, route_pair_parent
from .views import ModelView, SegView

# Alias publico (FBTSeg = nome do paper; RiskSegV2 = nome herdado da V2)
FBTSeg = RiskSegV2

__version__ = "0.2.2"

__all__ = [
    "FBTSeg",
    "RiskSegV2",
    "thesis_preset",
    "article_uci_preset",
    "article_synthetic_preset",
    "LinearProbabilityClassifier",
    "DatasetSpec",
    "article_specs",
    "get_spec",
    "load_article_dataset",
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
    "plot_tree",
    "plot_model_tree",
    "__version__",
]
