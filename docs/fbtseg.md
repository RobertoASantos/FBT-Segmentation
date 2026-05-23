# fbtseg — Find Best Tree Segmentation

Pacote Python fiel ao **Capitulo 4** da tese de Roberto Angelo Fernandes
Santos (UFPE, 2010) e aos dois artigos publicados pelos mesmos autores:

- **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing Segmentation Methods
  with Different Base Classifiers. *In:* International Conference on
  Artificial Intelligence (ICAI), Las Vegas, USA, 2012.
  [[PDF local]](reference/ICAI2012-rafs-proceed.pdf)
- **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing FBTSeg and NNTree
  Implementations with Established Ensemble Methods. *In:* IEEE
  International Conference on Tools with Artificial Intelligence
  (ICTAI), Athens, Greece, 2012.
  [[PDF local]](reference/ICTAI2012-Submitted-140-FBTSeg.pdf)

Para citações completas (BibTeX) e referências secundárias (Wolpert
1992, Thomas et al. 2002, Maji 2008, etc.), veja
[references.md](references.md).

Estimador binario, sklearn-compativel, com predicao vetorizada por folha
e screening fatorial em numpy puro.

## Estrutura do pacote

```
fbtseg/
  __init__.py    # FBTSeg (= RiskSegV2), presets, helpers
  __main__.py    # python -m fbtseg fit/predict
  cli.py         # CLI
  metrics.py     # error, auc, ks (=ks2), lift, precision, odds_ratio
  views.py       # SegView (codigos int) + ModelView (one-hot + scale + tracking)
  combiners.py   # StackingCombiner + MarginalOddsCombiner (com referencia)
  tree.py        # Node + roteamento vetorizado
  estimator.py   # FBTSeg/RiskSegV2 + 3 presets
  base_learners.py  # LinearProbabilityClassifier
  datasets.py    # loaders das 5 bases UCI do paper
```

Importacoes principais:

```python
from fbtseg import (
    FBTSeg,
    thesis_preset,
    article_uci_preset,
    article_synthetic_preset,
    LinearProbabilityClassifier,
    load_article_dataset,
)
```

`FBTSeg` e o nome publico; `RiskSegV2` continua como alias para nao
quebrar codigo existente.

## Algoritmo (fiel ao Capitulo 4)

1. **Pre-processamento**: `SegView` codifica colunas em inteiros
   (categoricas) ou indices de bin (numericas via quartis). `ModelView`
   prepara matriz numerica one-hot + escala [0,1].
2. **Em cada no terminal**:
   - separa treino interno e validacao (`validation_fraction`);
   - **Screening**: monta regressao fatorial
     `dummies(var) + main_effects + interacoes` por variavel candidata;
     ranqueia por `D` (metrica) na validacao;
   - **Busca de categoria/grupo**: para o top-`k` variaveis, testa
     categorias individuais e grupos (se variavel em
     `grouping_features`);
   - Treina modelos esquerda/direita, junta por `Stacking` ou
     `MarginalOdds` e calcula `D` na validacao;
   - **Aceita** se ganho >= `min_gain_pct` ou perda <= `max_loss_pct`.
3. **Filhos** treinam **sem a variavel ja usada** (efeito da
   segmentacao eliminado, conforme tese).
4. **Predicao**: roteia todas as observacoes por mascaras booleanas;
   cada folha chama `predict_proba` uma unica vez no seu bloco de
   linhas.

## Parametros mapeados a notacao da tese

| Tese | fbtseg |
|---|---|
| `P` (variaveis candidatas) | `screening_variables` |
| `D` (metrica) | `metric` (`error`, `auc`, `ks`, `lift`, `precision`, `odds_ratio`) |
| `M` (metodo de escolha) | screening fatorial + `screening_estimator` |
| `N` (nº max de ramos) | `2` (fixo, binario) |
| `f` (funcao de combinacao) | `combiner_method` (`stacking` ou `marginal_odds`) |
| `B` (criterios de parada) | `max_depth`, `min_samples_leaf`, `min_gain_pct`, `max_loss_pct` |
| `rQtdeVarTeste` | `top_k_variables` |
| `rQtdeDivisoes` | `n_numeric_bins` |
| `rUsaBlocos` | `grouping_features` |
| `rQtdeBlocos` | `max_group_size` |
| `rTamValidacao` | `validation_fraction` |
| `rTecnicaFatorial` | `screening_estimator` |

## Presets

```python
from fbtseg import thesis_preset, article_uci_preset, article_synthetic_preset

# Capitulo 4 da tese (max_depth=3, val=0.25, k=1, max_loss_pct=0.02):
model = thesis_preset(categorical_features=("cat_col",))

# Artigo ICAI 2012 (UCI: max_depth=2, val=0.35, k=1, sem perda aceita):
model = article_uci_preset()

# Artigo ICTAI 2012 (sinteticos: max_depth=3, min_leaf=10%, val=0.25):
model = article_synthetic_preset()
```

## Resultados de validacao (paper ICAI 2012)

Comparativo `LR(penalty=None)` vs `fbtseg.article_uci_preset()` em
3-fold CV; numeros do paper como referencia (Logistic como base
learner).

| Dataset  | LR (baseline) | Paper Simple | **fbtseg** | Paper FBTSeg |
|----------|---:|---:|---:|---:|
| Chess    | 2.41%  | 2.60%   | **1.41%**  | 1.02%  |
| Spambase | 7.56%  | 7.06%   | 7.82%      | 7.42%  |
| Magic    | 20.96% | 20.98%  | **16.07%** | 18.69% |
| German   | 26.50% | 25.70%  | 26.50%     | 26.40% |
| Adult    | 14.76% | 14.90%  | 14.85%     | 14.96% |

fbtseg alinha o **baseline** com o paper (causa raiz da divergencia da
V1) e aproxima ou supera o FBTSeg do paper. No Magic, fbtseg **bate** a
marca historica do paper.

## Performance

| Dataset (3 folds) | V1 fit_s | fbtseg fit_s | speedup fit | V1 pred_s | fbtseg pred_s | speedup pred |
|---|---:|---:|---:|---:|---:|---:|
| Chess    | 10.06 | 1.35 | **7.4x** | 1.38 | 0.025 | **55x**   |
| Spambase | 15.04 | 2.76 | **5.4x** | 2.23 | 0.006 | **370x**  |
| Magic    | ~35   | 2.05 | **17x**  | ~10  | 0.003 | **3000x** |

## Como rodar

```bash
# todos os testes
python -m pytest tests/

# benchmark fbtseg vs LR
python scripts/run_benchmark.py --datasets chess magic spambase german --n-splits 3

# replicacao Tabela 1 do paper ICAI 2012 (3 base learners x 4 bases)
python scripts/run_paper_table_replication.py \
    --datasets chess magic spambase german --n-splits 3 \
    --output-dir artifacts/paper_table

# validacao ICTAI sintetico
python scripts/validate_ictai_synthetic.py \
    --datasets 1 2 3 4 --sizes 1000 3000 5000 --n-replications 10
```

## CLI

O pacote expoe um CLI via `python -m fbtseg`:

```bash
# Treinar e salvar
python -m fbtseg fit --dataset chess --preset article_uci --output-dir runs/chess

# Treinar com CSV proprio
python -m fbtseg fit --csv data.csv --target y --preset thesis \
    --categorical cat_col_1 cat_col_2

# Predizer com modelo salvo
python -m fbtseg predict --model runs/chess/model.pkl --csv new_data.csv --output preds.csv
```

Apos instalar via `pip install -e .`, o entry-point `fbtseg` tambem
fica disponivel:

```bash
fbtseg fit --dataset adult --preset article_uci
```

## Base learners adicionais

fbtseg aceita qualquer estimador sklearn binario via `base_estimator`.
Para replicar a comparacao 3×N do paper ICAI 2012 (Linear/Logistic/MLP):

```python
from fbtseg import FBTSeg, LinearProbabilityClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 1. Logistic (default)
m_log = FBTSeg()

# 2. Linear probability (replica config "Linear" do paper)
m_lin = FBTSeg(base_estimator=LinearProbabilityClassifier())

# 3. MLP neural network
m_mlp = FBTSeg(
    base_estimator=MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42)
)
```

## Recursos suportados

| Recurso | fbtseg |
|---|---|
| `screening_variables` (P explicito) | sim |
| `categorical_features` | sim |
| `grouping_features` (rUsaBlocos por variavel) | sim |
| `sample_weight` | sim (`fit(X, y, sample_weight=...)`) |
| `classification_threshold` configuravel | sim |
| `min_gain_pct` / `max_loss_pct` (percentuais) | sim |
| `prediction_mode='leaf'` (default) | sim |
| `prediction_mode='pair_combiner'` | sim |
| `prediction_mode='cascade'` | sim (media uniforme ao longo do caminho) |
| `prediction_mode='global_stacking'` | sim (treinado **out-of-fold**, sem leakage) |
| `combiner_method='stacking'` | sim |
| `combiner_method='marginal_odds'` | sim (com segmento de **referencia** real) |
| `get_tree_summary()` | sim (DataFrame por no) |
| `get_summary()` | sim (dict agregado, com terminologia da tese) |
| `plot_model_tree()` | sim (texto ASCII) |
| `evaluate(X, y)` | sim (todas as metricas) |
| Suporte a `sklearn.base.clone` | sim |
| Suporte a binarios apenas | sim (como na tese) |

## Historico

A V1 (`RiskSegOptimizer`, `RiskSegRaizClassifier`) e a primeira V2
("checkpoint_001") foram movidas para `trash/v1/` apenas para
auditoria; nao devem ser usadas nem importadas novamente. Veja
[CHANGELOG.md](../CHANGELOG.md) para detalhes da transicao
V1 → fbtseg.

## Limitacoes conhecidas

- `factorial_max_interaction_features=None` no `thesis_preset` faz a
  regressao fatorial plena, mas pode ser caro em bases com muitas
  dummies; o `article_uci_preset` usa `16` como teto.
- O preset `thesis_preset` aceita `max_loss_pct=0.02` (perda toleravel
  de 2%) para escapar de maximos locais conforme a tese; quem prefere
  rejeicao estrita usa `article_uci_preset` (`max_loss_pct=0.0`).
