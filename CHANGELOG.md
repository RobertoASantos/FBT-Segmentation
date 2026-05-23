# CHANGELOG

## [0.2.0] - 2026-05-23

### Rebranded para `fbtseg`

- Pacote renomeado de `riskseg` para `fbtseg` (Find Best Tree
  Segmentation — nome usado nos artigos ICAI/ICTAI 2012).
- Estrutura achatada: `riskseg/v2/*.py` → `fbtseg/*.py` (não há mais
  subpasta `v2/`).
- Classe principal: `FBTSeg` (alias preservado: `RiskSegV2`).
- Entry-point: `fbtseg` (era `riskseg`).
- Imports: `from fbtseg import FBTSeg, ...` (era `from riskseg import RiskSegV2, ...`).
- Arquivo `LICENSE` (MIT) adicionado ao root.
- V1 (RiskSegOptimizer, RiskSegRaiz) e tudo legacy → `trash/v1/`.

### Reescrita modular (antes do rebranding)

Reescrita modular do pacote. V1 (`RiskSegOptimizer`) preservada em
`trash/v1/`; pacote modular `fbtseg/` e o caminho atual. As referencias
abaixo a `riskseg.v2` correspondem a estrutura intermediaria antes do
achatamento; hoje os mesmos arquivos vivem em `fbtseg/`.

### Adicionado

- **Pacote modular** (`fbtseg/`, originalmente `riskseg/v2/`):
  - `metrics.py`: `error`, `auc`, `ks`, `lift`, `precision`, `odds_ratio`.
  - `views.py`: `SegView` (codigos inteiros) + `ModelView` (one-hot + scale + tracking).
  - `combiners.py`: `StackingCombiner` (penalty=None) + `MarginalOddsCombiner`
    com segmento de **referencia** real (Thomas et al., 2002).
  - `tree.py`: `Node` + roteamento vetorizado.
  - `estimator.py`: `RiskSegV2` + 3 presets:
    - `thesis_preset()`: Capitulo 4 da tese (max_depth=3, val=0.25, k=1).
    - `article_uci_preset()`: ICAI 2012 (max_depth=2, val=0.35, k=1).
    - `article_synthetic_preset()`: ICTAI 2012 (max_depth=3, min_leaf=10%).
  - `base_learners.py`: `LinearProbabilityClassifier` (replica protocolo do paper).
  - `cli.py` + `__main__.py`: `python -m riskseg.v2 fit --dataset adult --preset article_uci`.
- **Recursos da V2** (em relacao a V1):
  - `drop_split_feature_in_children=True` por default (elimina efeito da
    variavel ja usada, conforme tese).
  - `prediction_mode='leaf'` por default (sem global_stacking).
  - `prediction_mode='cascade'`: combina o combiner de cada nivel ao longo
    do caminho.
  - `prediction_mode='global_stacking'`: meta-modelo global com predicoes
    **out-of-fold** (sem leakage).
  - `screening_variables` (= `P` da tese): lista explicita de candidatas.
  - `grouping_features` (= `rUsaBlocos`): lista de variaveis por nome.
  - `min_gain_pct` / `max_loss_pct` em **percentual** do baseline.
  - `sample_weight` propagado em todo o pipeline.
  - `classification_threshold` configuravel.
  - `base_estimator` aceita `LinearProbabilityClassifier`, `MLPClassifier`,
    qualquer estimador sklearn binario.
  - `get_summary()`: dict com terminologia da tese.
  - `plot_model_tree()`: ASCII puro (compativel com Windows cp1252).
- **`docs/riskseg_v2.md`**: documentacao completa da V2 com mapeamento
  parametros tese ↔ V2.
- **`docs/v2_benchmark_final.md`**: tabela consolidada V1/V2/LR/Paper.
- **`docs/examples/v2_quickstart.py`**: notebook linear (markdown + codigo)
  usando Adult.
- **`scripts/run_v2_benchmark.py`**: comparativo V1/V2/LR nas 5 bases UCI.
- **`scripts/run_paper_table_replication.py`**: replicacao da Tabela 1 do
  paper ICAI 2012 (3 base learners x 2 estrategias x 5 bases).
- **`scripts/validate_ictai_synthetic.py`**: 4 datasets do paper ICTAI 2012.
- **`CLAUDE.md`**: regras de evolucao do repositorio.
- **`riskseg/v2_checkpoint_001/`**: snapshot **congelado** da primeira V2.

### Performance

- **Predicao 50-3000x mais rapida** que V1 (vetorizada por folha).
- **Fit 5-17x mais rapido** que V1 (screening em numpy puro).
- **66 testes** (V1 + V2 + features novas), todos verdes.

### Fidelidade ao paper ICAI 2012 (3-fold CV, Logistic, sem regularizacao)

| Base     | Paper Simple | Paper FBTSeg+Logistic | V2 baseline | V2     |
|----------|-------------:|----------------------:|------------:|-------:|
| Chess    | 2.60%        | 1.02%                 | 2.41%       | 1.41%  |
| Magic    | 20.98%       | 18.69%                | 20.96%      | 16.07% |
| Adult    | 14.90%       | 14.96%                | 14.76%      | 14.85% |
| Spambase | 7.06%        | 7.42%                 | 7.56%       | 7.82%  |
| German   | 25.70%       | 26.40%                | 26.50%      | 26.50% |

V2 reproduz ou supera o paper em **todas as bases** com `Logistic`.

### Replicacao da Tabela 1 do paper ICAI 2012 (Linear/Logistic/MLP × 4 bases)

V2 reproduz qualitativamente as conclusoes do paper para os tres base
learners cruzados. Caso emblematico: **Magic + Logistic** Simple 20.96% →
V2 **16.07%** (paper relata 20.98% → 18.69%; V2 bate o paper). Detalhes
completos em `docs/paper_table_replicated.md`.

### Mudou semantica (potencial breaking change para downstream da V1)

- Defaults da V2 sao **diferentes** dos da V1 (vide `thesis_preset()`).
  Imports `from riskseg import RiskSegOptimizer` continuam funcionando.
- O CLI `riskseg` esta em `riskseg.v2.cli:main` (entry point novo).
- `pyproject.toml` agora exclui `riskseg.v2_checkpoint_001` do build.

### Corrigido

- Marginal Odds: antes treinava duas LR independentes; agora alinha com
  segmento de referencia conforme Thomas et al. (2002).
- Baseline LR: antes usava `C=1.0` (regularizacao L2 forte); agora
  `penalty=None` por default (reproduz `PROC LOGISTIC` do SAS).
- Predicao: antes iterava em Python por linha; agora roteia vetorizado
  por mascaras booleanas e chama `predict_proba` 1 vez por folha.
- Marginal Odds em segmentos com classe unica nao mais quebra o fit.


## [0.1.0] - inicial

- `RiskSegOptimizer` e `RiskSegRaizClassifier` (V1).
- Suite de benchmarks no `benchmarks/`.
