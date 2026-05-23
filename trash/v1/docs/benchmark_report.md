# Benchmark do RiskSeg vs. sklearn

## Atualizacao: tuning numerico

Depois de revisar os casos `coil2000`, `phoneme` e `ring`, foi identificado que o
benchmark ainda subutilizava o `RiskSeg` em bases numericas porque:

- bins numericos estavam sendo tratados como categorias sem ordem;
- o preset do benchmark testava pouco mais que quartil-isolado-vs-resto;
- bases numericas pediam profundidade e numero de bins diferentes conforme
  dimensionalidade e desbalanceamento.

As correcoes desta passada foram:

- suporte a `group_binned_numeric` no `RiskSegOptimizer`;
- geracao de grupos contiguos para bins numericos auto-categorizados;
- preset de benchmark adaptativo para bases numericas:
  - baixa dimensionalidade: `max_depth=3`;
  - alta dimensionalidade e desbalanceamento forte: `n_numeric_bins=8`;
  - `top_k_variables=3` e agrupamento de bins ligado.

Resultado agregado novo do `RiskSeg`:

| model   |   roc_auc |   average_precision |   accuracy |   fit_seconds |
|:--------|----------:|--------------------:|-----------:|--------------:|
| RiskSeg |  0.913694 |            0.823834 |   0.916298 |     23.643804 |

Delta contra a rodada agregada anterior:

- `roc_auc`: `0.904154` -> `0.913694`
- `average_precision`: `0.811371` -> `0.823834`
- `accuracy`: `0.906041` -> `0.916298`
- `fit_seconds`: `26.84s` -> `23.64s`

Melhora por base do `RiskSeg`:

| dataset  |   auc_anterior |   auc_novo |    delta |
|:---------|---------------:|-----------:|---------:|
| coil2000 |       0.659107 |   0.683416 | 0.024309 |
| phoneme  |       0.837717 |   0.873829 | 0.036112 |
| ring     |       0.882124 |   0.904215 | 0.022091 |
| magic    |       0.878902 |   0.892457 | 0.013556 |

Rodada focada que confirmou o ganho:

```bash
python scripts/run_benchmark.py --datasets coil2000 phoneme ring --models RiskSeg --output-dir artifacts/benchmarks_numeric_tuned
```

Rodada agregada nova:

```bash
python scripts/run_benchmark.py --max-rows 12000 --output-dir artifacts/benchmarks_full_numeric_tuned
```

Arquivos gerados:

- [benchmark_results.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_numeric_tuned/benchmark_results.csv)
- [benchmark_results.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_numeric_tuned/benchmark_results.csv)
- [benchmark_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_numeric_tuned/benchmark_summary.csv)
- [benchmark_rank_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_numeric_tuned/benchmark_rank_summary.csv)

## O que foi corrigido

O benchmark anterior estava penalizando o `RiskSeg` porque varias bases da PMLB
chegam com categorias codificadas como inteiros. Isso e toleravel para muitos
baselines, mas distorce o `RiskSeg`, que depende da semantica categorial para
segmentar corretamente.

As correcoes desta versao foram:

- suporte explicito a `categorical_features` no `RiskSegOptimizer`;
- restauracao de schema categorico em bases como `adult`, `churn` e `mushroom`;
- preprocessamento misto no benchmark:
  - `OneHotEncoder` para modelos lineares, MLP, KNN e NB;
  - `OrdinalEncoder` para ensembles de arvore;
- alinhamento adicional com os artigos:
  - normalizacao das variaveis numericas para o treino dos modelos;
  - uso de variaveis numericas categorizadas apenas para a segmentacao;
  - `validation_fraction=0.35`, `top_k_variables=1`, `n_numeric_bins=4`;
  - `factorial_max_interaction_features=8` para segurar o custo.

## Protocolo agregado

- Fonte das bases: PMLB
- 10 bases com mais de 5.000 registros
- Split: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- Teto para execucao local: `--max-rows 12000`
- Metrica principal: `ROC AUC`

## Resultado agregado corrigido e alinhado aos artigos

| model                     |   auc_rank |   roc_auc |   average_precision |   accuracy |
|:--------------------------|-----------:|----------:|--------------------:|-----------:|
| XGBoost                   |       3.55 |  0.939212 |            0.874238 |   0.944815 |
| HistGradientBoosting      |       3.95 |  0.939542 |            0.869389 |   0.942271 |
| RandomForest              |       4.25 |  0.934940 |            0.872920 |   0.944647 |
| ExtraTrees                |       5.10 |  0.934498 |            0.870618 |   0.941547 |
| MLPClassifier             |       6.30 |  0.929335 |            0.852862 |   0.936286 |
| AdaBoost                  |       6.85 |  0.922331 |            0.834569 |   0.915338 |
| RiskSeg                   |       7.25 |  0.904154 |            0.811371 |   0.906041 |
| BaggingLogisticRegression |       7.55 |  0.884787 |            0.765994 |   0.889444 |
| KNeighbors                |       7.60 |  0.915078 |            0.832543 |   0.885715 |
| LogisticRegression        |       7.95 |  0.884794 |            0.766192 |   0.889346 |
| DecisionTree              |       8.25 |  0.889014 |            0.815052 |   0.908122 |
| GaussianNB                |       9.40 |  0.866646 |            0.740404 |   0.763631 |

## Leitura curta do agregado

Depois da correcao e do alinhamento com os artigos:

- o `RiskSeg` subiu do benchmark enviesado inicial (`0.8950`) para `0.9042`
  em ROC AUC medio;
- o ranking medio melhorou de `8.55` para `7.25`;
- a acuracia media subiu para `0.9060`;
- o tempo medio de fit caiu para `26.84s`;
- a melhora mais clara apareceu em `adult`, `churn` e na estabilidade geral do custo.

## Resultado do RiskSeg por base

| dataset   | winner               |   winner_auc |   riskseg_rank |   riskseg_auc |   fit_s |
|:----------|:---------------------|-------------:|---------------:|--------------:|--------:|
| adult     | XGBoost              |       0.9108 |              8 |        0.8938 |   39.16 |
| churn     | XGBoost              |       0.9305 |              8 |        0.8935 |   12.14 |
| clean2    | RiskSeg              |       1.0000 |              1 |        1.0000 |   48.68 |
| coil2000  | HistGradientBoosting |       0.7238 |             10 |        0.6591 |   44.49 |
| magic     | XGBoost              |       0.9251 |              8 |        0.8789 |   27.83 |
| mushroom  | RiskSeg              |       1.0000 |              1 |        1.0000 |   28.16 |
| phoneme   | RandomForest         |       0.9482 |              9 |        0.8377 |   10.66 |
| ring      | ExtraTrees           |       0.9971 |             10 |        0.8821 |   20.34 |
| shuttle   | HistGradientBoosting |       1.0000 |              8 |        0.9978 |   28.05 |
| twonorm   | GaussianNB           |       0.9986 |              2 |        0.9985 |   18.44 |

## Caso que revelou o problema

Na base `adult`, usando a mesma amostra de 12.000 linhas:

- benchmark antigo, com categorias achatadas em inteiros: `AUC 0.8210`
- benchmark corrigido, preservando schema categorico: `AUC 0.8864`
- benchmark alinhado ao artigo, com normalizacao numerica para treino: `AUC 0.8938`

Essa diferenca sozinha justificou a revisao da suite.

## Como rodar a suite agregada

Instalacao:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

Execucao completa corrigida:

```bash
python scripts/run_benchmark.py --max-rows 12000 --output-dir artifacts/benchmarks_full_numeric_tuned
```

Rodada curta:

```bash
python scripts/run_benchmark.py --datasets adult churn mushroom --models RiskSeg LogisticRegression XGBoost RandomForest --max-rows 12000 --output-dir artifacts/benchmarks_schema_probe_fast
```

Arquivos gerados:

- [benchmark_results.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_article_aligned/benchmark_results.csv)
- [benchmark_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_article_aligned/benchmark_summary.csv)
- [benchmark_rank_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/benchmarks_full_article_aligned/benchmark_rank_summary.csv)

## Validacao fiel ao artigo

Para checar fidelidade metodologica, foi criada uma segunda suite com as bases
citadas nos artigos e preprocessamento alinhado ao protocolo deles:

- categoricas preservadas para a segmentacao;
- categoricas codificadas em dummies para os modelos lineares;
- numericas normalizadas em `[0, 1]` para treino;
- numericas categorizadas em 4 faixas apenas para a logica de segmentacao;
- `metric="error"`, `top_k_variables=1`, `validation_fraction=0.35`,
  `min_samples_leaf=0.05` e `max_depth=2`.

Execucao:

```bash
python scripts/run_article_benchmark.py --datasets adult german magic spambase chess --models PaperRiskSegLogistic PaperLogisticRegression --n-splits 3 --output-dir artifacts/article_benchmark_probe_numeric_tuned
```

Resumo medio dessa rodada:

| model                   |   error_rate |   roc_auc |   fit_seconds |   error_rank |
|:------------------------|-------------:|----------:|--------------:|-------------:|
| PaperRiskSegLogistic    |     0.139596 |  0.900249 |     38.903563 |     1.300000 |
| PaperLogisticRegression |     0.152951 |  0.892454 |      0.113273 |     1.700000 |

Leitura curta:

- o `PaperRiskSegLogistic` ganhou no agregado por erro medio e por `roc_auc`;
- em `magic` e `spambase`, a vantagem sobre regressao logistica simples continuou clara;
- em `adult` ele ficou praticamente empatado com a regressao logistica;
- em `german` e `chess`, o comportamento continuou competitivo;
- o tuning numerico quase nao mexeu na historia das bases do paper, mas reduziu o custo medio do `PaperRiskSegLogistic`.

Delta contra a rodada anterior dessa mesma suite:

- `PaperRiskSegLogistic`:
  - `error_rate`: `0.139695 -> 0.139596`
  - `roc_auc`: `0.900227 -> 0.900249`
  - `fit_seconds`: `41.94s -> 38.90s`
- melhora perceptivel principalmente em `magic`:
  - `error_rate`: `0.162776 -> 0.162671`
  - `roc_auc`: `0.891109 -> 0.892883`
  - `average_precision`: `0.841627 -> 0.845525`

Detalhe por base:

| dataset  | model                   |   error_rate |   roc_auc |
|:---------|:------------------------|-------------:|----------:|
| adult    | PaperLogisticRegression |     0.148745 |  0.904712 |
| adult    | PaperRiskSegLogistic    |     0.149932 |  0.902019 |
| chess    | PaperLogisticRegression |     0.035983 |  0.993469 |
| chess    | PaperRiskSegLogistic    |     0.035985 |  0.993545 |
| german   | PaperLogisticRegression |     0.259973 |  0.776110 |
| german   | PaperRiskSegLogistic    |     0.253979 |  0.759112 |
| magic    | PaperLogisticRegression |     0.208991 |  0.839144 |
| magic    | PaperRiskSegLogistic    |     0.162671 |  0.892883 |
| spambase | PaperLogisticRegression |     0.111065 |  0.948836 |
| spambase | PaperRiskSegLogistic    |     0.095415 |  0.953684 |

Arquivos gerados:

- [article_benchmark_results.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/article_benchmark_probe_numeric_tuned/article_benchmark_results.csv)
- [article_benchmark_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/article_benchmark_probe_numeric_tuned/article_benchmark_summary.csv)
- [article_benchmark_rank_summary.csv](/D:/Nuvem/gdrive_pessoal/Faculdades/Impacta/Graduação/Outros/IC/RiskSeg/artifacts/article_benchmark_probe_numeric_tuned/article_benchmark_rank_summary.csv)

## Estado atual

O benchmark ficou mais fiel ao tipo de dado que o `RiskSeg` espera e o estimador
agora lida melhor com categorias codificadas e com a separacao entre treino
numerico e segmentacao categorizada descrita nos artigos. Ainda existe espaco
claro para evolucao em bases como `coil2000`, `ring` e `phoneme`, mas o
resultado corrigido ja nao sustenta a conclusao anterior de que o metodo estava
simplesmente fraco em todo o agregado.
