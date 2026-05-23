# Comparacao RiskSeg atual vs. riskseg_raiz

Data da rodada: 2026-05-20.

## O que mudou

Foi criado um estimador separado, `RiskSegRaizClassifier`, exportado pelo pacote
como:

```python
from riskseg import RiskSegRaizClassifier, RiskSegRaiz
```

O `RiskSegOptimizer` atual foi mantido. O `riskseg_raiz` usa defaults alinhados
ao artigo/tese:

- metrica principal: erro de classificacao;
- profundidade maxima: 2;
- folha minima: 5%;
- validacao interna: 35%;
- numericas normalizadas para treino e discretizadas em 4 faixas para split;
- selecao de 1 variavel candidata por screening fatorial;
- predicao final por `local_combiner`, isto e, o combinador local do split mais
  especifico no caminho da arvore.
- sem suporte publico a `sample_weight`, porque o protocolo original da tese nao
  usa pesos por observacao.

## Comando usado

```bash
python scripts/run_article_benchmark.py \
  --output-dir artifacts/article_benchmark_raiz_compare \
  --datasets adult german magic spambase chess \
  --models PaperRiskSegRaizLogistic \
  --n-splits 3
```

Os resultados do RiskSeg atual e da regressao logistica simples foram
reaproveitados da rodada:

```text
artifacts/article_benchmark_probe_numeric_tuned/article_benchmark_results.csv
```

Tabela consolidada:

```text
artifacts/article_benchmark_raiz_compare/combined_comparison.csv
```

## Resultado por base

Erro medio em percentual. Menor e melhor.

| Base | Artigo Logistic | Artigo FBTSeg | Logistic sklearn | RiskSeg atual | riskseg_raiz | raiz - artigo |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 14.90 | 14.96 | 14.87 | 14.99 | 14.99 | +0.03 pp |
| Chess | 2.60 | 1.02 | 3.60 | 3.60 | 3.57 | +2.55 pp |
| German | 25.70 | 26.30 | 26.00 | 25.40 | 25.90 | -0.40 pp |
| Magic | 20.98 | 15.40 | 20.90 | 16.27 | 16.33 | +0.93 pp |
| Spambase | 8.69 | 8.19 | 11.11 | 9.54 | 9.52 | +1.33 pp |

Media simples nas cinco bases:

| Modelo | Erro medio |
|---|---:|
| Artigo Logistic | 14.57 |
| Artigo FBTSeg | 13.17 |
| Logistic sklearn | 15.30 |
| RiskSeg atual | 13.96 |
| riskseg_raiz | 14.06 |

## Leitura

O `riskseg_raiz` ficou muito proximo do RiskSeg atual na maioria das bases, mas
sem usar o `global_stacking` como predicao final. Isso confirma que separar a
versao raiz faz sentido: ela preserva a tese sem destruir a versao pratica.

O comportamento em Magic ficou coerente com o artigo: a regressao logistica
simples ficou perto de 20.9% de erro e o RiskSeg caiu para perto de 16.3%.
Adult tambem ficou praticamente igual ao artigo, onde FBTSeg com regressao
logistica nao melhorava de forma relevante.

Ainda nao reproduzimos dois pontos do artigo:

- Chess: o artigo reporta FBTSeg em 1.02%, mas nosso `riskseg_raiz` ficou em
  3.57%. A regressao logistica sklearn tambem ficou pior que a do artigo
  (3.60% contra 2.60%), entao ha diferenca de preprocessamento/protocolo ou de
  solver alem da propria segmentacao.
- Spambase: melhoramos bastante a logistica sklearn, mas ainda ficamos 1.33 pp
  acima do FBTSeg reportado.

## Proximos ajustes tecnicos

1. Reproduzir o preprocessamento do SAS Enterprise Miner mais de perto, em
   especial codificacao de categoricas e tratamento de variaveis de Chess.
2. Implementar `rUsaBlocos`/`rQtdeBlocos` por variavel, nao apenas chaves
   globais.
3. Testar `factorial_max_interaction_features=None` em Chess e Spambase para
   remover a heuristica de limite por variancia.
4. Revisar a implementacao literal de Marginal Odds e adicionar a metrica Odds
   Ratio da tese.
