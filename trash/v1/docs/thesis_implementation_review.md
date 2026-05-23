# Revisao da Implementacao RiskSeg vs. Tese

Referencia principal: `docs/reference/Tese_Roberto_Final_Biblioteca.docx`, em
especial o Capitulo 4. Referencias complementares: os artigos extraidos em
`artifacts/papers/ICAI2012-rafs-proceed.txt` e
`artifacts/papers/ICTAI2012-Submitted-140-FBTSeg.txt`.

## Conclusao executiva

A implementacao atual e uma boa implementacao pratica da familia
RISKSEG/FBTSeg, mas ainda nao e uma reproducao literal unica da tese. Ela mistura
tres perfis:

- **modo tese/artigo**: arvore de segmentacao, screening fatorial, treino de
  especialistas por segmento e validacao interna;
- **modo sklearn pratico**: estimador binario clonavel, `fit`,
  `predict_proba`, `predict`, `score`, presets e benchmarks;
- **extensoes de performance**: `global_stacking`, tuning numerico,
  `group_binned_numeric`, metricas adicionais e limite de interacoes fatoriais.

O maior desvio conceitual e o `prediction_mode="global_stacking"` como caminho
default. Ele melhora calibracao e desempenho em algumas bases, mas adiciona um
meta-modelo global sobre as folhas, enquanto a tese descreve a combinacao dos
pares de segmentos no processo da arvore.

## Mapa da tecnica na tese

A tese descreve o RISKSEG como um metodo de combinacao por segmentacao:

1. Em cada no terminal, selecionar variaveis candidatas por um metodo fatorial
   de interacoes.
2. Para as `k` melhores variaveis candidatas, testar categorias ou grupos de
   categorias como regras binarias de divisao.
3. Treinar classificadores especialistas para os dois segmentos resultantes.
4. Combinar os escores dos segmentos por Stacking ou Marginal Odds.
5. Comparar a segmentacao contra o modelo simples do no usando uma metrica `D`.
6. Aceitar ou rejeitar o split conforme ganho minimo ou perda maxima aceitavel.
7. Repetir sequencialmente nos nos terminais ate atingir criterios de parada.
8. Para numericas, criar faixas categoricas aproximadamente balanceadas para
   segmentacao, mantendo as numericas normalizadas para treino dos modelos.

## Pontos fiéis

### Estrutura de arvore e recursao

A classe `RiskSegOptimizer` implementa a arvore recursiva em
`riskseg/optimizer.py`. O fluxo de `fit` cria `model_view_` e `seg_view_`,
treina a raiz e coleta `node_table_`, `split_table_` e folhas. A funcao
`_fit_node` treina o modelo do no, cria divisao treino/validacao interna,
faz screening, testa splits, aceita/rejeita e recorre nos filhos.

Estado: **aderente**.

### Alvo binario

A implementacao valida o alvo como binario e declara tags sklearn de
classificador binario. Isso e coerente com a tese e os estudos de risco com
resposta dicotomica.

Estado: **aderente**.

### Screening fatorial

O metodo `_screen_candidate_variables` avalia variaveis candidatas usando
desenhos fatoriais montados por `_build_factorial_design_for_variable`. O modo
`factorial_target_interactions` inclui dummies da variavel de segmentacao,
efeitos principais e interacoes com colunas selecionadas do espaco de modelo.

Estado: **aderente com aproximacao computacional**.

Diferença relevante: a tese conceitua a comparacao das regressoes fatoriais por
variavel. A implementacao limita interacoes por
`factorial_max_interaction_features` e seleciona colunas por variancia, o que e
uma heuristica pratica para custo, nao uma regra descrita literalmente.

### Teste de categorias e grupos

O metodo `_evaluate_candidate_splits` testa grupos contra complemento, treina
modelos para esquerda/direita, combina escores e ordena pela metrica escolhida.
`_generate_split_groups` gera categorias individuais e, se habilitado,
agrupamentos.

Estado: **aderente**.

### Especialistas por segmento

Cada split aceito cria `left_model` e `right_model`, e cada filho passa a ter
seu proprio `node_model`. A implementacao permite qualquer estimador sklearn
compatível com `predict_proba`, o que respeita a tese ao permitir diferentes
indutores.

Estado: **aderente e extensivel**.

### Combinacao por Stacking e Marginal Odds

Ha dois combinadores: `LogisticStackingCombiner` e `MarginalOddsCombiner`. O
stacking local segue bem a ideia descrita: uma coluna para o escore do segmento
ao qual o registro pertence e zero na outra.

Estado: **stacking aderente; marginal odds parcialmente aderente**.

Ponto a revisar: a tese descreve Marginal Odds como recalibracao com um segmento
de referencia e calibracao do outro com pesos da regressao logistica. A
implementacao atual ajusta calibradores separados por segmento e prediz com
membership. O efeito pratico e parecido, mas nao e garantidamente o mesmo
algoritmo.

### Numericas para treino e segmentacao

A implementacao separa `model_view_` e `seg_view_`. Numericas podem ser
normalizadas para treino (`scale_model_numeric=True`) e categorizadas para
segmentacao (`auto_bin_numeric=True`, `n_numeric_bins`). O ajuste recente
`group_binned_numeric` permite grupos contiguos em bins numericos.

Estado: **aderente aos artigos e a tese, com melhoria recente**.

## Principais diferencas para a tese

### 1. `global_stacking` como predicao final default

Na tese, a combinacao aparece no nivel dos pares de segmentos, durante a
construcao da arvore. A implementacao atual treina tambem um meta-modelo global
sobre as folhas (`_fit_global_leaf_stacking`) e usa isso como default em
`prediction_mode="global_stacking"`.

Impacto: melhora calibracao em algumas bases, mas muda a tecnica final. O
resultado deixa de ser apenas a arvore segmentada localmente e passa a ser uma
arvore mais um stacking global de folhas.

Severidade: **alta para fidelidade; baixa/moderada para uso pratico**.

### 2. Remocao do efeito da variavel segmentadora nos descendentes

A tese sugere que segmentar reduz a complexidade dos proximos modelos, pois o
efeito da variavel/categoria usada na divisao fica isolado. A implementacao
mantem todas as colunas no espaco de treino dos modelos descendentes.

Impacto: pode deixar modelos filhos reaprendendo a propria regra de divisao,
reduzindo interpretabilidade e, em alguns casos, piorando generalizacao.

Severidade: **alta para fidelidade teorica**.

### 3. `rUsaBlocos` e `rQtdeBlocos`

Na tese, `rUsaBlocos` e uma lista de variaveis que podem ter categorias
agrupadas, com `rQtdeBlocos` controlando o tamanho maximo. No codigo,
`use_grouping` e global, `max_group_size` e global, e `group_binned_numeric`
e separado para bins numericos.

Impacto: o comportamento fica menos expressivo que o parametro original. Pode
aumentar custo em variaveis onde blocagem nao deveria ocorrer ou impedir
blocagem seletiva.

Severidade: **media**.

### 4. Metricas disponiveis

A tese lista erro de classificacao, KS2, ROC e Odds Ratio por intervalos de
escore. O codigo oferece `error`, `ks`, `auc`, `rocmin`, `lift`, `precision` e
`combined`.

Impacto: bom para uso pratico, mas ainda falta Odds Ratio da tese e ha duvida se
o `ks` implementado corresponde exatamente ao KS2 citado.

Severidade: **media**.

### 5. Aceitacao de perda maxima

A tese explicita ganho minimo ou perda maxima aceitavel. O codigo usa
`min_gain`, e aceita split quando `gain > min_gain`. Isso permite perda se
`min_gain` for negativo, mas a semantica nao esta documentada como percentual
nem nomeada como perda maxima.

Impacto: funcionalmente possivel, mas menos claro para quem tenta reproduzir os
parametros originais.

Severidade: **baixa/media**.

### 6. Pre-selecao fatorial com limite por variancia

`factorial_max_interaction_features` controla custo limitando colunas de
interacao. Isso foi importante para performance, mas nao e a busca fatorial
plena da tese.

Impacto: pode perder variaveis interativas relevantes de baixa variancia.

Severidade: **media para performance; baixa para operacao em bases grandes**.

### 7. Presets atuais misturam paper e benchmark

`paper_preset` usa `max_depth=2`, `validation_fraction=0.35` e `top_k=1`,
alinhado ao artigo ICAI/UCI. Ja o benchmark amplo usa uma heuristica adaptativa
em `recommend_riskseg_benchmark_params`. A tese e o artigo ICTAI sintetico
tambem usam configuracoes diferentes, como profundidade 3, validacao 25% e
`rQtdeVarTeste=3` em alguns experimentos.

Impacto: correto para benchmarks especificos, mas confuso como API publica se
nao houver nomes explicitos para os modos.

Severidade: **media**.

### 8. Stacking global treinado no mesmo conjunto

O meta-modelo global de folhas e treinado sobre predicoes geradas por modelos
treinados no proprio conjunto de treino. Isso pode inflar metricas internas.
Os benchmarks externos reduzem esse risco, mas o `summary_` de treino pode
parecer melhor do que e.

Impacto: risco de overfitting e leitura otimista do resumo interno.

Severidade: **media**.

### 9. Documentacao de benchmark ficou com historico misturado

`docs/benchmark_report.md` contem a rodada nova no topo, mas ainda preserva
tabelas antigas sob o titulo de resultado agregado corrigido e links antigos em
uma secao posterior. Isso nao quebra a tecnica, mas pode confundir a leitura.

Severidade: **baixa, mas facil de corrigir**.

## Melhorias recomendadas

### Prioridade 1: separar modos oficiais

Criar presets nomeados e documentados:

- `RiskSegOptimizer.thesis_preset()`: reproducao mais literal do Capitulo 4.
- `RiskSegOptimizer.article_uci_preset()`: protocolo ICAI/UCI.
- `RiskSegOptimizer.article_synthetic_preset()`: protocolo ICTAI/sintetico.
- `RiskSegOptimizer.performance_preset()`: modo sklearn pratico.

O `paper_preset` atual deveria virar alias ou ser depreciado para evitar
ambiguidade.

### Prioridade 2: modo de predicao fiel a tese

Implementar e validar um modo `prediction_mode="thesis"` ou tornar
`cascade`/`leaf` o default do preset de tese, deixando `global_stacking` como
extensao explicita.

Teste esperado: na suite do artigo, comparar `leaf`, `cascade` e
`global_stacking` sem mudar o resto da configuracao.

### Prioridade 3: `rUsaBlocos` por variavel

Adicionar parametros:

- `grouping_features=None`
- `max_group_size_by_feature=None`

Manter `use_grouping` como compatibilidade, mas permitir controle fiel por
variavel. Para bins numericos, manter `group_binned_numeric` separado.

### Prioridade 4: implementar Odds Ratio e revisar KS2

Adicionar metrica `odds_ratio` por faixas de escore e confirmar se `ks` deve ser
KS simples ou KS2 conforme a tese. Isso e importante para reproduzir estudos de
risco de credito.

### Prioridade 5: Marginal Odds literal

Reimplementar ou adicionar `combiner_method="marginal_odds_reference"` seguindo
a descricao da tese:

1. calibrar escore de cada segmento contra alvo;
2. escolher segmento de referencia;
3. alinhar o outro segmento usando pesos da regressao logistica;
4. documentar a escala final do escore.

### Prioridade 6: reduzir reaprendizado da regra nos descendentes

Experimentar uma opcao:

- `drop_split_feature_in_children=True`; ou
- `mask_split_dummy_in_children=True` para remover apenas dummies/efeitos da
  variavel usada; ou
- `conditioned_child_models=True`, mantendo a variavel para auditoria mas fora
  do treino do especialista.

Isso deve ser testado porque pode melhorar fidelidade, mas nem sempre
performance.

### Prioridade 7: stacking global out-of-fold

Se `global_stacking` permanecer como extensao, treinar o meta-modelo de folhas
com predicoes out-of-fold dentro do treino. Isso reduz vazamento interno e deixa
o `summary_` mais honesto.

### Prioridade 8: documentar parametros originais lado a lado

Adicionar uma tabela de equivalencia:

| Tese | Implementacao atual |
|:--|:--|
| `D` | `metric` |
| `M` | `screening_mode` + `screening_estimator` |
| `rQtdeVarTeste` | `top_k_variables` |
| `rQtdeDivisoes` | `n_numeric_bins` |
| `rUsaBlocos` | `use_grouping` / futura `grouping_features` |
| `rQtdeBlocos` | `max_group_size` |
| profundidade maxima | `max_depth` |
| minimo por segmento | `min_samples_leaf` |
| validacao | `validation_fraction` |
| funcao de juncao | `combiner_method` |

### Prioridade 9: limpar o relatorio de benchmark

Separar o historico em secoes:

- rodada enviesada inicial;
- rodada corrigida por schema categorico;
- rodada artigo;
- rodada numerica tunada;
- resultados vigentes.

Isso evita que a documentacao conte duas historias ao mesmo tempo.

## Estado atual de maturidade

Como tecnica sklearn-like, o projeto esta em bom estado: tem pacote, testes,
benchmarks, presets, suporte a categorias e numericas, e resultado competitivo
em varias bases. Como reproducao academica da tese, ainda precisa de uma
camada de fidelidade explicita: presets por protocolo, Marginal Odds literal,
Odds Ratio, `rUsaBlocos` por variavel e modo de predicao sem stacking global
por padrao.
