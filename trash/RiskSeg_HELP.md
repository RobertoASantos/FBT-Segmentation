# RiskSegOptimizer — Guia rápido de uso

## O que é
`RiskSegOptimizer` implementa uma árvore de modelos com segmentação por categorias/grupos de categorias e combinação dos modelos por `stacking` ou `marginal_odds`. A ideia central é:

1. **Escolher a melhor variável candidata para segmentar** via um screening fatorial/interativo.
2. **Testar splits binários** nessa variável (`grupo` vs `complemento`).
3. **Treinar especialistas locais** para os dois lados do split.
4. **Comparar o modelo global do nó** com o modelo segmentado combinado.
5. **Repetir recursivamente** até a profundidade máxima ou até não haver ganho.
6. **Alinhar as folhas terminais** em um stacking global, quando `prediction_mode="global_stacking"`.

## Estimadores disponíveis
O framework separa 5 papéis de estimadores:

- `screening_estimator`: escolhe a melhor variável candidata.
- `node_estimator`: modelo global do nó (benchmark do nó).
- `segment_estimator`: modelos especialistas dos segmentos (`left_model`, `right_model` e folhas).
- `local_combiner_estimator`: meta-modelo do combiner local do split.
- `global_combiner_estimator`: meta-modelo do stacking global das folhas.

Se você não informar esses estimadores, o framework usa `LogisticRegression` como default.

## Hiperparâmetros principais

### Estrutura da árvore
- `max_depth`: profundidade máxima da árvore.
- `min_samples_leaf`: massa mínima por folha. Pode ser inteiro ou proporção.
- `min_gain`: ganho mínimo exigido para aceitar um split.
- `validation_fraction`: fração do nó reservada para validação interna do split.

### Screening e segmentação
- `screening_mode`: hoje o fluxo principal usa `"factorial_lr"`.
- `top_k_variables`: quantas variáveis candidatas do screening vão para a etapa de split.
- `use_top_k_variables`: habilita/desabilita o filtro top-k.
- `use_grouping`: permite testar agrupamentos de categorias.
- `max_group_size`: tamanho máximo dos grupos de categorias testados.

### Tratamento de numéricas
- `auto_bin_numeric`: transforma variáveis numéricas em categorias para a segmentação.
- `n_numeric_bins`: quantidade de faixas.
- `numeric_binning`: `"quantile"` ou `"uniform"`.

### Métrica de otimização
- `metric`: `"combined"`, `"lift"`, `"ks"`, `"precision"`, `"auc"`, `"error"`, `"rocmin"`.
- `top_rate`: taxa de topo usada em `Lift@k` e `Precision@k`.
- `metric_weights`: pesos `(lift, ks)` quando `metric="combined"`.
- `classification_threshold`: limiar usado na métrica de erro.

### Combinação dos modelos
- `combiner_method`: `"stacking"` ou `"marginal_odds"`.
- `prediction_mode`: `"leaf"`, `"cascade"` ou `"global_stacking"`.
- `global_stacking_C`: regularização do stacking global default quando o estimador global não é informado.

### Controle geral
- `random_state`: seed.
- `stratify`: estratificação no holdout interno do nó.
- `verbose`: nível de log.

## Como escolher os modos

### Quando usar `prediction_mode="global_stacking"`
Use quando você quer **alinhar os scores das folhas terminais** em uma escala global comum. É o modo mais natural quando o objetivo é evitar distribuições muito diferentes entre folhas.

### Quando usar `prediction_mode="leaf"`
Use quando quiser a saída do **modelo da folha terminal** diretamente, sem meta-modelo global.

### Quando usar `prediction_mode="cascade"`
Use quando quiser preservar uma leitura mais hierárquica do caminho na árvore.

### Quando usar `combiner_method="stacking"`
É a escolha mais geral quando você quer combinar explicitamente os dois lados do split por um meta-modelo.

### Quando usar `combiner_method="marginal_odds"`
É útil quando você quer uma combinação mais simples e orientada a transformação de score/odds por lado do split.

## Sugestões práticas de configuração

### Configuração conservadora
- `metric="combined"`
- `metric_weights=(0.7, 0.3)`
- `max_depth=2` ou `3`
- `min_samples_leaf=0.10`
- `top_k_variables=3`
- `use_grouping=True`
- `max_group_size=2`
- `prediction_mode="global_stacking"`
- todos os estimadores = `LogisticRegression`

### Configuração para reduzir sobreajuste
- diminuir `max_depth`
- aumentar `min_samples_leaf`
- reduzir `top_k_variables`
- usar maior regularização nos combinadores
- comparar sempre treino vs teste via `evaluate_dataset`

### Configuração para foco em extremos positivos
- `metric="ks"` ou `metric="lift"`
- `top_rate` menor, como `0.03` ou `0.05`
- manter `prediction_mode="global_stacking"` se quiser alinhar scores de folhas

## Fluxo mínimo de uso

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

riskseg = RiskSegOptimizer(
    screening_estimator=LogisticRegression(
        solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42
    ),
    node_estimator=LogisticRegression(
        solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42
    ),
    segment_estimator=LogisticRegression(
        solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=42
    ),
    local_combiner_estimator=LogisticRegression(
        solver="lbfgs", C=0.5, max_iter=5000, random_state=42
    ),
    global_combiner_estimator=LogisticRegression(
        solver="lbfgs", C=0.2, max_iter=5000, random_state=42
    ),
    screening_mode="factorial_lr",
    combiner_method="stacking",
    prediction_mode="global_stacking",
    metric="combined",
    top_rate=0.05,
    metric_weights=(0.7, 0.3),
    max_depth=3,
    min_samples_leaf=0.10,
    validation_fraction=0.25,
    auto_bin_numeric=True,
    n_numeric_bins=4,
    numeric_binning="quantile",
    use_grouping=True,
    max_group_size=2,
    top_k_variables=3,
    verbose=1,
)

riskseg.fit(X_train, y_train)
p_test = riskseg.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, p_test)
print(auc_test)

summary = riskseg.get_summary()
test_summary = riskseg.evaluate_dataset(X_test, y_test, dataset_name="test")
```

## O que sempre inspecionar
- `get_summary()`
- `get_node_table()`
- `get_split_table()`
- `evaluate_dataset(X_test, y_test)`
- `plot_model_tree()`

## Interpretação do resumo
- `global_simple_*`: benchmark global sem segmentação.
- `riskseg_global_stacking_*`: resultado final in-sample do stacking global.
- `train_final_prediction_*`: re-predição do treino pela rota final do modelo.
- `delta_*_vs_global_simple`: ganho/perda em relação ao benchmark simples.

## Regra prática
Se o ganho in-sample for alto, mas o ganho no teste cair muito, o próximo ajuste normalmente é:
1. reduzir complexidade da árvore,
2. endurecer regularização,
3. revisar o screening fatorial.
