# fbtseg

[![PyPI version](https://img.shields.io/pypi/v/fbtseg.svg)](https://pypi.org/project/fbtseg/)
[![Python](https://img.shields.io/pypi/pyversions/fbtseg.svg)](https://pypi.org/project/fbtseg/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Implementação em Python do método **FBTSeg** (*Find Best Tree
Segmentation*, originalmente nomeado RISKSEG) proposto na tese de
Roberto Angelo Fernandes Santos (UFPE, 2010) e nos artigos
[ICAI 2012](docs/reference/ICAI2012-rafs-proceed.pdf) e
[ICTAI 2012](docs/reference/ICTAI2012-Submitted-140-FBTSeg.pdf).
Estimador sklearn-compatível, focado em alvo binário e reprodução fiel
dos resultados publicados.

## Instalação

```bash
pip install fbtseg
```

ou em modo desenvolvimento:

```bash
pip install -e .[test]
```

## Uso rápido

```python
from fbtseg import FBTSeg, article_uci_preset

model = article_uci_preset(categorical_features=("workclass", "education"))
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print(model.plot_model_tree())
summary = model.get_summary()
```

Custom:

```python
from fbtseg import FBTSeg

model = FBTSeg(
    max_depth=3,
    min_samples_leaf=0.05,
    metric="error",
    top_k_variables=1,                  # rQtdeVarTeste
    n_numeric_bins=4,                   # rQtdeDivisoes
    grouping_features=("education",),    # rUsaBlocos
    max_group_size=2,                    # rQtdeBlocos
    combiner_method="stacking",          # ou 'marginal_odds'
    prediction_mode="leaf",              # 'leaf' | 'pair_combiner' | 'cascade' | 'global_stacking'
    drop_split_feature_in_children=True,
)
model.fit(X_train, y_train)
```

## Como funciona

Em cada nó terminal o método:

1. Faz **screening fatorial** — uma regressão por variável candidata,
   incluindo dummies da variável, efeitos principais e interações.
2. Para a(s) melhor(es) variável(eis), testa **segmentações binárias**
   (categoria-vs-complemento ou grupo de categorias).
3. Treina **especialistas** para cada segmento e os combina por
   `Stacking` ou `Marginal Odds`.
4. **Aceita a divisão** se ela melhora a métrica D em validação interna
   (com tolerância configurável de ganho mínimo / perda máxima).
5. Recorre nos filhos até atingir profundidade máxima ou esgotar
   candidatos.

A predição é vetorizada por folha (1 chamada a `predict_proba` por
folha, não 1 por linha) — 50× a 3000× mais rápida que implementações
ingênuas.

## CLI

```bash
# Treinar e salvar
fbtseg fit --dataset chess --preset article_uci --output-dir runs/chess

# CSV próprio
fbtseg fit --csv data.csv --target y --preset thesis \
    --categorical cat_col_1 cat_col_2

# Predição com modelo salvo
fbtseg predict --model runs/chess/model.pkl --csv new_data.csv --output preds.csv
```

## Reprodução do paper ICAI 2012

```bash
# Tabela 1 do paper — Linear/Logistic/MLP × bases UCI
python scripts/run_paper_table_replication.py --datasets chess magic spambase german \
    --n-splits 3 --output-dir artifacts/paper_table

# Datasets sintéticos do ICTAI 2012
python scripts/validate_ictai_synthetic.py --datasets 1 2 3 4 \
    --sizes 1000 3000 5000 --n-replications 10
```

### Destaques de fidelidade

| Dataset | Base | Paper Simple | Paper FBTSeg | fbtseg Simple | fbtseg |
|---|---|---:|---:|---:|---:|
| Chess | Logistic | 2.60% | 1.02% | 2.41% | **1.41%** |
| Magic | Logistic | 20.98% | 18.69% | 20.96% | **16.07%** |
| Magic | Linear | 32.03% | 16.06% | 21.61% | **16.93%** |
| Spambase | Linear | 14.89% | 13.51% | 11.15% | **10.45%** |

Detalhes em [docs/paper_table_replicated.md](docs/paper_table_replicated.md).

## Estrutura do repositório

```
fbtseg/             # pacote (estimator, views, tree, combiners, metrics, base_learners, datasets, CLI)
tests/              # testes (33 cobrindo fit, predict, presets, base learners, métricas)
scripts/            # scripts de reprodução dos experimentos
docs/               # documentação + reference (tese + 2 PDFs)
artifacts/          # resultados de execuções
trash/v1/           # implementação anterior (V1), preservada para auditoria histórica
```

## Documentação

- [docs/fbtseg.md](docs/fbtseg.md): API completa + mapeamento parâmetros tese ↔ código.
- [docs/paper_table_replicated.md](docs/paper_table_replicated.md): replicação da Tabela 1 do ICAI 2012.
- [docs/benchmark_final.md](docs/benchmark_final.md): benchmark fbtseg vs LR.
- [docs/examples/quickstart.py](docs/examples/quickstart.py): notebook linear executável.
- [CHANGELOG.md](CHANGELOG.md): histórico de versões.

## Licença

[MIT](LICENSE). Use, copie, modifique e distribua livremente.

## Citação

Se você usa este pacote em pesquisa, cite a tese e/ou os artigos:

```bibtex
@phdthesis{santos2010metodo,
  author = {Santos, Roberto Angelo Fernandes},
  title  = {Um Método para Segmentação de Preditores},
  school = {Universidade Federal de Pernambuco},
  year   = {2010},
  url    = {http://www.cin.ufpe.br/~roberto/AlunosPG/Teses/2010-PhD-Roberto.zip},
}

@inproceedings{santos2012comparing,
  author    = {Santos, Roberto Angelo Fernandes and Barros, Roberto Souto Maior de},
  title     = {Comparing Segmentation Methods with Different Base Classifiers},
  booktitle = {Proceedings of the 2012 International Conference on Artificial Intelligence (ICAI 2012)},
  year      = {2012},
}

@inproceedings{santos2012fbtseg,
  author    = {Santos, Roberto Angelo Fernandes and Barros, Roberto Souto Maior de},
  title     = {Comparing FBTSeg and NNTree Implementations with Established Ensemble Methods},
  booktitle = {Proceedings of the 24th IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2012)},
  year      = {2012},
}
```

## Referências

- **SANTOS, R. A. F.** *Um Método para Segmentação de Preditores.* Tese
  (Doutorado em Ciência da Computação) — Centro de Informática, UFPE,
  Recife, 2010. Orientador: Prof. Dr. Roberto Souto Maior de Barros.
  ([docx](docs/reference/Tese_Roberto_Final_Biblioteca.docx))
- **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing Segmentation Methods
  with Different Base Classifiers. *In:* Proceedings of the
  International Conference on Artificial Intelligence (ICAI), Las
  Vegas, USA, 2012.
  ([PDF](docs/reference/ICAI2012-rafs-proceed.pdf))
- **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing FBTSeg and NNTree
  Implementations with Established Ensemble Methods. *In:* Proceedings
  of the IEEE International Conference on Tools with Artificial
  Intelligence (ICTAI), Athens, Greece, 2012.
  ([PDF](docs/reference/ICTAI2012-Submitted-140-FBTSeg.pdf))

Para citações completas (BibTeX) e referências secundárias citadas
pelos artigos (Wolpert 1992 — Stacking, Thomas et al. 2002 — Marginal
Odds, Maji 2008 — NNTree, etc.), veja
[docs/references.md](docs/references.md).
