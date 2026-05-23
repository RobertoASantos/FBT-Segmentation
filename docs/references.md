# Referências bibliográficas

## Trabalhos que fundamentam o `fbtseg`

### Tese de doutorado

> **SANTOS, R. A. F.** *Um Método para Segmentação de Preditores.*
> Tese (Doutorado em Ciência da Computação) — Centro de Informática,
> Universidade Federal de Pernambuco. Recife, 2010.
> Orientador: Prof. Dr. Roberto Souto Maior de Barros.
>
> **Capítulo 4** descreve o método RISKSEG (renomeado para FBTSeg nos
> artigos posteriores) e é a referência principal para esta
> implementação.
>
> Arquivo local: [`docs/reference/Tese_Roberto_Final_Biblioteca.docx`](reference/Tese_Roberto_Final_Biblioteca.docx).

### Artigos publicados

> **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing Segmentation
> Methods with Different Base Classifiers. *In:* International
> Conference on Artificial Intelligence (ICAI 2012), Las Vegas, USA.
> Proceedings... CSREA Press, 2012.
>
> Sigla: **ICAI 2012**. Cinco bases UCI (Adult, Chess, German, Magic,
> Spambase), três base learners (Linear, Logistic, MLP), três métodos
> de segmentação (NNTree, TradSeg, FBTSeg). 10-fold cross-validation,
> métrica = erro de classificação.
>
> Arquivo local: [`docs/reference/ICAI2012-rafs-proceed.pdf`](reference/ICAI2012-rafs-proceed.pdf).

> **SANTOS, R. A. F.; BARROS, R. S. M.** Comparing FBTSeg and
> NNTree Implementations with Established Ensemble Methods. *In:*
> IEEE International Conference on Tools with Artificial Intelligence
> (ICTAI 2012), Athens, Greece. Proceedings... IEEE, 2012.
>
> Sigla: **ICTAI 2012**. Quatro datasets sintéticos (D1–D4) com 3
> tamanhos de amostra (1k, 3k, 5k) e 30 replicações. Comparação de
> FBTSeg/NNTree com Bagging, Boosting, TradSeg.
>
> Arquivo local: [`docs/reference/ICTAI2012-Submitted-140-FBTSeg.pdf`](reference/ICTAI2012-Submitted-140-FBTSeg.pdf).

## Como citar este pacote

Se você usar o `fbtseg` em trabalho acadêmico, por favor cite a tese e
o artigo ICAI 2012 (que descreve o método em mais detalhes):

```bibtex
@phdthesis{Santos2010Riskseg,
  author = {Santos, Roberto Angelo Fernandes},
  title  = {{Um M\'etodo para Segmenta\c{c}\~ao de Preditores}},
  school = {Universidade Federal de Pernambuco, Centro de Inform\'atica},
  year   = {2010},
  type   = {Tese de Doutorado},
  address= {Recife, Brasil},
}

@inproceedings{Santos2012ICAI,
  author    = {Santos, Roberto Angelo Fernandes and de Barros, Roberto Souto Maior},
  title     = {{Comparing Segmentation Methods with Different Base Classifiers}},
  booktitle = {Proceedings of the International Conference on Artificial Intelligence (ICAI)},
  publisher = {CSREA Press},
  address   = {Las Vegas, USA},
  year      = {2012},
}

@inproceedings{Santos2012ICTAI,
  author    = {Santos, Roberto Angelo Fernandes and de Barros, Roberto Souto Maior},
  title     = {{Comparing FBTSeg and NNTree Implementations with Established Ensemble Methods}},
  booktitle = {Proceedings of the IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  publisher = {IEEE},
  address   = {Athens, Greece},
  year      = {2012},
}
```

## Referências citadas pelos artigos

Lista parcial, com os trabalhos que sustentam as escolhas algorítmicas
do FBTSeg:

- **WOLPERT, D. H.** Stacked Generalization. *Neural Networks*, v. 5,
  n. 2, p. 241–259, 1992. — origem do método de **Stacking** usado em
  `combiner_method="stacking"`.

- **THOMAS, L. C.; EDELMAN, D.; CROOK, J.** *Credit Scoring and its
  Applications.* Society for Industrial and Applied Mathematics,
  Philadelphia, USA, 2002. — origem do método **Marginal Odds** usado
  em `combiner_method="marginal_odds"` e referência geral para
  modelagem de risco de crédito.

- **MAIMON, O.; ROKACH, L.** *Decomposition Methodology for Knowledge
  Discovery and Data Mining: Theory and Applications.* Series in
  Machine Perception and Artificial Intelligence, v. 61, World
  Scientific, 2005. — framework genérico de segmentação que o FBTSeg
  estende.

- **MAJI, P.** Efficient Design of Neural Network Tree using a New
  Splitting Criterion. *Neurocomputing*, v. 71, n. 4–6, p. 787–800,
  2008. — método **NNTree** usado como baseline competitivo no paper
  ICAI 2012.

- **CONOVER, W. J.** *Practical Nonparametric Statistics.* 3. ed. John
  Wiley & Sons, 1999. — estatística **KS2** (Kolmogorov–Smirnov de
  duas amostras), citada como métrica `D` válida.

- **FAWCETT, T.** An Introduction to ROC Analysis. *Pattern
  Recognition Letters*, v. 27, n. 8, p. 861–874, 2006. — métrica
  **AUC/ROC**, também citada como `D` no FBTSeg.

- **BLAKE, C.; MERZ, C.** *UCI Repository of Machine Learning
  Databases.* University of California, Irvine, USA. — origem das
  cinco bases usadas em `fbtseg.datasets.article_specs()`.

## Trabalhos adicionais que motivaram a V2

A reescrita deste pacote (V1 `riskseg` → `fbtseg`) também ficou alinhada
às seguintes recomendações de implementação:

- **PEDREGOSA, F.** et al. Scikit-learn: Machine Learning in Python.
  *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.
  — API `fit/predict_proba/predict`, suporte a `clone`, contrato
  binário de `classes_`.

- **HARRIS, C. R.** et al. Array programming with NumPy. *Nature*,
  v. 585, p. 357–362, 2020. — operações vetorizadas que substituem o
  loop por linha da V1 no roteamento da árvore.
