# Replicacao Tabela 1 do paper ICAI 2012

K-fold CV (k=3), max_depth=2

| Dataset | Base | Simple | V2 (FBTSeg) | Delta |
|---|---|---:|---:|---:|
| chess | Linear | 5.94% ± 0.76 | 5.38% ± 0.94 | -0.56 |
| chess | Logistic | 2.41% ± 0.39 | 1.41% ± 0.16 | -1.00 |
| chess | MLP | 3.85% ± 0.33 | 3.85% ± 0.33 | +0.00 |
| german | Linear | 25.90% ± 2.69 | 25.90% ± 2.69 | +0.00 |
| german | Logistic | 26.50% ± 2.97 | 26.50% ± 2.97 | +0.00 |
| german | MLP | 25.60% ± 2.18 | 28.40% ± 1.40 | +2.80 |
| magic | Linear | 21.61% ± 0.37 | 16.93% ± 0.50 | -4.68 |
| magic | Logistic | 20.96% ± 0.31 | 16.07% ± 0.40 | -4.89 |
| magic | MLP | 17.23% ± 0.32 | 17.26% ± 0.73 | +0.03 |
| spambase | Linear | 11.15% ± 0.91 | 10.45% ± 1.42 | -0.70 |
| spambase | Logistic | 7.56% ± 0.92 | 7.82% ± 1.08 | +0.26 |
| spambase | MLP | 8.48% ± 0.26 | 10.65% ± 3.54 | +2.17 |
