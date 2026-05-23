# Instruções para agentes de código neste repositório

## Estrutura

- **`fbtseg/`**: a implementação única e atual. Toda evolução acontece aqui.
- **`tests/`**: testes — devem ficar verdes em qualquer PR.
- **`scripts/`**: scripts de reprodução do paper (benchmark, paper table, ICTAI sintético).
- **`docs/`**: documentação ativa + `docs/reference/` com a tese e PDFs.
- **`artifacts/`**: CSVs/JSONs produzidos pelos scripts.
- **`trash/v1/`**: implementação V1 antiga (RiskSegOptimizer, RiskSegRaiz),
  testes V1, docs V1 — **não voltar**. Está aqui só para auditoria
  histórica.

## Regras de evolução

1. **fbtseg é estritamente binária.** Não adicionar multi-classe — a
   tese é focada em risco dicotômico.
2. **Default de baseline é `LogisticRegression(penalty=None)`**, não
   `C=1.0`. Isso é o que reproduz os números do paper.
3. **Modelos descendentes não recebem a variável de split** (default
   `drop_split_feature_in_children=True`). Vem da tese ("A segmentação
   elimina o efeito da variável/categoria nos modelos seguintes").
4. **`prediction_mode` default é `'leaf'`** — sem `global_stacking`.
   Quem quiser global_stacking paga o custo do OOF (já implementado).
5. **Predição vetorizada por folha.** Loop por linha em Python é
   regressão de performance.
6. **Não voltar a importar de `trash/v1/`.** Se algo lá é útil, traga
   para `fbtseg/` revisado, não importe direto.

## Como rodar

```bash
# pacote completo
python -m pytest tests/

# benchmark fbtseg vs LR
python scripts/run_v2_benchmark.py --datasets chess magic spambase german --n-splits 3

# Replicação da Tabela 1 do paper ICAI 2012
python scripts/run_paper_table_replication.py --datasets chess magic spambase german --n-splits 3

# Validação ICTAI sintético
python scripts/validate_ictai_synthetic.py --datasets 1 2 3 4 --sizes 1000 3000 5000 --n-replications 10

# CLI
python -m fbtseg fit --dataset chess --preset article_uci --output-dir runs/chess
# ou, depois de pip install:
fbtseg fit --dataset chess --preset article_uci --output-dir runs/chess
```

## Antes de aceitar um PR

1. `pytest` verde.
2. Benchmark em pelo menos uma base do paper (Chess ou Magic) não
   regrediu mais que 0.5 p.p. em error_rate.
3. Predição continua vetorizada (sem `iloc[[i]]` ou `for i in range(n)`
   no caminho de predição).
4. Se mudou semântica ou default, atualize `docs/fbtseg.md` e o
   `CHANGELOG.md`.

## Referências

- Tese: `docs/reference/Tese_Roberto_Final_Biblioteca.docx` (Cap. 4)
- ICAI 2012: `artifacts/papers/ICAI2012-rafs-proceed.txt`
- ICTAI 2012: `artifacts/papers/ICTAI2012-Submitted-140-FBTSeg.txt`
- Docs do pacote: `docs/fbtseg.md`
- Replicação Tabela 1: `docs/paper_table_replicated.md`
