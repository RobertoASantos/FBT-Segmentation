# V2 Checkpoint 001

Snapshot **congelado** da V2 logo após a primeira versão funcional
(2026-05-22). Mantido como backup imutável para o caso de uma evolução
posterior introduzir regressão.

- Testes verde no momento do snapshot: **18/18** em `tests/test_riskseg_v2.py`.
- Benchmark validado: Chess **1.41% err**, Magic **16.07% err** (3-fold CV).
- Performance: predict 50-3000x mais rápido que V1.

Para usar este checkpoint em vez da V2 viva (caso a versão evoluída quebre):

```python
import sys
sys.path.insert(0, "riskseg/v2_checkpoint_001")  # truque para o import direto
# Ou: importe a classe específica do módulo
```

**Não editar este diretório.** Toda evolução vai para `riskseg/v2/`.
