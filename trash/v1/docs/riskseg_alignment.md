# RiskSeg Thesis Alignment

Referencia principal: `docs/reference/Tese_Roberto_Final_Biblioteca.docx`,
especialmente o Capitulo 4.

> **Atualizacao 2026-05-22:** este documento descreve a aderencia da
> implementacao **V1** (`RiskSegOptimizer`). A V2 (`RiskSegV2`,
> documentada em [riskseg_v2.md](riskseg_v2.md)) resolve os pontos
> listados em "Lacunas a decidir" e fica como o caminho recomendado.
> Veja tambem [thesis_implementation_review.md](thesis_implementation_review.md)
> para a revisao critica completa que motivou a V2.

## Pontos aderentes

- Alvo binario: a implementacao exige `y` binario, coerente com o foco da tese
  em modelos de risco com resposta dicotomica.
- Arvore de modelos: cada no treina um modelo global, testa segmentacoes e
  cria filhos recursivamente ate atingir criterios de parada.
- Screening fatorial: variaveis candidatas sao avaliadas por desenhos
  fatoriais/interativos antes da busca semi-exaustiva de categorias.
- Segmentacao binaria: os splits sao do tipo grupo de categorias contra
  complemento, como a configuracao binaria recomendada na tese.
- Especialistas locais: apos aceitar um split, modelos sao treinados para cada
  segmento.
- Combinacao de escores: a implementacao suporta `stacking` e
  `marginal_odds`, os dois caminhos descritos na tese para alinhamento dos
  escores.
- Numericas categorizadas: variaveis numericas podem virar faixas antes de
  participar da segmentacao.
- Blocos de categorias: `use_grouping` e `max_group_size` implementam a ideia
  de testar agrupamentos de categorias.

## Extensoes praticas

- `prediction_mode="global_stacking"` treina um meta-modelo global sobre as
  folhas. Isso ajuda a calibrar as folhas em uma escala comum, mas vai alem do
  fluxo mais literal descrito no Capitulo 4.
- As metricas incluem `lift`, `precision`, `auc` e uma metrica `combined`
  alem de erro, KS e ROCMIN.
- `screening_variables` permite restringir variaveis de split sem remover
  essas variaveis do espaco usado pelos modelos preditivos.
- `factorial_target_interactions` limita as interacoes fatoriais para controlar
  custo computacional em bases largas.

## Lacunas a decidir

- A tese sugere que a segmentacao reduz a complexidade dos treinamentos
  seguintes ao retirar o efeito da variavel/categoria segmentadora. O codigo
  atual mantem todas as colunas no espaco dos modelos descendentes.
- O parametro equivalente a `rUsaBlocos` e global (`use_grouping`), nao uma
  lista por variavel.
- Com `classifier_tags.multi_class=False`, a configuracao leve usada nos testes
  (`max_depth=1`, `screening_mode="factorial_segment_only"`) passa no
  `check_estimator` completo do sklearn. O metodo permanece deliberadamente
  binario, em coerencia com a tese.
