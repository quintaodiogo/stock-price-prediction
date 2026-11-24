# Comparação de normalizações

Objetivo: descobrir qual normalização dá melhores previsões num modelo geral. O código está em
`src/normalization.py` (variantes) e `src/compare_normalizations.py` (experimento).

```
.venv/bin/python src/compare_normalizations.py --universe todos             # 32 ativos
.venv/bin/python src/compare_normalizations.py --universe sem_minis         # 30 ativos
.venv/bin/python src/compare_normalizations.py --universe sem_minis_cripto  # 28 ativos
```

## 1. Datas em comum

Todos os ativos do conjunto ficam nas **mesmas datas**: só entram as datas em que todos têm candle
(interseção), e os retornos são calculados entre datas consecutivas desse calendário comum. Sem isso, o
retorno de "1 dia" de cripto (7 dias por semana) e o de um índice (5) não seriam comparáveis.

O custo é histórico: a interseção é limitada pelo ativo mais curto.

| Conjunto (`--universe`) | Ativos | Datas comuns | Período do painel | Folds de teste |
|---|---|---|---|---|
| `todos` | 32 | 694 | 2021-06 a 2026-03 | 5 (2022 a 2026) |
| `sem_minis` | 30 | 940 | 2018-04 a 2026-09 | 8 (2019 a 2026) |
| `sem_minis_cripto` | 28 | 2.953 | 2008-09 a 2026-09 | 17 (2010 a 2026) |

Os minis têm um buraco de jun a out/2023 e terminam em datas diferentes (WDO em abril de 2026); por isso
`todos` perde datas. O retorno que cruza um buraco maior que 7 dias fica ausente.

## 2. Variantes

`x` é o retorno log (diferença em US10Y e VIX), `Close` o preço. Janela móvel: 252 pregões, mínimo 63.
Todas usam só dados até t. As features (lags, desvio e média de `z` em janelas de 5/21/63 dias) são
calculadas sobre `z`; RSI, range e volume relativo são iguais em todas.

| Nome | `z_t` | Observação |
|---|---|---|
| `raw` | `x_t` | referência, sem normalizar |
| `vol_scaled` | `x_t / std_252(x)` | a usada no painel padrão; não subtrai a média |
| `vol_scaled_clip` | `vol_scaled` limitado a ±3 | corta caudas |
| `ewma_vol` | `x_t / vol EWMA` (λ ≈ 0,94) | reage mais rápido a mudanças de volatilidade |
| `zscore_rolling` | `(x_t - média_252) / std_252` | z-score móvel |
| `zscore_expanding` | `(x_t - média até t) / std até t` | z-score com toda a história passada |
| `minmax_rolling` | `(x_t - min_252) / (max_252 - min_252)` | min-max móvel, em [0, 1] |
| `minmax_expanding` | idem, com mínimo e máximo desde o início | em [0, 1] |
| `robust_rolling` | `(x_t - mediana_252) / (1,4826 · MAD_252)` | robusto a outliers |
| `rank_rolling` | percentil de `x_t` nos últimos 252 | em (0, 1] |
| `minmax_price_rolling` | posição do log-preço em [min_252, max_252] | normaliza o **nível**, não o retorno |
| `zscore_price_rolling` | z-score do log-preço na janela de 252 | idem |
| `cross_section_z` | z-score entre ativos na mesma data | usa só a própria data; exige ≥ 5 ativos |

Min-max e z-score "globais" (mínimo, máximo e média da série inteira) **não** foram incluídos: vazam o
futuro. As versões acima só olham o passado (o teste `test_every_normalization_ignores_the_future` confere
isso para cada variante).

## 3. Como comparamos

Só as features mudam. Os alvos de treino (`y_*`, divididos pela volatilidade móvel `s_t`), os folds e o
modelo (gradient boosting do scikit-learn, hiperparâmetros fixos) são os mesmos. A avaliação usa os
alvos **crus** (`r_*`), para não premiar uma normalização por mudar a escala do alvo. Walk-forward
expansivo com embargo (ver [dados.md](dados.md)), primeiro teste um ano depois do início do painel.

Métricas, médias entre folds (desvio entre folds em `reports/results/`):
AUC da direção (0,5 = acaso); IC de Spearman do retorno de 1 dia; IC de Spearman da volatilidade de
5 dias; RMSE da log-volatilidade (menor é melhor).

## 4. Resultados

Tabelas completas e por fold: `reports/results/normalizacao_<conjunto>_resumo.csv` e `.csv`.

**Conjunto `sem_minis_cripto` (28 ativos, 17 folds, 2010 a 2026): o mais longo, o mais confiável.**

| Normalização | AUC direção | IC retorno 1d | IC vol 5d | RMSE log-vol |
|---|---|---|---|---|
| raw | 0,5161 | 0,0169 | 0,298 | 0,545 |
| vol_scaled | 0,5133 | 0,0196 | 0,351 | 0,524 |
| vol_scaled_clip | 0,5136 | 0,0188 | **0,353** | **0,524** |
| ewma_vol | 0,5152 | 0,0214 | 0,317 | 0,542 |
| zscore_rolling | 0,5132 | 0,0198 | 0,344 | 0,526 |
| zscore_expanding | 0,5155 | 0,0219 | 0,292 | 0,552 |
| minmax_rolling | 0,5143 | 0,0106 | 0,352 | 0,526 |
| minmax_expanding | 0,5168 | 0,0170 | 0,275 | 0,559 |
| robust_rolling | 0,5175 | **0,0277** | 0,289 | 0,543 |
| rank_rolling | 0,5100 | 0,0077 | 0,304 | 0,538 |
| minmax_price_rolling | 0,5119 | 0,0113 | 0,282 | 0,547 |
| zscore_price_rolling | 0,5123 | 0,0081 | 0,297 | 0,545 |
| cross_section_z | **0,5198** | 0,0243 | 0,279 | 0,549 |

**O que dá para afirmar**

- **Volatilidade (alvo mais previsível): a escala móvel ajuda.** `vol_scaled`, `vol_scaled_clip`,
  `zscore_rolling` e `minmax_rolling` ficam à frente de `raw` e das variantes expansivas, de rank, de
  preço e de entre-ativos: IC cerca de 0,05 maior e RMSE cerca de 0,02 menor. O padrão se repete nos
  três conjuntos (`minmax_rolling` e `vol_scaled_clip` estão entre os melhores em `todos` e `sem_minis` também).
  Limitar a ±3 praticamente não muda nada.
- **Direção e retorno de 1 dia: não dá para separar as normalizações.** A AUC fica entre 0,510 e 0,520
  com desvio entre folds de 0,01 a 0,016, e o IC do retorno entre 0,008 e 0,028 com desvio de cerca de
  0,02. A diferença entre variantes é menor que o ruído entre anos. `cross_section_z` e `robust_rolling`
  aparecem na frente em alguns conjuntos, mas não de forma consistente (`cross_section_z` tem a melhor AUC
  em `sem_minis` e em `sem_minis_cripto`, mas não em `todos`). Não use isso para escolher.
- **Normalizar o nível do preço (min-max e z-score de preço) não ajudou** em nenhum alvo, e o min-max com
  janela expansiva é o pior em volatilidade: o máximo histórico só sobe e a escala fica desatualizada.
- Sobre `raw`: para o gradient boosting, que é invariante a escala por feature, a diferença vem da escala
  *entre ativos e épocas*, e só aparece na volatilidade.

## 5. Limites

- Só gradient boosting. Em redes neurais a escala pesa mais e o resultado pode mudar.
- `todos` tem poucos folds (5) e poucos dados de treino no primeiro; as diferenças ali são fracas.
- Com o conjunto completo (32 ativos) o histórico é de 2021 a 2026. Para os 17 folds de 2010 a 2026 é
  preciso tirar minis e cripto. Essa é uma troca entre mais ativos e mais história.
- Sem correção para múltiplas comparações (13 variantes x 4 métricas): um "primeiro lugar" isolado pode
  ser acaso.
- Os minis são contratos contínuos sem ajuste de rolagem (ver [dados.md](dados.md)).
