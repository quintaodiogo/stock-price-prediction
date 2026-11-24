# Dados e features (Pessoa A)

Como reproduzir, do zero:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_data.py     # data/raw/<NOME>_1d.csv (30 ativos do Yahoo)
.venv/bin/python src/minis_to_daily.py    # WIN e WDO: data/intraday/<NOME>_M1.csv -> data/raw/<NOME>_1d.csv
.venv/bin/python src/validate_data.py     # relatório de qualidade
.venv/bin/python src/features.py          # data/processed/panel.parquet e folds.json (opções: --norm, --universe)
.venv/bin/python src/compare_normalizations.py   # compara as normalizações (ver normalizacao.md)
.venv/bin/python -m pytest tests          # testes anti-vazamento
```

`data/raw/*.csv`, `data/intraday/` e `data/processed/` não vão para o git: são regenerados pelos comandos acima.

## 1. Os 32 ativos

Yahoo Finance via `yfinance` (os 30 primeiros) e, para WIN e WDO, candles de 1 minuto exportados do MT5 e agregados em dias; histórico máximo, preços ajustados, até o último pregão fechado
(o dia corrente é descartado porque o candle ainda está aberto). O catálogo está em `src/assets.py`.

| Classe | Arquivo (ticker) | Início |
|---|---|---|
| Índice | SP500 (^GSPC) | 1927 |
| Índice | NASDAQ (^IXIC) | 1971 |
| Índice | DOWJONES (^DJI) | 1992 |
| Índice | FTSE100 (^FTSE) | 1984 |
| Índice | DAX (^GDAXI) | 1987 |
| Índice | NIKKEI225 (^N225) | 1965 |
| Índice | HANGSENG (^HSI) | 1986 |
| Índice | IBOVESPA (^BVSP) | 1993 |
| Commodity | GOLD (GC=F), SILVER (SI=F), WTI (CL=F), COPPER (HG=F) | 2000 |
| Câmbio | EURUSD, GBPUSD (2003), USDJPY (1996), USDBRL (2003) | |
| Cripto | BTC (2014), ETH (2017) | |
| Juros | US10Y (^TNX, 1962), VIX (^VIX, 1990) | |
| Mini B3 | WIN (Ibovespa, 2020), WDO (dólar, 2020) | ver seção 2 |
| Ação B3 | PETR4, VALE3, ITUB4, BBDC4, BBAS3, ABEV3, WEGE3, B3SA3, SUZB3, RENT3 (ticker `.SA`) | a conferir após o download |

## 2. Validação dos CSVs

`src/validate_data.py` confere datas duplicadas ou fora de ordem, buracos, preços nulos ou não
positivos, High < Low, Close fora de [Low, High], saltos diários suspeitos e volume ausente.

**Sem problema:** nenhuma data duplicada ou fora de ordem, nenhum preço nulo, nenhum High < Low
(exceto 1 linha em GOLD), nenhum split não ajustado detectado (auto_adjust ligado).

**Achados e como cada um é tratado:**

| Achado | Onde | Tratamento |
|---|---|---|
| Buraco de 576 dias (25/08/2004 a 24/03/2006) | USDBRL | O retorno que cruza um buraco maior que 7 dias (3 em cripto) fica ausente. O painel do USDBRL começa em 2004 e o retorno de 24/03/2006 não existe. Não se preenche o passado. |
| Buracos de até 26 dias em ago/2008 | EURUSD, USDJPY, USDBRL | Mesma regra. |
| Preço negativo (-37,63 em 20/04/2020) | WTI | Log não existe; o retorno desse dia e do seguinte ficam ausentes. É um evento real, não erro. |
| Saltos acima de 20% em um dia | SP500 (1987), HANGSENG (1987, 1989), IBOVESPA, WTI (2020), BTC/ETH, SILVER (30/01/2026), COPPER (31/07/2025), USDBRL (2006) | Eventos de mercado reais, mantidos. O de USDBRL em 24/03/2006 é efeito do buraco acima e é descartado. SILVER 30/01/2026 e COPPER 31/07/2025 devem ser conferidos em outra fonte se forem citados. |
| Volume zero ou ausente | Câmbio, VIX, US10Y (100%), índices antigos, alguns futuros | Volume zero vira ausente (nunca 0). `f_volrel` fica ausente nesses casos. |
| Close fora de [Low, High] | Futuros e câmbio (dezenas a centenas de dias) | OHLC do Yahoo é inconsistente nesses ativos; usamos só o Close para retornos. |
| Minis: contrato contínuo `$N` sem ajuste de rolagem | WIN, WDO | A troca de vencimento pode embutir um salto que não é retorno real; maiores retornos diários (WIN 6,6%, WDO 4,5%) são plausíveis, mas a rolagem não foi tratada. Candle diário = primeiro Open, maior High, menor Low, último Close e soma de `<VOL>` do dia. |
| Minis: buracos nos dados | WIN e WDO | Faltam jun a out/2023 (WIN 137 dias, WDO 110), mais buracos menores em 2021 e 2022. Valem as regras de buraco acima. WDO termina em 01/04/2026, WIN em 25/09/2026. |
| Retorno zero em 8,7% dos dias | US10Y | Juros cotados em passos de 0,01; esperado. |

Séries antigas de índices têm High = Low (sem range intradiário). Por isso `f_range` pode ficar ausente,
e o painel não descarta essas linhas.

## 3. Normalização

- **Mesmas datas**: o painel usa só as datas em que todos os ativos do conjunto têm candle (interseção,
  `normalization.align_dates`). O retorno é entre datas consecutivas desse calendário; buraco maior que 7
  dias o invalida (para todos os ativos, inclusive cripto). Conjuntos em `UNIVERSES`: `todos` (32 ativos,
  694 datas, 2021 a 2026), `sem_minis` (30) e `sem_minis_cripto` (28, desde 2008).
- Série base `x_t`: retorno log do Close, `ln(Close_t / Close_{t-1})`. Para `US10Y` e `VIX`, que são níveis,
  a diferença `Close_t - Close_{t-1}`.
- Escala `s_t`: desvio-padrão de `x` nos 252 dias até t (mínimo 63 observações). Só passado.
- `z_t` é a série normalizada que alimenta as features. O padrão é `vol_scaled` (`x_t / s_t`, "retorno /
  volatilidade móvel, sem subtrair a média"); há 12 outras variantes (z-score, min-max, robusta,
  percentil, entre ativos...), descritas e comparadas em [normalizacao.md](normalizacao.md).
- Os alvos de treino `y_*` usam o `s_t` da data da previsão, que é conhecido naquele momento, igual para
  toda normalização. A coluna `scale` está no painel para reverter ao retorno original, e as colunas
  `r_ret_h1` e `r_vol_h5` guardam os alvos crus usados na avaliação.

## 4. Features (`f_*`), todas com dados até t

Calculadas sobre `z_t`. Janelas com buraco aceitam 80% das observações.

| Feature | Fórmula |
|---|---|
| `f_z_lag0` a `f_z_lag4` | `z_{t-k}` |
| `f_z_std5`, `f_z_std21`, `f_z_std63` | desvio-padrão de `z` nas últimas 5, 21 e 63 observações |
| `f_z_mean21`, `f_z_mean63` | média de `z` nas últimas 21 e 63 observações |
| `f_rsi14` | RSI de 14 dias sobre o Close, entre 0 e 1 |
| `f_range` | `(High - Low) / Close` dividido pela média desse range em 252 dias; ausente se High < Low |
| `f_volrel` | `Volume_t` dividido pela média do volume nos últimos 21 dias; ausente sem volume |

## 5. Alvos (`y_*`), usam dados após t

| Alvo | Definição |
|---|---|
| `y_ret_h1` | `x_{t+1} / s_t` |
| `y_ret_h5` | `soma(x_{t+1..t+5}) / (s_t * sqrt(5))` |
| `y_dir_h1` | 1 se `x_{t+1} > 0`, 0 se `< 0`, ausente se `= 0` |
| `y_vol_h5` | desvio-padrão de `x_{t+1..t+5}` dividido por `s_t` |

## 6. Painel e folds

`data/processed/panel.parquet`: uma linha por ativo e data, com `date, asset, asset_class, scale, f_*, y_*, r_*`.
As colunas `scale` e `r_*` são acréscimos ao contrato original da seção 7 do README (só adicionam, não
quebram B e C). As features mudaram de nome (`f_ret_lagk` -> `f_z_lagk`, `f_volN` -> `f_z_stdN`,
`f_momN` -> `f_z_meanN`) e agora são calculadas sobre `z_t`.
Linhas sem escala, sem features ou sem alvo de retorno e volatilidade são removidas (início de cada série e
últimos 5 dias). Todos os ativos usam as mesmas datas.

`data/processed/folds.json`: walk-forward expansivo, teste de um ano; o primeiro teste é um ano depois do
início do painel, mas não antes de 2010 (17 folds de 2010 a 2026 em `sem_minis_cripto`; 5 folds de 2022 a
2026 em `todos`). O último vai até a data final dos dados. O treino termina 11 dias corridos antes do teste e o embargo
cobre os 10 dias entre os dois, mais que os 5 pregões do maior horizonte (`h = 5`).
Treino = `date <= train_end`; teste = `test_start <= date <= test_end`.

## 7. Testes (`tests/test_no_leakage.py`)

- Cortar o futuro dos dados não altera nenhuma feature de datas passadas, para cada uma das 13
  normalizações.
- Alinhamento: todos os ativos ficam nas mesmas datas; min-max e percentil ficam em [0, 1]; o z-score
  entre ativos tem média zero em cada data.
- Trocar os preços futuros por lixo não altera nenhuma feature passada. Verificado por mutação:
  um `shift(-3)` na escala faz este teste falhar.
- Os alvos olham para frente (conferido contra o cálculo manual).
- Preço não positivo e buracos longos não geram retorno; `US10Y` e `VIX` usam diferença.
- Folds: embargo maior que o horizonte, folds em ordem e sem sobreposição de teste.

## 8. Limites conhecidos

- Os ativos têm calendários diferentes (cripto opera 7 dias, o resto não). O painel mantém só as datas
  comuns, então descarta fins de semana de cripto e feriados de qualquer mercado; o histórico fica limitado
  pelo ativo mais curto (ver [normalizacao.md](normalizacao.md)).
- Ativos correlacionados (SP500, NASDAQ, DOWJONES) continuam no painel; o cuidado de retirá-los juntos no
  cenário "ativo não visto" é de quem monta esse experimento.
- Dados do Yahoo não foram conferidos contra outra fonte.
