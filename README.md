# Modelo geral de previsão de preços de ativos

Objetivo: treinar **um único modelo** com vários ativos (índices, commodities, câmbio, cripto, juros) e
reaproveitá-lo em cada ativo via **transfer learning**, em vez de treinar um modelo isolado por ativo.

Pergunta central: *um modelo pré-treinado em muitos ativos prevê melhor um ativo novo do que um modelo
treinado só com o histórico dele?*

```
obter ativos  ->  normalizar  ->  pré-treino (modelo geral)  ->  transfer learning (por ativo)  ->  avaliação
   Pessoa A         Pessoa A            Pessoa B                        Pessoa C                     Pessoa C
```

## 1. Dados

- 20 ativos diários do Yahoo Finance (`yfinance`, histórico máximo, preços ajustados), a baixar para
  `data/raw/<NOME>_1d.csv` (colunas `date, Open, High, Low, Close, Volume`). O download está em
  `src/download_data.py` e a validação em `src/validate_data.py` (ver [docs/dados.md](docs/dados.md)).
  - Índices: S&P 500, Nasdaq, Dow Jones, FTSE 100, DAX, Nikkei 225, Hang Seng, Ibovespa.
  - Commodities: ouro, prata, petróleo WTI, cobre.
  - Câmbio: EUR/USD, USD/JPY, GBP/USD, USD/BRL.
  - Cripto: BTC, ETH.
  - Juros e volatilidade: Treasury 10Y (`^TNX`), VIX.
- Os ativos começam em datas diferentes (S&P 500 em 1927, ETH em 2017). O painel final usa só o período
  em que cada ativo existe; nada de preencher o passado com valores inventados.
- Volume não existe ou é zero em câmbio, VIX e juros. Tratar como ausente, não como zero.

## 2. Normalização

O modelo só consegue ser "geral" se os ativos estiverem na mesma escala e sem tendência de nível.

1. **Preço -> retorno logarítmico**: `r_t = ln(Close_t / Close_{t-1})`. Remove o nível (S&P em 5000 e
   USDBRL em 5 passam a ser comparáveis). Exceções: `US10Y` e `VIX` são níveis, use a diferença.
2. **Padronizar por ativo** com média e desvio calculados **apenas na janela de treino** (ou em janela
   móvel passada). Nunca usar estatísticas do período de teste.
3. **Features** independentes do ativo: retornos defasados, volatilidade realizada (janelas 5/21/63),
   momentum, RSI, range `(High-Low)/Close`, volume relativo à média móvel quando existir.
4. **Identificador do ativo** (embedding ou one-hot da classe) como feature opcional, para o modelo
   distinguir comportamentos por classe.

**Não usar min-max 0–1 sobre o preço**: o mínimo e o máximo vêm da série inteira e vazam o futuro, a
escala quebra quando surge uma nova máxima, e o resultado só reflete a posição na tendência.

**Qual variante escolher ainda está em aberto.** Um teste preliminar com gradient boosting indicou
que retorno cru, z-score expansivo e z-score móvel de 252 dias dão resultados parecidos, mas esse
código e esses dados foram perdidos e o resultado **não é reproduzível**: refazer antes de citá-lo.
Falta testar em redes neurais, onde a escala pesa mais. O painel atual usa a variante "retorno /
volatilidade móvel de 252 dias, sem subtrair a média" (ver [docs/dados.md](docs/dados.md)).

## 3. Alvos

Definir antes de treinar, porque cada um tem dificuldade muito diferente:

| Alvo | Tipo | Métrica principal |
|---|---|---|
| Retorno em `h` dias | regressão | IC (correlação de Spearman), RMSE vs. baseline |
| Direção (sobe/desce) | classificação | AUC, Sharpe do backtest |
| Volatilidade realizada futura | regressão | RMSE, QLIKE vs. baseline |

Comece por `h = 1` e `h = 5`.

## 4. Treino e transfer learning

**Antes de tudo, validar a hipótese.** Que o modelo geral supere os modelos por ativo ainda não foi
demonstrado para este painel (diário, classes de ativos misturadas). A literatura de apoio usa dados
intradiários de ações parecidas entre si, e o sinal em retornos diários é fraco. O primeiro
experimento deve comparar, no mesmo walk-forward: (a) modelo geral, (b) modelo por classe de ativo,
(c) modelo por ativo, (d) baselines (retorno zero, volatilidade passada ou GARCH). Se (a) não ganhar
nos alvos que importam, o plano muda. Ver [docs/referencias.md](docs/referencias.md).

1. **Baselines obrigatórios**: previsão "amanhã = hoje" (retorno zero), média histórica, e um modelo
   treinado só no ativo alvo. Sem bater esses, o resultado não conta.
2. **Modelo geral**: treinar no painel empilhado de vários ativos.
   - Árvores (LightGBM/XGBoost): rápido, bom ponto de partida.
   - Redes sequenciais (LSTM/GRU/Transformer pequeno): janelas de `L` dias de features normalizadas.
3. **Transfer learning**, três cenários, do mais fácil ao mais exigente:
   - **Zero-shot**: aplica o modelo geral ao ativo sem treinar nele.
   - **Fine-tuning**: continua o treino no ativo alvo (redes: congelar camadas iniciais e ajustar as
     finais; árvores: continuar boosting com poucas árvores e learning rate baixo).
   - **Ativo não visto**: treina o modelo geral **sem** o ativo alvo e testa nele. É o teste real de
     generalização.
4. Validar por **classe de ativo** (treinar sem todos os câmbios, testar nos câmbios) para saber se o
   que se transfere é estrutura de mercado ou só parecença entre ativos.

## 5. Avaliação sem vazamento

- **Split temporal**, nunca aleatório. Use *walk-forward*: treina até `t`, testa em `t..t+k`, avança.
- **Embargo** de `h` dias entre treino e teste, porque o alvo olha `h` dias à frente.
- Ativos correlacionados (S&P, Nasdaq, Dow) no treino e no teste geram vazamento de informação de
  mercado. No cenário "ativo não visto", retire também os irmãos próximos.
- Cuidado com **R² alto em preço**: prever o preço de amanhã com o de hoje já dá R² perto de 1. Avalie
  sempre em retornos e compare com o baseline.
- Reportar média e dispersão entre folds, não só o melhor resultado.

## 6. Divisão de atividades

As três frentes se encontram em dois contratos de dados (seção 7), para cada um trabalhar sem esperar
o outro.

### Pessoa A: Dados e features

- Manter `src/download_data.py`; validar os 20 CSVs (datas duplicadas, buracos, splits, zeros).
- Implementar normalização e features em `src/features.py`, sem vazamento.
- Gerar o **painel** `data/processed/panel.parquet` e o arquivo de folds `data/processed/folds.json`
  (walk-forward com embargo).
- Testes em `tests/` garantindo que nenhuma feature usa dado futuro.
- **Pronto quando**: painel gerado, testes passando, documento curto em `docs/` com cada feature e sua
  fórmula.

### Pessoa B: Modelo geral

- Implementar baselines e o modelo geral em `src/models/`, lendo só o painel e os folds.
- Começar por LightGBM/XGBoost, depois uma rede sequencial simples.
- Salvar modelos em `models/` e previsões em `reports/results/` (formato da seção 7).
- Ajustar hiperparâmetros usando apenas dados de treino/validação de cada fold.
- **Pronto quando**: modelo geral treinado em todos os folds, previsões salvas, comparação com os
  baselines.

### Pessoa C: Transfer learning e avaliação

- Implementar zero-shot, fine-tuning e "ativo não visto" em `src/transfer/`, a partir dos modelos da
  Pessoa B.
- Implementar `src/evaluate.py`: métricas, backtest de direção, tabelas e gráficos por ativo e por
  classe de ativo.
- Escrever as conclusões em `docs/` e `reports/`: quando a transferência ajuda, quando não ajuda.
- **Pronto quando**: tabela final comparando baseline, treino só no ativo, zero-shot e fine-tuning,
  para os 20 ativos.

### Ordem de trabalho

| Etapa | A | B | C |
|---|---|---|---|
| 1 | Reescrever o download; painel mínimo (retornos + 3 features) e folds | Baselines sobre o painel mínimo | Esqueleto de `evaluate.py` com métricas; ler os artigos marcados **[ler]** |
| 1b | | Comparar modelo geral, por classe e por ativo (ver seção 4) | Avaliar a comparação e decidir se segue o plano |
| 2 | Conjunto completo de features | Modelo geral de árvores | Zero-shot e "ativo não visto" |
| 3 | Revisão de vazamento e testes | Rede sequencial | Fine-tuning e análise por classe |
| 4 | Documentação dos dados | Tuning final | Relatório final |

A etapa 1 é a prioridade: com um painel mínimo e folds definidos, B e C começam em paralelo.

## 7. Contratos entre as frentes

**Painel** (`data/processed/panel.parquet`, saída de A, entrada de B e C), uma linha por ativo e data:

| Coluna | Descrição |
|---|---|
| `date`, `asset`, `asset_class` | chaves e classe (indice, commodity, cambio, cripto, juros) |
| `scale` | desvio-padrão móvel de 252 dias usado na normalização (para reverter ao retorno original) |
| `f_*` | features já normalizadas |
| `y_ret_h1`, `y_ret_h5`, `y_dir_h1`, `y_vol_h5` | alvos |

**Folds** (`data/processed/folds.json`): lista de `{train_end, embargo_end, test_start, test_end}`.

**Previsões** (saída de B e C em `reports/results/preds_<modelo>.parquet`): `date, asset, fold, target,
y_true, y_pred`. Com esse formato, `evaluate.py` compara qualquer modelo sem mudar código.

## 8. Estrutura do repositório

```
data/raw/          dados brutos baixados
data/processed/    painel e folds
src/               download_data.py, features.py, models/, transfer/, evaluate.py
models/            modelos treinados
notebooks/         exploração
reports/figures/   gráficos
reports/results/   previsões e tabelas de métricas
docs/              decisões, lições aprendidas e referencias.md (artigos e como aplicá-los)
tests/             testes (principalmente anti-vazamento)
```

## 9. Regras do time

- Uma branch por pessoa e PR para `main`; o contrato de dados muda só com aviso aos outros dois.
- Semente fixa e configuração versionada, para qualquer resultado ser reproduzível.
- Registrar em `docs/` cada decisão e cada experimento que deu errado; resultado negativo também conta.
