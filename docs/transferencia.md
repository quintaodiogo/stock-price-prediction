# Transfer learning e avaliação (Pessoa C)

Pergunta: um modelo pré-treinado em vários ativos prevê melhor um ativo do que um modelo treinado só
com o histórico dele, e o que acontece com um ativo que ele nunca viu?

Alvo principal: **volatilidade futura** (`y_vol_h5`), o único em que a Pessoa B achou sinal
([modelo.md](modelo.md), seção 5). Direção e retorno só entram no backtest, como resultado negativo.

Painel oficial: **`todos_sem_alinhar`** (32 ativos, cada um com o próprio histórico, 17 folds), o mesmo
da Pessoa B ([modelo.md](modelo.md), seção 12.1).

## 1. Como reproduzir

```
python src/minis_to_daily.py                                     # WIN e WDO a partir de data/intraday/*_M1.csv
python src/models/prepare_universe.py --universe todos --no-align
python src/models/run_1b.py --universe todos_sem_alinhar         # baselines e LightGBM da Pessoa B
python src/transfer/run_transfer.py --universe todos_sem_alinhar # zero-shot, fine-tuning e ativo não visto (~30 min)
python src/evaluate.py --universe todos_sem_alinhar              # tabelas e gráficos
python -m pytest tests/test_transfer.py --noconftest
```

Saídas em `reports/results/todos_sem_alinhar/` e `reports/figures/todos_sem_alinhar/`. O `--noconftest`
evita um segfault do LightGBM quando o PyTorch é carregado antes (o `conftest.py` da Pessoa B faz isso).
Os demais testes rodam com `pytest tests --ignore=tests/test_transfer.py`.

Os minis (WIN, WDO) vêm de candles de 1 minuto do MT5, agregados em diário. O WIN vai até 2026-09-25 e o
WDO só até 2026-04-01.

## 2. Os três cenários (`src/transfer/run_transfer.py`)

Mesmo painel, mesmos 17 folds e mesmo LightGBM da Pessoa B.

| Cenário | O que faz |
|---|---|
| zero-shot | Modelo geral treinado em todos os ativos, aplicado a cada ativo sem treino extra. É o mesmo modelo do `lgbm_geral`, então os números são idênticos |
| fine-tuning | Parte do modelo geral e continua o boosting só nas linhas de treino do ativo, com learning rate 0,01 (o geral usa 0,03). Escolhe 0, 10, 25, 50 ou 100 árvores extras numa validação interna (últimos 20% do treino do ativo, com embargo). Para essa escolha, usa uma versão do modelo geral treinada só na parte de ajuste, porque o modelo completo já viu a validação |
| ativo não visto | Para cada ativo, treina o modelo geral **sem o ativo e sem os irmãos próximos** (`src/transfer/siblings.py`: S&P/Nasdaq/Dow, FTSE/DAX, ouro/prata, EUR/GBP, bancos; esconder o Ibovespa esconde também o WIN e as ações da B3, e o USDBRL esconde o WDO) e testa nele |

Os testes (`tests/test_transfer.py`) conferem que o ativo e os irmãos nunca entram no treino do
"não visto" e que as previsões seguem o contrato de dados.

## 3. Avaliação (`src/evaluate.py`)

Compara qualquer `preds_*.parquet` nas **mesmas linhas** (date, asset, fold, target): 17 folds,
2010 a 2026, 123.206 linhas. Gera IC (Spearman), RMSE (e AUC) por modelo, ativo e classe, a tabela final
por ativo e o backtest de direção (long/short sobre a probabilidade de alta, sem custos, mesmas linhas
para todos os modelos). Saídas: `avaliacao_*.csv`, `tabela_final_y_vol_h5.csv`, `backtest_direcao.csv`.

## 4. Resultados (`y_vol_h5`)

Média entre os 17 folds; `vence só ativo` = em quantos folds o IC supera o modelo treinado só no ativo.

| Modelo | IC | IC desvio | RMSE | Vence só ativo | Vence geral |
|---|---|---|---|---|---|
| vol. passada (baseline) | 0,304 | 0,098 | 0,544 | 0% | 0% |
| treino só no ativo | 0,364 | 0,083 | 0,511 | — | 24% |
| geral = zero-shot | 0,378 | 0,088 | 0,503 | 76% | — |
| treino por classe (Pessoa B) | 0,387 | 0,082 | 0,503 | 100% | 76% |
| **fine-tuning** | **0,389** | 0,084 | **0,500** | **100%** | **100%** |
| ativo não visto | 0,372 | 0,088 | 0,505 | 59% | 0% |

Por ativo (32 ativos, IC de todas as linhas de teste do ativo):

- zero-shot supera o treino só no ativo em **26 dos 32**; o ativo não visto, em 23 dos 32;
- fine-tuning supera o treino só no ativo em 30 dos 32, e o zero-shot em 17 dos 32;
- o não visto fica abaixo do zero-shot em 26 dos 32.

Por classe (IC, treino só no ativo → zero-shot → fine-tuning → não visto):

| Classe | só ativo | zero-shot | fine-tuning | não visto |
|---|---|---|---|---|
| ação BR | 0,295 | 0,355 | 0,354 | 0,354 |
| câmbio | 0,415 | 0,431 | 0,442 | 0,429 |
| commodity | 0,344 | 0,350 | 0,363 | 0,345 |
| cripto | 0,321 | 0,398 | 0,394 | 0,397 |
| índice | 0,494 | 0,504 | 0,509 | 0,496 |
| juros | 0,494 | 0,424 | 0,485 | 0,410 |
| mini | 0,181 | 0,316 | 0,306 | 0,313 |

Tabela completa por ativo em `reports/results/todos_sem_alinhar/tabela_final_y_vol_h5.csv`.

## 5. Conclusões

1. **A transferência ajuda na volatilidade.** O modelo geral supera o treino só no ativo na média (IC
   0,378 contra 0,364) e em 26 dos 32 ativos. O ativo não visto fica a apenas 0,005 de IC do modelo que o
   viu, o que sugere que o que se aprende é estrutura comum da volatilidade (persistência, agrupamento),
   não parecença com ativos específicos.
2. **Os ativos com pouco histórico são os que mais ganham.** Minis (0,18 para 0,32) e cripto (0,32 para
   0,40) têm de 6 a 12 anos de dados contra 18 ou mais dos demais. O WIN vai de 0,12 (só no ativo) para
   0,38 (zero-shot), e o não visto fica em 0,38: o modelo prevê o WIN bem sem nunca ter visto o WIN nem o
   Ibovespa.
3. **Fine-tuning ajuda pouco, na média.** +0,011 de IC sobre o zero-shot, menor que o desvio entre folds
   (0,08), apesar de vencer o geral nos 17 folds. O número de árvores extras escolhido foi 0 em 97 dos
   488 pares (ativo, fold), ou seja, em um quinto dos casos o modelo geral já bastava.
4. **O fine-tuning compensa nos ativos que fogem do padrão.** Maior ganho: VIX (0,32 para 0,43, perto dos
   0,45 do treino só no ativo) e a classe de juros (0,42 para 0,49, igualando o treino só no ativo).
   Nesses, o modelo só no ativo chega a superar o geral. Em minis e cripto o fine-tuning piora um pouco
   (WIN: 0,378 para 0,347).
5. **VIX é o limite da generalização.** Não visto, o IC cai para 0,30 (0,45 treinando só nele): a
   dinâmica do VIX não existe nos outros ativos. Vale o mesmo, em menor grau, para a classe de juros
   (0,49 só no ativo, 0,41 não visto).
6. **Direção e retorno: nada que se sustente.** No backtest long/short (5.790 dias, sem custos) o Sharpe
   é 0,37 para a média histórica (só viés de alta), 0,64 para `lgbm_ativo`, 0,67 para `lgbm_geral` e 0,73
   para `lgbm_classe`, com acerto de 51 a 52%. A diferença para a média histórica é pequena e não foi
   testada; os cenários de transferência não foram rodados para esses alvos.

## 6. Ressalvas

- Sem teste estatístico formal; a evidência é a contagem de folds e ativos vencidos e a comparação com o
  desvio entre folds.
- O ganho do fine-tuning sobre o zero-shot (+0,011) é menor que o desvio entre folds. Não dá para dizer
  que existe de forma geral, só nos ativos citados acima.
- Cada ativo tem um número diferente de folds de treino (painel sem alinhar), então a média entre folds
  mistura períodos com ativos diferentes.
- Uma única normalização (`vol_scaled`) e só LightGBM. Fine-tuning da GRU não foi feito, e é onde a
  técnica costuma pesar mais.
- Os irmãos próximos foram definidos à mão (`siblings.py`); ativos ainda correlacionados (por exemplo
  Ibovespa e USDBRL) continuam no treino do "não visto", o que pode inflar o resultado.
- O WDO termina em 2026-04-01, cinco meses antes dos demais.
- O backtest não tem custos nem alavancagem, e o retorno usado é o retorno dividido pela volatilidade.
